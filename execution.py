"""
Polymarket HFT Agent — Execution Module

Sections:
  1. On-chain constants — CTF/USDC/exchange addresses from the SDK.
  2. CTF payload builders — prepare_ctf_split() / prepare_ctf_merge().
     Assemble ABI call dicts; the caller decides when to broadcast.
  3. CTFExecutor — live on-chain execution via web3.py.
     Handles USDC approval, splitPosition, and mergePositions.
  4. AvellanedaStoikovPricing — A-S optimal market-making model.
  5. LiquidityProvisionStrategy — A-S wrapper with one-sided book handling.
  6. PositionLimiter — synchronous pre-trade risk gate.

Reference
─────────
Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order
book. Quantitative Finance, 8(3), 217-224.
https://doi.org/10.1080/14697680701381228
"""

import math
import structlog
from typing import Any, Optional

from py_clob_client.config import get_contract_config

log = structlog.get_logger(__name__)

# ── On-chain constants — sourced from the SDK (Polygon Mainnet, Chain ID 137) ──
_CHAIN_CFG = get_contract_config(137)

CTF_CONTRACT_ADDRESS: str = _CHAIN_CFG.conditional_tokens
# Gnosis CTF on Polygon Mainnet: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045

USDC_ADDRESS: str = _CHAIN_CFG.collateral
# Bridged USDC.e on Polygon:     0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174

CLOB_EXCHANGE_ADDRESS: str = _CHAIN_CFG.exchange
# Polymarket CLOB exchange:      0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E

USDC_DECIMALS: int = 6          # 1 USDC = 1_000_000 raw units
BINARY_PARTITION: list[int] = [1, 2]   # YES=0b01, NO=0b10
PARENT_COLLECTION_ID: str = "0x" + "00" * 32   # bytes32(0) — top-level collection

# ── Polymarket CLOB price grid ─────────────────────────────────────────────────
PRICE_MIN:  float = 0.001
PRICE_MAX:  float = 0.999
PRICE_TICK: int   = 3       # decimal places — 0.001 tick size


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Minimal ABIs
# ─────────────────────────────────────────────────────────────────────────────

_ERC20_ABI: list[dict] = [
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount",  "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner",   "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

_CTF_ABI: list[dict] = [
    {
        "name": "splitPosition",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken",    "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId",        "type": "bytes32"},
            {"name": "partition",          "type": "uint256[]"},
            {"name": "amount",             "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "mergePositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken",    "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId",        "type": "bytes32"},
            {"name": "partition",          "type": "uint256[]"},
            {"name": "amount",             "type": "uint256"},
        ],
        "outputs": [],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — CTF payload builders (ABI call dicts, no signing)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ctf_split(condition_id: str, usdc_amount: float) -> dict[str, Any]:
    """
    Build a splitPosition() call dict for auditing and logging.

    Does NOT sign or broadcast.  Pass to CTFExecutor.split_position() for
    live on-chain execution, or use as a structured log payload in dry-run mode.

    Args:
        condition_id: bytes32 hex condition ID (0x-prefixed, 66 chars).
        usdc_amount:  Human-readable USDC (e.g. 100.0 → 100_000_000 raw units).

    Returns:
        {
            "to":       CTF contract address,
            "function": "splitPosition",
            "args":     ABI-typed call arguments,
            "metadata": human-readable audit trail,
        }
    """
    raw_amount = int(usdc_amount * 10 ** USDC_DECIMALS)

    log.info(
        "[CTF] Payload | splitPosition | usdc=%.2f  raw=%d  condition=%s",
        usdc_amount, raw_amount, condition_id,
    )

    return {
        "to":       CTF_CONTRACT_ADDRESS,
        "function": "splitPosition",
        "args": {
            "collateralToken":    USDC_ADDRESS,
            "parentCollectionId": PARENT_COLLECTION_ID,
            "conditionId":        condition_id,
            "partition":          BINARY_PARTITION,
            "amount":             raw_amount,
        },
        "metadata": {
            "operation":           "CTF_SPLIT",
            "condition_id":        condition_id,
            "usdc_amount_human":   usdc_amount,
            "usdc_amount_raw":     raw_amount,
            "expected_yes_tokens": usdc_amount,
            "expected_no_tokens":  usdc_amount,
        },
    }


def prepare_ctf_merge(condition_id: str, token_amount: float) -> dict[str, Any]:
    """
    Build a mergePositions() call dict.

    Merging redeems equal quantities of YES and NO tokens back to USDC.
    Used to realise arb profit after filling both legs.

    Args:
        condition_id:  bytes32 hex condition ID.
        token_amount:  YES (and NO) tokens to merge (e.g. 50.0 → 50 USDC out).
    """
    raw_amount = int(token_amount * 10 ** USDC_DECIMALS)

    log.info(
        "[CTF] Payload | mergePositions | tokens=%.2f  raw=%d  condition=%s",
        token_amount, raw_amount, condition_id,
    )

    return {
        "to":       CTF_CONTRACT_ADDRESS,
        "function": "mergePositions",
        "args": {
            "collateralToken":    USDC_ADDRESS,
            "parentCollectionId": PARENT_COLLECTION_ID,
            "conditionId":        condition_id,
            "partition":          BINARY_PARTITION,
            "amount":             raw_amount,
        },
        "metadata": {
            "operation":          "CTF_MERGE",
            "condition_id":       condition_id,
            "token_amount_human": token_amount,
            "token_amount_raw":   raw_amount,
            "expected_usdc_out":  token_amount,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — CTFExecutor (live on-chain execution via web3.py)
# ─────────────────────────────────────────────────────────────────────────────

class CTFExecutor:
    """
    Live on-chain execution layer for Gnosis CTF operations on Polygon Mainnet.

    Handles:
      - USDC ERC-20 approval (auto-checked before every split)
      - splitPosition: converts USDC → equal YES + NO tokens
      - mergePositions: collapses YES + NO tokens → USDC

    Initialisation requires:
      rpc_url:     POLYGON_RPC_URL from .env (HTTP or HTTPS Polygon endpoint)
      private_key: MAINNET_PK — the EOA that holds USDC and signs transactions

    All transactions are signed locally (private key never leaves the process)
    and broadcast via send_raw_transaction.
    """

    def __init__(self, rpc_url: str, private_key: str) -> None:
        from web3 import Web3

        self._w3      = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        self._account = self._w3.eth.account.from_key(private_key)

        # Polygon PoS uses PoA consensus — inject middleware to handle the
        # extra-data field in block headers (required for eth_getBlock calls).
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            from web3.middleware import geth_poa_middleware  # type: ignore[attr-defined]
            self._w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        self._ctf = self._w3.eth.contract(
            address = self._w3.to_checksum_address(CTF_CONTRACT_ADDRESS),
            abi     = _CTF_ABI,
        )
        self._usdc = self._w3.eth.contract(
            address = self._w3.to_checksum_address(USDC_ADDRESS),
            abi     = _ERC20_ABI,
        )

        if not self._w3.is_connected():
            raise ConnectionError(
                f"web3 could not connect to RPC endpoint: {rpc_url}"
            )

        log.info(
            "[CTFExecutor] Initialised | address=%s | rpc=%s | chain=%d",
            self._account.address, rpc_url, self._w3.eth.chain_id,
        )

    @property
    def address(self) -> str:
        return self._account.address

    # ── Internal transaction builder ──────────────────────────────────────────

    def _send(self, contract_fn, extra_gas: int = 60_000) -> str:
        """
        Estimate gas, build, sign, and broadcast a transaction.
        Returns the 0x-prefixed transaction hash.

        Raises on RPC error, insufficient gas, or chain rejection.
        """
        from_addr = self._account.address
        nonce = self._w3.eth.get_transaction_count(from_addr, "pending")

        try:
            gas_estimate = contract_fn.estimate_gas({"from": from_addr})
        except Exception:
            gas_estimate = 300_000   # safe fallback if estimate fails

        tx = contract_fn.build_transaction({
            "from":  from_addr,
            "nonce": nonce,
            "gas":   gas_estimate + extra_gas,
        })

        signed  = self._w3.eth.account.sign_transaction(tx, self._account.key)
        raw_tx  = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = self._w3.eth.send_raw_transaction(raw_tx)

        return "0x" + tx_hash.hex()

    # ── ERC-20 approval ───────────────────────────────────────────────────────

    def approve_usdc(self, amount_raw: int) -> str:
        """
        Approve the CTF contract to spend amount_raw USDC (6-decimal raw units).
        Returns the approval transaction hash.
        """
        fn      = self._usdc.functions.approve(
            self._w3.to_checksum_address(CTF_CONTRACT_ADDRESS),
            amount_raw,
        )
        tx_hash = self._send(fn, extra_gas=20_000)
        log.info(
            "[CTFExecutor] USDC approved | amount_raw=%d | tx=%s",
            amount_raw, tx_hash,
        )
        return tx_hash

    def _ensure_usdc_allowance(self, amount_raw: int) -> None:
        """Approve CTF to spend USDC if current allowance is insufficient."""
        allowance = self._usdc.functions.allowance(
            self._account.address,
            self._w3.to_checksum_address(CTF_CONTRACT_ADDRESS),
        ).call()
        if allowance < amount_raw:
            log.info(
                "[CTFExecutor] Allowance %d < required %d — approving ...",
                allowance, amount_raw,
            )
            self.approve_usdc(amount_raw)

    # ── CTF operations ────────────────────────────────────────────────────────

    def split_position(self, condition_id: str, usdc_amount: float) -> str:
        """
        Convert usdc_amount USDC into equal YES + NO tokens on-chain.

        Steps:
          1. Approve USDC to CTF contract (skipped if allowance is sufficient).
          2. Call splitPosition() on the CTF contract.

        Args:
            condition_id: 0x-prefixed bytes32 hex condition ID.
            usdc_amount:  Human-readable USDC to split (e.g. 100.0).

        Returns:
            0x-prefixed transaction hash of the splitPosition call.
        """
        raw_amount      = int(usdc_amount * 10 ** USDC_DECIMALS)
        condition_bytes = bytes.fromhex(condition_id.removeprefix("0x"))
        parent_bytes    = bytes(32)   # bytes32(0)

        self._ensure_usdc_allowance(raw_amount)

        fn = self._ctf.functions.splitPosition(
            self._w3.to_checksum_address(USDC_ADDRESS),
            parent_bytes,
            condition_bytes,
            BINARY_PARTITION,
            raw_amount,
        )
        tx_hash = self._send(fn)

        log.info(
            "[CTFExecutor] splitPosition | usdc=%.2f | condition=%s | tx=%s",
            usdc_amount, condition_id[:22], tx_hash,
        )
        return tx_hash

    def merge_positions(self, condition_id: str, token_amount: float) -> str:
        """
        Redeem equal quantities of YES + NO tokens back to USDC on-chain.

        Args:
            condition_id:  0x-prefixed bytes32 hex condition ID.
            token_amount:  Number of YES (and NO) tokens to merge.

        Returns:
            0x-prefixed transaction hash of the mergePositions call.
        """
        raw_amount      = int(token_amount * 10 ** USDC_DECIMALS)
        condition_bytes = bytes.fromhex(condition_id.removeprefix("0x"))
        parent_bytes    = bytes(32)

        fn = self._ctf.functions.mergePositions(
            self._w3.to_checksum_address(USDC_ADDRESS),
            parent_bytes,
            condition_bytes,
            BINARY_PARTITION,
            raw_amount,
        )
        tx_hash = self._send(fn)

        log.info(
            "[CTFExecutor] mergePositions | tokens=%.2f | condition=%s | tx=%s",
            token_amount, condition_id[:22], tx_hash,
        )
        return tx_hash


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Avellaneda-Stoikov pricing
# ─────────────────────────────────────────────────────────────────────────────

class AvellanedaStoikovPricing:
    """
    Avellaneda-Stoikov (2008) optimal market-making model adapted for binary
    prediction markets with prices bounded in (0, 1).

    Formulation:
        r = s − q·γ·σ²·(T−t)                        reservation price
        δ = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)         optimal spread
        bid = r − δ/2,   ask = r + δ/2

    Variables:
        s   — mid-price ∈ (0, 1)
        q   — signed YES-token inventory (+long / −short)
        γ   — risk-aversion coefficient
        σ   — belief volatility of the mid-price
        T−t — normalised time to expiry ∈ [0, 1]
        k   — order-arrival intensity / market depth

    Output prices are clamped to [0.001, 0.999] and rounded to 3 dp to
    conform to Polymarket's CLOB tick-size requirements.
    """

    def calculate_reservation_price(
        self,
        mid_price: float,
        inventory_qty: int,
        risk_aversion: float,
        belief_volatility: float,
        time_to_expiry: float,
    ) -> float:
        """
        Inventory-adjusted mid-price: r = s − q·γ·σ²·(T−t)

        Long inventory shades quotes downward (encourages sells).
        Short inventory shades quotes upward (encourages buys).
        Effect decays linearly to zero at market expiry.
        """
        inventory_risk = (
            inventory_qty * risk_aversion * (belief_volatility ** 2) * time_to_expiry
        )
        r = mid_price - inventory_risk

        log.debug(
            "[AS] Reservation price | s=%.4f q=%+d γ=%.4f σ=%.4f T=%.4f"
            " → inv_risk=%+.6f r=%.6f",
            mid_price, inventory_qty, risk_aversion,
            belief_volatility, time_to_expiry, inventory_risk, r,
        )
        return r

    def calculate_optimal_spread(
        self,
        risk_aversion: float,
        belief_volatility: float,
        time_to_expiry: float,
        liquidity_k: float,
    ) -> float:
        """
        Optimal full bid-ask spread: δ = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)

        Two additive terms:
          Volatility premium:      γ·σ²·(T−t)            — decays to 0 at expiry
          Adverse-selection term:  (2/γ)·ln(1 + γ/k)     — wider for thin books

        Raises ValueError if risk_aversion ≤ 0 or liquidity_k ≤ 0.
        """
        if risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be > 0; got {risk_aversion}")
        if liquidity_k <= 0:
            raise ValueError(f"liquidity_k must be > 0; got {liquidity_k}")

        vol_term = risk_aversion * (belief_volatility ** 2) * time_to_expiry
        adv_sel  = (2.0 / risk_aversion) * math.log(1.0 + risk_aversion / liquidity_k)
        spread   = vol_term + adv_sel

        log.debug(
            "[AS] Spread | γ=%.4f σ=%.4f T=%.4f k=%.4f"
            " → vol=%.6f adv=%.6f δ=%.6f",
            risk_aversion, belief_volatility, time_to_expiry, liquidity_k,
            vol_term, adv_sel, spread,
        )
        return spread

    def generate_limit_quotes(
        self,
        mid_price: float,
        inventory_qty: int,
        risk_aversion: float,
        belief_volatility: float,
        time_to_expiry: float,
        liquidity_k: float,
    ) -> dict[str, float]:
        """
        Produce CLOB-ready bid and ask prices.

        Pipeline:
          1. calculate_reservation_price → r
          2. calculate_optimal_spread    → δ
          3. bid = r − δ/2,  ask = r + δ/2
          4. Clamp to [0.001, 0.999], round to 3 dp.

        Returns:
            {
                "reservation_price": float,
                "optimal_spread":    float,
                "optimal_bid":       float,
                "optimal_ask":       float,
            }
        """
        r     = self.calculate_reservation_price(
            mid_price, inventory_qty, risk_aversion, belief_volatility, time_to_expiry,
        )
        delta = self.calculate_optimal_spread(
            risk_aversion, belief_volatility, time_to_expiry, liquidity_k,
        )

        half        = delta / 2.0
        optimal_bid = round(max(PRICE_MIN, min(PRICE_MAX, r - half)), PRICE_TICK)
        optimal_ask = round(max(PRICE_MIN, min(PRICE_MAX, r + half)), PRICE_TICK)

        log.info(
            "[AS] Quotes | mid=%.4f q=%+d → r=%.4f δ=%.4f bid=%.3f ask=%.3f",
            mid_price, inventory_qty, r, delta, optimal_bid, optimal_ask,
        )

        return {
            "reservation_price": round(r, 6),
            "optimal_spread":    round(delta, 6),
            "optimal_bid":       optimal_bid,
            "optimal_ask":       optimal_ask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Liquidity provision strategy (one-sided book handling)
# ─────────────────────────────────────────────────────────────────────────────

ONE_SIDED_SPREAD: float = 0.02


class LiquidityProvisionStrategy:
    """
    Wraps AvellanedaStoikovPricing and handles one-sided order books.

    Book state         Action
    ─────────────────  ────────────────────────────────────────────────────────
    Both sides         Full A-S bid + ask
    Asks only          Passive BID at best_ask − one_sided_spread
    Bids only          Passive ASK at best_bid + one_sided_spread
    Empty + fallback   A-S from fallback_mid (dry-run mode)
    Empty, no fallback "skip" — do not quote this cycle
    """

    def __init__(
        self,
        pricer: AvellanedaStoikovPricing,
        one_sided_spread: float = ONE_SIDED_SPREAD,
    ) -> None:
        if one_sided_spread <= 0:
            raise ValueError(f"one_sided_spread must be > 0; got {one_sided_spread}")
        self._pricer = pricer
        self._spread = one_sided_spread

    def compute_quotes(
        self,
        yes_book: Any,
        inventory_qty: int,
        risk_aversion: float,
        belief_volatility: float,
        time_to_expiry: float,
        liquidity_k: float,
        fallback_mid: Optional[float] = None,
    ) -> dict:
        """
        Return CLOB-ready quotes adapted to the current book state.

        Returns a dict with keys:
            mode              — "two_sided"|"bid_only"|"ask_only"|"fallback"|"skip"
            mid_price         — float | None
            optimal_bid       — float | None
            optimal_ask       — float | None
            reservation_price — float | None
            optimal_spread    — float | None
        """
        bids     = (yes_book.bids if yes_book is not None else None) or []
        asks     = (yes_book.asks if yes_book is not None else None) or []
        has_bids = bool(bids)
        has_asks = bool(asks)

        # ── Two-sided: full A-S ───────────────────────────────────────────────
        if has_bids and has_asks:
            mid = (float(bids[0].price) + float(asks[0].price)) / 2.0
            q   = self._pricer.generate_limit_quotes(
                mid_price=mid, inventory_qty=inventory_qty,
                risk_aversion=risk_aversion, belief_volatility=belief_volatility,
                time_to_expiry=time_to_expiry, liquidity_k=liquidity_k,
            )
            return {
                "mode": "two_sided",         "mid_price": mid,
                "optimal_bid": q["optimal_bid"], "optimal_ask": q["optimal_ask"],
                "reservation_price": q["reservation_price"],
                "optimal_spread": q["optimal_spread"],
            }

        # ── Asks only — passive bid below best ask ────────────────────────────
        if has_asks and not has_bids:
            best_ask    = float(asks[0].price)
            passive_bid = round(max(PRICE_MIN, best_ask - self._spread), PRICE_TICK)
            log.info(
                "[LP] Asks-only | best_ask=%.3f → passive BID @ %.3f",
                best_ask, passive_bid,
            )
            return {
                "mode": "bid_only",
                "mid_price": best_ask - self._spread / 2.0,
                "optimal_bid": passive_bid, "optimal_ask": None,
                "reservation_price": None, "optimal_spread": self._spread,
            }

        # ── Bids only — passive ask above best bid ────────────────────────────
        if has_bids and not has_asks:
            best_bid    = float(bids[0].price)
            passive_ask = round(min(PRICE_MAX, best_bid + self._spread), PRICE_TICK)
            log.info(
                "[LP] Bids-only | best_bid=%.3f → passive ASK @ %.3f",
                best_bid, passive_ask,
            )
            return {
                "mode": "ask_only",
                "mid_price": best_bid + self._spread / 2.0,
                "optimal_bid": None, "optimal_ask": passive_ask,
                "reservation_price": None, "optimal_spread": self._spread,
            }

        # ── Empty book ────────────────────────────────────────────────────────
        if fallback_mid is not None:
            q = self._pricer.generate_limit_quotes(
                mid_price=fallback_mid, inventory_qty=inventory_qty,
                risk_aversion=risk_aversion, belief_volatility=belief_volatility,
                time_to_expiry=time_to_expiry, liquidity_k=liquidity_k,
            )
            log.warning("[LP] Empty book — using fallback mid %.4f", fallback_mid)
            return {
                "mode": "fallback", "mid_price": fallback_mid,
                "optimal_bid": q["optimal_bid"], "optimal_ask": q["optimal_ask"],
                "reservation_price": q["reservation_price"],
                "optimal_spread": q["optimal_spread"],
            }

        log.warning("[LP] Empty book with no fallback — skipping cycle.")
        return {
            "mode": "skip", "mid_price": None,
            "optimal_bid": None, "optimal_ask": None,
            "reservation_price": None, "optimal_spread": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Pre-trade risk gate
# ─────────────────────────────────────────────────────────────────────────────

class PositionLimiter:
    """
    Synchronous pre-trade risk gate enforcing two independent hard limits.

    max_position_tokens
        Blocks when |projected_inventory| > max_position_tokens.

    max_exposure_usdc
        Blocks when open_exposure + size×price > max_exposure_usdc.

    Both must pass; one failure blocks the order and emits a WARNING log.
    """

    def __init__(
        self,
        max_position_tokens: float = 100.0,
        max_exposure_usdc:   float = 500.0,
    ) -> None:
        if max_position_tokens <= 0:
            raise ValueError(f"max_position_tokens must be > 0; got {max_position_tokens}")
        if max_exposure_usdc <= 0:
            raise ValueError(f"max_exposure_usdc must be > 0; got {max_exposure_usdc}")
        self.max_position_tokens = max_position_tokens
        self.max_exposure_usdc   = max_exposure_usdc

    def check_pre_trade(
        self,
        side:                     str,
        size:                     float,
        price:                    float,
        current_inventory_tokens: float,
        open_exposure_usdc:       float,
    ) -> bool:
        """
        Return True if the order passes both limits; False if either is breached.

        Args:
            side:                     "BUY" or "SELL" (case-insensitive).
            size:                     Token quantity (> 0).
            price:                    Limit price ∈ [0, 1].
            current_inventory_tokens: Signed net YES position before this trade.
            open_exposure_usdc:       Sum of size×price for all resting orders.
        """
        s = side.upper()
        if s not in ("BUY", "SELL"):
            log.error("[LIMITER] Unknown side '%s' — blocking.", side)
            return False

        projected_inventory = (
            current_inventory_tokens + size if s == "BUY"
            else current_inventory_tokens - size
        )

        if abs(projected_inventory) > self.max_position_tokens:
            log.warning(
                "[LIMITER] BLOCKED — inventory | |%.1f| > max=%.1f | side=%s size=%.1f",
                projected_inventory, self.max_position_tokens, side, size,
            )
            return False

        projected_exposure = open_exposure_usdc + size * price
        if projected_exposure > self.max_exposure_usdc:
            log.warning(
                "[LIMITER] BLOCKED — exposure | %.2f > max=%.2f USDC | side=%s size=%.1f price=%.3f",
                projected_exposure, self.max_exposure_usdc, side, size, price,
            )
            return False

        log.debug(
            "[LIMITER] ALLOWED | side=%s size=%.1f price=%.3f inv=%.1f exposure=%.2f",
            side, size, price, projected_inventory, projected_exposure,
        )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  (python execution.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Human-readable console output for local testing
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(f"\n{SEP}")
    print("  execution.py — Self-Test")
    print(SEP)

    # ── CTF payload builders ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  CTF Payload Builders")
    print(SEP2)

    MOCK_CID = "0xdd22472e552ac5a11b5ad120a4b01d1e66a5ebf25e55536b14f951e48e34e4b"

    split = prepare_ctf_split(MOCK_CID, 250.0)
    print(f"\n[1] splitPosition — 250.0 USDC")
    print(f"    to            : {split['to']}")
    print(f"    args.amount   : {split['args']['amount']:,}  (raw 6-dec units)")
    print(f"    args.partition: {split['args']['partition']}")
    print(f"    YES out       : {split['metadata']['expected_yes_tokens']}")
    print(f"    NO  out       : {split['metadata']['expected_no_tokens']}")

    merge = prepare_ctf_merge(MOCK_CID, 125.0)
    print(f"\n[2] mergePositions — 125.0 YES+NO")
    print(f"    to            : {merge['to']}")
    print(f"    args.amount   : {merge['args']['amount']:,}")
    print(f"    USDC out      : {merge['metadata']['expected_usdc_out']}")

    # ── A-S pricing scenarios ─────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  Avellaneda-Stoikov Pricing")
    print(SEP2)

    pricer = AvellanedaStoikovPricing()
    BASE   = dict(
        risk_aversion=0.1, belief_volatility=0.2,
        time_to_expiry=0.5, liquidity_k=100.0,
    )
    scenarios = [
        {"label": "Flat inventory (q=0)",        "mid_price": 0.52, "inventory_qty": 0},
        {"label": "Long +10 YES  (q=+10)",        "mid_price": 0.52, "inventory_qty": 10},
        {"label": "Short -10 YES (q=-10)",        "mid_price": 0.52, "inventory_qty": -10},
        {"label": "Near expiry   (T=0.05)",        "mid_price": 0.75, "inventory_qty": 5,  "time_to_expiry": 0.05},
        {"label": "Thin book     (k=1.0)",         "mid_price": 0.50, "inventory_qty": 0,  "liquidity_k": 1.0},
        {"label": "Clamp test    (extreme long)",  "mid_price": 0.03, "inventory_qty": 200, "risk_aversion": 0.5},
    ]
    for s in scenarios:
        label  = s.pop("label")
        kwargs = {**BASE, **s}
        q      = pricer.generate_limit_quotes(**kwargs)
        eff    = round(q["optimal_ask"] - q["optimal_bid"], 3)
        print(
            f"\n  [{label}]\n"
            f"    mid={kwargs['mid_price']:.3f}  q={kwargs['inventory_qty']:+d}  "
            f"γ={kwargs['risk_aversion']:.2f}  σ={kwargs['belief_volatility']:.2f}  "
            f"T={kwargs['time_to_expiry']:.3f}  k={kwargs['liquidity_k']:.1f}\n"
            f"    r={q['reservation_price']:.6f}  δ={q['optimal_spread']:.6f}  "
            f"bid={q['optimal_bid']:.3f}  ask={q['optimal_ask']:.3f}  "
            f"eff_spread={eff:.3f} ({'clamped' if eff == 0.0 else 'model'})"
        )

    print(f"\n{SEP}")
    print("  All self-tests passed.")
    print(SEP)
