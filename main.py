"""
Polymarket HFT Agent — Production Orchestrator

Execution cycle (every MAKER_LOOP_INTERVAL seconds):
  A. Fetch YES/NO order books from the CLOB.
  B. Structural-arb check — if YES_ask + NO_ask < $1.00, execute CTF
     splitPosition on-chain and skip the passive maker cycle.
  C. Avellaneda-Stoikov pricing — compute reservation price and optimal spread.
  D. Submit GTC POST_ONLY limit orders via the CLOB (or print DRY RUN notice).
  E. Cancel-before-replace — cancel all resting orders, register fresh quotes.
     Tracked orders are keyed by live exchange order IDs.

CLI flags
  --dry-run   Connect to live APIs, run A-S math, but bypass order submission
              and on-chain execution. Prints DRY RUN notices instead.
  --cycles N  Stop after N cycles (omit to run forever in live mode).
"""

import sys
import time
import argparse
import logging
import structlog
from typing import Optional

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.exceptions import PolyApiException
from py_order_utils.model import POLY_PROXY

from config import load_settings
from scanner import check_structural_arbitrage, fetch_active_markets
from execution import (
    AvellanedaStoikovPricing,
    LiquidityProvisionStrategy,
    PositionLimiter,
    CTFExecutor,
    CTF_CONTRACT_ADDRESS,
    prepare_ctf_split,
)

# ── Module logger ─────────────────────────────────────────────────────────────
log = structlog.get_logger(__name__)

# ── Daemon timing ─────────────────────────────────────────────────────────────
MAKER_LOOP_INTERVAL: float = 5.0      # seconds between full pricing cycles

# ── Arbitrage capital ─────────────────────────────────────────────────────────
ARB_SPLIT_USDC: float = 100.0         # USDC to split per arb opportunity

# ── Avellaneda-Stoikov parameters ─────────────────────────────────────────────
# Calibrate from live fill data; these are conservative starting defaults.
INVENTORY_QTY:      int   = 0
RISK_AVERSION:      float = 0.1       # gamma — moderate inventory skew
BELIEF_VOLATILITY:  float = 0.05      # sigma — conservative vol for stable markets
TIME_TO_EXPIRY:     float = 0.5       # T-t normalised [0, 1]
LIQUIDITY_K:        float = 10.0      # k — moderate market depth

# ── Order sizing ──────────────────────────────────────────────────────────────
ORDER_SIZE_TOKENS: float = 10.0       # YES tokens per side per cycle

# ── Dry-run fallback (not used in live mode) ──────────────────────────────────
_FALLBACK_MID:           float = 0.52
_DRY_PLACEHOLDER_YES           = "0x" + "ee" * 32
_DRY_PLACEHOLDER_NO            = "0x" + "ff" * 32


# ─────────────────────────────────────────────────────────────────────────────
# Logging — structlog JSON (production) / ConsoleRenderer (dry-run / dev)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(json_mode: bool = True) -> None:
    """
    Configure structlog.

    json_mode=True  — one JSON object per line  (production / systemd / VPS)
    json_mode=False — human-readable colours    (local dry-run / development)
    """
    shared: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_mode:
        structlog.configure(
            processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared,
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    else:
        structlog.configure(
            processors=shared + [structlog.dev.ConsoleRenderer()],
            logger_factory=structlog.PrintLoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def initialize_client() -> ClobClient:
    """
    Bootstrap ClobClient through dual-layer authentication.

    L1 — EIP-712 TypedDataSign with signature_type=POLY_PROXY and the
         counterfactual proxy as the funder.
    L2 — HMAC API credentials derived from the L1 nonce. The client is
         upgraded in-place; one object carries both auth layers.

    Falls back to L1-only (scan mode) if L2 derivation fails.
    """
    settings = load_settings()

    log.info(
        "[INIT] Bootstrapping ClobClient | host=%s chain_id=%d env=%s",
        settings.CLOB_API_URL, settings.CHAIN_ID, settings.ENV_MODE,
    )
    log.info("[INIT] Proxy/funder: %s", settings.DEPOSIT_PROXY_ADDRESS)

    client = ClobClient(
        host           = settings.CLOB_API_URL,
        key            = settings.MAINNET_PK,
        chain_id       = settings.CHAIN_ID,
        signature_type = POLY_PROXY,
        funder         = settings.DEPOSIT_PROXY_ADDRESS,
    )
    log.info("[INIT] L1 ready | signer: %s", client.get_address())

    try:
        creds: ApiCreds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        log.info("[INIT] L2 credentials bound | api_key=%s...", creds.api_key[:8])
    except Exception as exc:
        log.warning(
            "[INIT] L2 derivation failed — SCAN-ONLY mode. "
            "Order posting disabled until proxy %s is initialised on-chain. Error: %s",
            settings.DEPOSIT_PROXY_ADDRESS, exc,
        )

    return client


def initialize_ctf_executor(dry_run: bool) -> Optional[CTFExecutor]:
    """
    Initialise the on-chain CTF executor if POLYGON_RPC_URL is set.

    Returns None in dry-run mode or when POLYGON_RPC_URL is blank, which
    disables on-chain arb execution without crashing the bot.
    """
    if dry_run:
        return None

    settings = load_settings()

    if not settings.POLYGON_RPC_URL:
        log.warning(
            "[INIT] POLYGON_RPC_URL not set — on-chain CTF split/merge disabled. "
            "Set POLYGON_RPC_URL in .env to enable live arb execution."
        )
        return None

    try:
        executor = CTFExecutor(settings.POLYGON_RPC_URL, settings.MAINNET_PK)
        log.info("[INIT] CTFExecutor ready | address=%s", executor.address)
        return executor
    except Exception as exc:
        log.error(
            "[INIT] CTFExecutor init failed — on-chain execution disabled: %s", exc
        )
        return None


def resolve_target_market(client: ClobClient) -> str:
    """
    Return the condition ID to trade.

    Priority:
      1. TARGET_CONDITION_ID from .env — validated live against the CLOB.
      2. Auto-select: highest-volume active market from Gamma, confirmed on CLOB.

    Raises RuntimeError if no live market can be found.
    """
    settings      = load_settings()
    configured_id = (settings.TARGET_CONDITION_ID or "").strip()

    if configured_id:
        log.info("[MARKET] Verifying TARGET_CONDITION_ID %s ...", configured_id)
        try:
            market   = client.get_market(configured_id)
            question = (market.get("question") or "(untitled)")[:80]
            log.info("[MARKET] Confirmed: '%s'", question)
            return configured_id
        except PolyApiException as exc:
            err = str(exc).lower()
            if "404" in err or "not found" in err:
                log.warning(
                    "[MARKET] %s returned 404 — market may have resolved. "
                    "Falling back to auto-select.", configured_id,
                )
            else:
                log.warning("[MARKET] CLOB error verifying market: %s. Falling back.", exc)
        except Exception as exc:
            log.warning("[MARKET] Unexpected error: %s. Falling back to auto-select.", exc)
    else:
        log.info("[MARKET] TARGET_CONDITION_ID not set — auto-selecting highest-volume market ...")

    try:
        with httpx.Client(timeout=15.0) as http:
            candidates = fetch_active_markets(http)
    except Exception as exc:
        raise RuntimeError(
            f"Auto-select failed — Gamma API unreachable: {exc}. "
            "Set TARGET_CONDITION_ID in .env."
        ) from exc

    if not candidates:
        raise RuntimeError(
            "Gamma API returned 0 active markets. Set TARGET_CONDITION_ID in .env."
        )

    log.info(
        "[MARKET] Gamma returned %d candidates — probing CLOB ...", len(candidates)
    )

    for candidate in candidates:
        cid = (
            candidate.get("conditionId")
            or candidate.get("condition_id")
            or ""
        )
        if not cid:
            continue

        question = (candidate.get("question") or "(untitled)")[:60]
        vol      = candidate.get("volume24hr") or candidate.get("volume") or 0

        try:
            client.get_market(cid)
        except PolyApiException as exc:
            if "404" in str(exc).lower() or "not found" in str(exc).lower():
                continue
            raise

        log.info(
            "[MARKET] Auto-selected | vol24h=$%,.0f | '%s'",
            float(vol) if vol else 0, question,
        )
        log.info("[MARKET] TIP: pin this market with TARGET_CONDITION_ID=%s", cid)
        return cid

    raise RuntimeError(
        "Auto-select exhausted all Gamma candidates — none are live on the CLOB. "
        "Set TARGET_CONDITION_ID in .env."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_token_ids(
    client: ClobClient,
    condition_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """Return (yes_token_id, no_token_id) for a binary market, or (None, None)."""
    try:
        market = client.get_market(condition_id)
    except Exception as exc:
        log.error("[INIT] Could not fetch market %s: %s", condition_id, exc)
        return None, None

    tokens = market.get("tokens", [])
    if len(tokens) < 2:
        log.error("[INIT] Market %s returned fewer than 2 tokens: %s", condition_id, tokens)
        return None, None

    yes_tok = next((t for t in tokens if "yes" in t.get("outcome", "").lower()), None)
    no_tok  = next((t for t in tokens if "no"  in t.get("outcome", "").lower()), None)

    if not yes_tok or not no_tok:
        log.error("[INIT] Could not identify YES/NO outcomes in %s", condition_id)
        return None, None

    yes_id = yes_tok.get("token_id") or yes_tok.get("tokenID") or None
    no_id  = no_tok.get("token_id")  or no_tok.get("tokenID")  or None
    return yes_id, no_id


def _compute_open_exposure(tracked_orders: dict) -> float:
    """Return total USDC notional (sum of size × price) for all resting orders."""
    return sum(o["size"] * o["price"] for o in tracked_orders.values())


# ─────────────────────────────────────────────────────────────────────────────
# Core trading loop
# ─────────────────────────────────────────────────────────────────────────────

def run_bot(
    market_id: str,
    client: ClobClient,
    ctf_executor: Optional[CTFExecutor],
    *,
    dry_run: bool = False,
    max_cycles: Optional[int] = None,
) -> None:
    """
    Continuous maker daemon for a single binary prediction market.

    Each cycle:
      A. Fetch YES/NO order books.
      B. Structural-arb check — execute splitPosition on-chain if spread exists.
      C. A-S pricing — compute reservation price and optimal spread.
      D. Submit GTC POST_ONLY orders (or print DRY RUN notices).
      E. Cancel-before-replace — cancel all resting orders, register fresh quotes.
         Order tracker keyed by live exchange order IDs.

    Args:
        market_id:    Polymarket bytes32 condition ID.
        client:       ClobClient in L2 mode for order management.
        ctf_executor: Live on-chain executor; None disables arb execution.
        dry_run:      When True, bypass all order submission and on-chain calls.
        max_cycles:   Stop after N cycles; run forever when None.
    """
    lp_strategy = LiquidityProvisionStrategy(AvellanedaStoikovPricing())
    limiter     = PositionLimiter(
        max_position_tokens = 100.0,
        max_exposure_usdc   = 500.0,
    )

    # Keyed by live exchange order IDs (or dry_<side>_<ts> in dry-run mode).
    tracked_orders:   dict[str, dict] = {}
    inventory_tokens: float           = float(INVENTORY_QTY)

    # ── One-time token ID resolution ──────────────────────────────────────────
    log.info("[BOT] Resolving YES/NO token IDs for %s ...", market_id)
    yes_token_id, no_token_id = _resolve_token_ids(client, market_id)

    if not yes_token_id or not no_token_id:
        if dry_run:
            log.warning(
                "[BOT] [DRY RUN] Token resolution failed — using placeholder IDs. "
                "A-S math will run against fallback mid %.4f.", _FALLBACK_MID,
            )
            yes_token_id = _DRY_PLACEHOLDER_YES
            no_token_id  = _DRY_PLACEHOLDER_NO
        else:
            log.critical(
                "[BOT] Cannot start — failed to resolve token IDs for %s. "
                "Verify this is a valid active conditionId on Polygon Mainnet.", market_id,
            )
            return

    log.info("[BOT] YES token: %s", yes_token_id)
    log.info("[BOT] NO  token: %s", no_token_id)

    mode_label = (
        f"DRY-RUN ({max_cycles} cycles)"                    if dry_run
        else (f"LIVE ({max_cycles} cycles)"                  if max_cycles is not None
        else "LIVE (Ctrl-C to stop)")
    )
    log.info(
        "[BOT] Starting | mode=%s | interval=%.1fs | arb_capital=%.0f USDC",
        mode_label, MAKER_LOOP_INTERVAL, ARB_SPLIT_USDC,
    )
    log.info(
        "[BOT] A-S params | q=%d γ=%.3f σ=%.3f T=%.2f k=%.1f",
        INVENTORY_QTY, RISK_AVERSION, BELIEF_VOLATILITY, TIME_TO_EXPIRY, LIQUIDITY_K,
    )

    cycle_count      = 0
    consecutive_404s = 0

    try:
        while True:
            # ── Cycle limit ───────────────────────────────────────────────────
            if max_cycles is not None and cycle_count >= max_cycles:
                log.info("[BOT] Completed %d/%d cycles — stopping.", cycle_count, max_cycles)
                break

            cycle_t0    = time.monotonic()
            cycle_label = (
                f"CYCLE {cycle_count + 1}/{max_cycles}"
                if max_cycles is not None else "CYCLE"
            )
            log.info("[%s] %s", cycle_label, "=" * 56)

            try:
                # ── A: Fetch order books ──────────────────────────────────────
                books_ok = False
                yes_book = no_book = None

                try:
                    yes_book = client.get_order_book(yes_token_id)
                    no_book  = client.get_order_book(no_token_id)
                    books_ok = True
                    log.info(
                        "[A] Books | YES: %d bids / %d asks  |  NO: %d bids / %d asks",
                        len(yes_book.bids or []), len(yes_book.asks or []),
                        len(no_book.bids  or []), len(no_book.asks  or []),
                    )
                except (PolyApiException, httpx.RequestError, OSError) as book_exc:
                    if dry_run:
                        log.warning(
                            "[A] [DRY RUN] Book fetch failed: %s — "
                            "proceeding with fallback mid %.4f.", book_exc, _FALLBACK_MID,
                        )
                    else:
                        raise

                # ── B: Structural-arb check + on-chain execution ──────────────
                if books_ok:
                    spread = check_structural_arbitrage(market_id, yes_book, no_book)

                    if spread is not None:
                        prepare_ctf_split(market_id, ARB_SPLIT_USDC)   # build for logging

                        if dry_run:
                            log.info(
                                "[B] [DRY RUN] ARB DETECTED | spread=%.6f | "
                                "Would execute splitPosition for %.2f USDC.",
                                spread, ARB_SPLIT_USDC,
                            )
                        elif ctf_executor is not None:
                            try:
                                tx_hash = ctf_executor.split_position(market_id, ARB_SPLIT_USDC)
                                log.info(
                                    "[B] ARB EXECUTED | spread=%.6f | "
                                    "splitPosition tx=%s", spread, tx_hash,
                                )
                            except Exception as arb_exc:
                                log.error(
                                    "[B] CTF splitPosition FAILED | spread=%.6f | %s",
                                    spread, arb_exc,
                                )
                        else:
                            log.warning(
                                "[B] ARB DETECTED | spread=%.6f | "
                                "POLYGON_RPC_URL not configured — "
                                "set it in .env to enable on-chain execution.", spread,
                            )

                        log.info("[B] Skipping passive maker this cycle — arb fill pending.")
                        continue

                    log.info("[B] No structural arb — proceeding to passive maker.")

                # ── C: A-S quote computation ──────────────────────────────────
                lp_result = lp_strategy.compute_quotes(
                    yes_book          = yes_book if books_ok else None,
                    inventory_qty     = int(inventory_tokens),
                    risk_aversion     = RISK_AVERSION,
                    belief_volatility = BELIEF_VOLATILITY,
                    time_to_expiry    = TIME_TO_EXPIRY,
                    liquidity_k       = LIQUIDITY_K,
                    fallback_mid      = _FALLBACK_MID if (dry_run or not books_ok) else None,
                )

                if lp_result["mode"] == "skip":
                    log.warning("[C] Empty book, no fallback — skipping cycle.")
                    continue

                mid_disp = lp_result["mid_price"] or 0.0
                bid_disp = f"{lp_result['optimal_bid']:.3f}" if lp_result["optimal_bid"] is not None else "-"
                ask_disp = f"{lp_result['optimal_ask']:.3f}" if lp_result["optimal_ask"] is not None else "-"
                log.info(
                    "[C] LP mode=%-10s | mid=%.4f | bid=%s | ask=%s | spread=%s",
                    lp_result["mode"], mid_disp, bid_disp, ask_disp,
                    f"{lp_result['optimal_spread']:.4f}" if lp_result["optimal_spread"] else "-",
                )

                # ── D: Order submission ───────────────────────────────────────
                open_exposure = _compute_open_exposure(tracked_orders)
                cycle_ts      = int(time.time())

                def _emit_order(side: str, price: float, size: float) -> Optional[str]:
                    """
                    Risk-check then submit one GTC POST_ONLY limit order.

                    Returns the live exchange order ID on success (used as the
                    tracker key), or None if the order was blocked or failed.
                    In dry-run mode returns a synthetic ID without submitting.
                    """
                    if not limiter.check_pre_trade(
                        side                     = side,
                        size                     = size,
                        price                    = price,
                        current_inventory_tokens = inventory_tokens,
                        open_exposure_usdc       = open_exposure,
                    ):
                        log.warning(
                            "[D] PositionLimiter blocked: side=%s size=%.1f price=%.3f",
                            side, size, price,
                        )
                        return None

                    order_args = OrderArgs(
                        token_id = yes_token_id,
                        price    = price,
                        size     = size,
                        side     = side,
                    )

                    if dry_run:
                        dry_id = f"dry_{side.lower()}_{cycle_ts}"
                        print(
                            f"\033[1;96m[DRY RUN]\033[0m  Would submit GTC POST_ONLY: "
                            f"{side:<4}  {size} shares @ {price:.3f}  (id={dry_id})"
                        )
                        return dry_id

                    try:
                        order    = client.create_order(order_args)
                        resp     = client.post_order(order, orderType=OrderType.GTC)
                        order_id = (
                            (resp or {}).get("orderID")
                            or (resp or {}).get("id")
                            or f"live_{side.lower()}_{cycle_ts}"
                        )
                        log.info(
                            "[D] LIVE %s | token=%s...%s | price=%.3f | size=%.1f | id=%s",
                            side, order_args.token_id[:8], order_args.token_id[-6:],
                            price, size, order_id,
                        )
                        return order_id
                    except Exception as order_exc:
                        log.error(
                            "[D] Order FAILED | side=%s price=%.3f size=%.1f | %s",
                            side, price, size, order_exc,
                        )
                        return None

                if dry_run:
                    print()

                bid_id = (
                    _emit_order("BUY",  lp_result["optimal_bid"],  ORDER_SIZE_TOKENS)
                    if lp_result["optimal_bid"]  is not None else None
                )
                ask_id = (
                    _emit_order("SELL", lp_result["optimal_ask"], ORDER_SIZE_TOKENS)
                    if lp_result["optimal_ask"] is not None else None
                )

                if dry_run:
                    print()

                # ── E: Cancel-before-replace ──────────────────────────────────
                if tracked_orders:
                    if not dry_run:
                        try:
                            client.cancel_all()
                            log.info(
                                "[E] cancel_all() — %d stale order(s) cancelled.",
                                len(tracked_orders),
                            )
                        except Exception as cancel_exc:
                            log.warning("[E] cancel_all() failed — proceeding: %s", cancel_exc)
                    else:
                        log.info(
                            "[E] [DRY RUN] Would cancel %d stale order(s).",
                            len(tracked_orders),
                        )
                    tracked_orders.clear()

                # Register newly submitted orders using live exchange IDs
                if bid_id and lp_result["optimal_bid"] is not None:
                    tracked_orders[bid_id] = {
                        "side": "BUY",  "price": lp_result["optimal_bid"],
                        "size": ORDER_SIZE_TOKENS, "token_id": yes_token_id,
                    }
                if ask_id and lp_result["optimal_ask"] is not None:
                    tracked_orders[ask_id] = {
                        "side": "SELL", "price": lp_result["optimal_ask"],
                        "size": ORDER_SIZE_TOKENS, "token_id": yes_token_id,
                    }

                if tracked_orders:
                    log.info(
                        "[E] Tracker | %d order(s) resting: %s",
                        len(tracked_orders), list(tracked_orders.keys()),
                    )

                consecutive_404s = 0

            # ── Per-cycle error handling ──────────────────────────────────────
            except PolyApiException as exc:
                err = str(exc).lower()
                if "429" in err or "rate" in err:
                    log.warning("[CYCLE] HTTP 429 — backing off 2s: %s", exc)
                    time.sleep(2.0)
                elif "404" in err or "not found" in err:
                    consecutive_404s += 1
                    log.warning(
                        "[CYCLE] Market not found (%d/3): %s", consecutive_404s, exc
                    )
                    if consecutive_404s >= 3:
                        log.critical(
                            "[BOT] Market %s appears closed (3 consecutive 404s). "
                            "Restart to select a new market.", market_id,
                        )
                        break
                else:
                    consecutive_404s = 0
                    log.warning("[CYCLE] CLOB API error — skipping cycle: %s", exc)

            except httpx.RequestError as exc:
                log.warning("[CYCLE] Network error — skipping cycle: %s", exc)

            except Exception as exc:
                log.error("[CYCLE] Unexpected error — skipping cycle: %s", exc, exc_info=True)

            # ── Latency profiling + inter-cycle sleep ─────────────────────────
            finally:
                elapsed_ms = (time.monotonic() - cycle_t0) * 1000
                sleep_time = max(0.0, MAKER_LOOP_INTERVAL - elapsed_ms / 1000)
                log.info(
                    "[%s] Elapsed: %.1fms | Sleeping: %.2fs",
                    cycle_label, elapsed_ms, sleep_time,
                )
                cycle_count += 1
                time.sleep(sleep_time)

    # ── Graceful shutdown on Ctrl-C / SIGTERM ─────────────────────────────────
    except KeyboardInterrupt:
        log.info("[SHUTDOWN] Signal received — graceful shutdown ...")

        if tracked_orders:
            if not dry_run:
                try:
                    client.cancel_all()
                    log.info("[SHUTDOWN] All open orders cancelled — book is clean.")
                except Exception as cancel_exc:
                    log.warning("[SHUTDOWN] cancel_all() failed: %s", cancel_exc)
            else:
                log.info(
                    "[SHUTDOWN] [DRY RUN] %d order(s) discarded (no live cancel).",
                    len(tracked_orders),
                )
            tracked_orders.clear()
        else:
            log.info("[SHUTDOWN] No open orders — nothing to cancel.")

        log.info("[SHUTDOWN] Daemon stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure UTF-8 stdout/stderr so A-S notation renders on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Polymarket HFT Agent — Production Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Simulation mode: connect to live APIs, run A-S math,\n"
            "but print notices instead of submitting orders or executing on-chain."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N cycles (omit to run forever).",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        dest="log_json",
        help="Force JSON log output even in dry-run mode (default: colour console).",
    )
    args = parser.parse_args()

    # JSON logging in live mode; readable colours in dry-run / dev mode.
    json_logs = args.log_json or (not args.dry_run)
    _configure_logging(json_mode=json_logs)

    max_cycles = args.cycles

    # ── Safety banner — printed before any log output ─────────────────────────
    if not args.dry_run:
        sys.stdout.write(
            "\n"
            "\033[1;31m" + "!" * 68 + "\n"
            "!!! LIVE TRADING ENABLED - CAPITAL AT RISK !!!\n"
            + "!" * 68 + "\033[0m\n\n"
        )
        sys.stdout.flush()

    log.info("=" * 68)
    log.info("  Polymarket HFT Agent — Production")
    log.info("  CTF    : %s", CTF_CONTRACT_ADDRESS)
    log.info("  Mode   : %s", "DRY-RUN" if args.dry_run else "LIVE")
    if args.cycles is not None:
        log.info("  Cycles : %d", args.cycles)
    log.info("=" * 68)

    clob_client  = initialize_client()
    ctf_executor = initialize_ctf_executor(dry_run=args.dry_run)

    try:
        target_market_id = resolve_target_market(clob_client)
    except RuntimeError as exc:
        log.critical("[STARTUP] %s", exc)
        log.critical(
            "[STARTUP] Cannot start — no valid target market. "
            "Check network or set TARGET_CONDITION_ID in .env."
        )
        sys.exit(1)

    log.info("=" * 68)
    log.info("  Target : %s", target_market_id)
    log.info("=" * 68)

    run_bot(
        target_market_id,
        clob_client,
        ctf_executor,
        dry_run   = args.dry_run,
        max_cycles = max_cycles,
    )
