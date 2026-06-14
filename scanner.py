"""
Polymarket HFT Agent — Market Scanner

Responsibilities:
  fetch_active_markets()       — Query Gamma API for top binary markets by volume.
  check_structural_arbitrage() — Detect YES_ask + NO_ask < $1.00 on live books.
  _scan_cycle()                — Bulk-fetch all order books in one HTTP round-trip.
  run_scanner()                — Standalone arb-detection daemon (optional utility).

The two functions imported by main.py are fetch_active_markets and
check_structural_arbitrage; the daemon loop is provided for independent use.
"""

import time
import structlog
from typing import Optional

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BookParams, OrderBookSummary
from py_clob_client.exceptions import PolyApiException

from config import load_settings

# ── Constants ─────────────────────────────────────────────────────────────────
GAMMA_API_URL:            str   = "https://gamma-api.polymarket.com"
CLOB_MAX_BOOKS_PER_REQUEST: int = 100
MARKET_FETCH_LIMIT:         int = 50
POLL_INTERVAL_SECONDS:    float = 1.0
BACKOFF_SECONDS:          float = 2.0

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — consumed by main.py
# ─────────────────────────────────────────────────────────────────────────────

def fetch_active_markets(http: httpx.Client) -> list[dict]:
    """
    Return the top MARKET_FETCH_LIMIT active binary markets from the Gamma API,
    ordered by 24 h volume descending.

    Each dict contains at minimum:
        conditionId : str   — CLOB condition ID
        tokens      : list  — [{"token_id": str, "outcome": "Yes"|"No"}, ...]
        question    : str   — human-readable market description
        volume24hr  : float — 24-hour trading volume in USD
    """
    resp = http.get(
        f"{GAMMA_API_URL}/markets",
        params={
            "active":    "true",
            "closed":    "false",
            "limit":     MARKET_FETCH_LIMIT,
            "order":     "volume24hr",
            "ascending": "false",
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    # Gamma returns either a bare list or {"data": [...]} depending on version
    return payload if isinstance(payload, list) else payload.get("data", [])


def check_structural_arbitrage(
    market_id: str,
    yes_book: OrderBookSummary,
    no_book: OrderBookSummary,
) -> Optional[float]:
    """
    Detect a structural arbitrage opportunity on a binary market.

    Since exactly one of YES or NO resolves to $1.00, buying both legs at a
    combined cost below $1.00 guarantees a risk-free profit equal to the spread.

    Condition:  YES_ask + NO_ask < $1.00
    Profit:     1.00 - (YES_ask + NO_ask)

    Returns the guaranteed profit per $1 pair when arb exists, else None.
    Returns None silently when either book side is empty (not actionable).
    """
    yes_ask = _best_ask(yes_book)
    no_ask  = _best_ask(no_book)

    if yes_ask is None or no_ask is None:
        return None

    total_cost = yes_ask + no_ask

    if total_cost < 1.00:
        spread = round(1.00 - total_cost, 6)
        log.warning(
            "[ARB] Structural arbitrage detected | market=%s | "
            "yes_ask=%.4f | no_ask=%.4f | total=%.4f | profit=%.6f",
            market_id, yes_ask, no_ask, total_cost, spread,
        )
        return spread

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_ask(book: OrderBookSummary) -> Optional[float]:
    """Return the lowest ask price from a book, or None if the ask side is empty."""
    if not book or not book.asks:
        return None
    return float(book.asks[0].price)


def _extract_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Extract YES and NO ERC-1155 token IDs from a Gamma API market dict.

    Handles both snake_case (token_id) and camelCase (tokenID) from older payloads.
    Returns (yes_token_id, no_token_id) or (None, None) on malformed data.
    """
    tokens = market.get("tokens", [])
    if len(tokens) < 2:
        return None, None

    yes_tok = next((t for t in tokens if "yes" in t.get("outcome", "").lower()), None)
    no_tok  = next((t for t in tokens if "no"  in t.get("outcome", "").lower()), None)

    if not yes_tok or not no_tok:
        return None, None

    yes_id = yes_tok.get("token_id") or yes_tok.get("tokenID") or None
    no_id  = no_tok.get("token_id")  or no_tok.get("tokenID")  or None
    return yes_id, no_id


def _scan_cycle(markets: list[dict], clob: ClobClient) -> int:
    """
    Bulk-fetch order books for all markets in one HTTP call, then run
    check_structural_arbitrage() on each YES/NO pair.

    Uses get_order_books() (single POST) rather than one GET per token,
    reducing per-cycle HTTP round-trips from 2*N to 1.

    Returns the number of arb opportunities found this cycle.
    """
    valid_markets: list[dict]       = []
    bulk_params:   list[BookParams] = []

    for market in markets:
        yes_id, no_id = _extract_token_ids(market)
        if not yes_id or not no_id:
            continue
        valid_markets.append(market)
        bulk_params.append(BookParams(token_id=yes_id))
        bulk_params.append(BookParams(token_id=no_id))

    if not valid_markets:
        return 0

    all_books: list[OrderBookSummary] = clob.get_order_books(
        bulk_params[:CLOB_MAX_BOOKS_PER_REQUEST]
    )

    found = 0
    for i, market in enumerate(valid_markets):
        if 2 * i + 1 >= len(all_books):
            break

        market_id = market.get("conditionId") or market.get("id", f"market-{i}")
        if check_structural_arbitrage(market_id, all_books[2 * i], all_books[2 * i + 1]) is not None:
            found += 1

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Standalone arb-detection daemon (optional — not used by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_scanner() -> None:
    """
    Continuously poll all active markets and log arb opportunities.
    Runs indefinitely until interrupted with Ctrl-C.
    """
    settings = load_settings()
    clob = ClobClient(host=settings.CLOB_API_URL, chain_id=settings.CHAIN_ID)

    log.info(
        "[SCANNER] Started | clob=%s | gamma=%s | interval=%.1fs",
        settings.CLOB_API_URL, GAMMA_API_URL, POLL_INTERVAL_SECONDS,
    )

    with httpx.Client() as http:
        while True:
            try:
                markets = fetch_active_markets(http)
                log.info(
                    "[SCANNER] Fetched %d markets — scanning order books ...",
                    len(markets),
                )

                found = _scan_cycle(markets, clob)

                if found == 0:
                    log.info("[SCANNER] No structural arb this cycle.")
                else:
                    log.warning(
                        "[SCANNER] %d arb opportunit%s flagged.",
                        found, "y" if found == 1 else "ies",
                    )

            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                log.warning(
                    "[SCANNER] HTTP %d from %s — backing off %.1fs",
                    code, exc.request.url.host, BACKOFF_SECONDS,
                )
                time.sleep(BACKOFF_SECONDS)
                continue

            except (httpx.RequestError, PolyApiException, OSError) as exc:
                log.warning("[SCANNER] Network/API error — backing off %.1fs: %s", BACKOFF_SECONDS, exc)
                time.sleep(BACKOFF_SECONDS)
                continue

            except Exception as exc:
                log.error("[SCANNER] Unexpected error — backing off: %s", exc, exc_info=True)
                time.sleep(BACKOFF_SECONDS)
                continue

            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Human-readable console output for standalone scanner use
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

    try:
        run_scanner()
    except KeyboardInterrupt:
        log.info("[SCANNER] Stopped by operator (Ctrl-C).")
