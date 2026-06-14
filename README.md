# Polymarket HFT Agent

Production market-making daemon for Polymarket binary prediction markets.

**Stack:** Avellaneda-Stoikov (2008) optimal spread · Gnosis CTF on-chain arb (web3.py) · Polymarket CLOB GTC POST_ONLY orders · structlog JSON logging · systemd hardened service.

---

## The three commands

### 1 — Build the workspace on your local machine

```bash
python build_workspace.py --dir ./polymarket-hft
```

`build_workspace.py` runs `audit_workspace()` before writing a single byte. The audit verifies:
- All 6 required packages pinned in `requirements.txt` (Python 3.12-compatible)
- All `Field()` definitions and validators present in `config.py`
- All idempotency guards, permission matrix, and secure-user flags in `deploy.sh`
- Zero stub markers; live `create_order`, `cancel_all`, `split_position`, `ProcessorFormatter` wired in `main.py`

If any check fails the script exits before touching your disk. When all 40 checks are green, 6 LF-encoded files land in `./polymarket-hft/`.

---

### 2 — Copy files to your VPS

```bash
scp -r ./polymarket-hft root@YOUR_VPS_IP:/tmp/hft_deploy
```

Replace `YOUR_VPS_IP` with your server address. The remote directory is created automatically.

---

### 3 — Run the deployment script on the VPS

```bash
ssh root@YOUR_VPS_IP "cd /tmp/hft_deploy && bash deploy.sh"
```

`deploy.sh` runs 10 idempotent steps — safe to execute 100 times on the same server:

| Step | What it does |
|------|-------------|
| 0 | Root privilege check |
| 1 | Pre-flight: verify all 5 source files present |
| 2 | `apt-get install` Python 3, venv, build tools |
| 3 | Create `polymarket-bot` system user — no home dir, no login shell |
| 4 | Create `/opt/hft_agent/` (root:root 755) + `/opt/hft_agent/logs/` (polymarket-bot 750) |
| 5 | Install source files at 640 permissions (owner rw, group r, no world) |
| 6 | Create or reuse `.venv`; `pip install -r requirements.txt` |
| 7 | Write `.env` template — **skipped if `.env` already exists** (secrets never overwritten) |
| 8 | Write `/etc/systemd/system/hft_agent.service` |
| 9 | `systemctl daemon-reload && systemctl enable hft_agent` |
| 10 | 1-cycle `--dry-run` smoke test as `polymarket-bot` |

The service is **enabled but not started**. Populate `.env` with credentials before starting.

---

## After deployment: fill in credentials

```bash
ssh root@YOUR_VPS_IP "nano /opt/hft_agent/.env"
```

Required:

```env
MAINNET_PK=<64 hex chars — EOA private key, no 0x prefix>
DEPOSIT_PROXY_ADDRESS=<40 hex chars — Polymarket deposit proxy, no 0x prefix>
POLYGON_RPC_URL=https://polygon-rpc.com
```

Optional:

```env
TARGET_CONDITION_ID=   # leave blank to auto-select highest-volume market
CHAIN_ID=137
ENV_MODE=mainnet
```

---

## Start / monitor / stop

```bash
sudo systemctl start hft_agent          # go live
sudo journalctl -u hft_agent -f         # stream JSON logs
sudo systemctl status hft_agent         # service health
sudo systemctl stop hft_agent           # graceful stop (cancels open orders)
```

---

## Local dry-run

```bash
cd ./polymarket-hft
pip install -r requirements.txt
# create .env with your credentials
python main.py --dry-run --cycles 3
```

Connects to live CLOB and runs full A-S math; prints `[DRY RUN]` notices instead of submitting orders or executing on-chain.

---

## Safety

Live mode prints a bold-red banner before any API call:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! LIVE TRADING ENABLED - CAPITAL AT RISK !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Orders are submitted immediately on start. Verify credentials and risk limits before running.

---

## Project files

| File | Purpose |
|------|---------|
| `build_workspace.py` | Single-file workspace builder + auditor |
| `main.py` | Daemon orchestrator — A-S loop, order tracking, CTF arb |
| `execution.py` | CTFExecutor (web3), A-S pricing, PositionLimiter |
| `config.py` | Pydantic v2 settings — validates `.env` at startup |
| `scanner.py` | Gamma API scanner, structural arb detector |
| `requirements.txt` | Pinned production dependencies (Python 3.11/3.12) |
| `deploy.sh` | Idempotent VPS deployment automation (10 steps) |
