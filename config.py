"""
Polymarket HFT Agent — Configuration

Validates all required environment variables at startup using Pydantic v2.
Raises a clear ValidationError immediately on any malformed value, preventing
silent failures deep in the auth or on-chain execution sequence.

ENV_MODE (OS environment variable — not sourced from any .env file):
  "mainnet"  (default) — loads .env,          CHAIN_ID defaults to 137
  "testnet"            — loads .env.testnet,  enforces CHAIN_ID=80002

Set before launching:
  bash:               ENV_MODE=mainnet python main.py
  PowerShell:         $env:ENV_MODE = "mainnet"; py main.py
"""
import os

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once at import time — determines which .env file Pydantic reads.
_ENV_MODE:        str = os.environ.get("ENV_MODE", "mainnet").lower()
_ENV_FILE:        str = ".env.testnet" if _ENV_MODE == "testnet" else ".env"
_DEFAULT_CHAIN_ID: int = 80002 if _ENV_MODE == "testnet" else 137


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file          = _ENV_FILE,
        env_file_encoding = "utf-8",
        case_sensitive    = True,
    )

    ENV_MODE: str = Field(
        default     = _ENV_MODE,
        description = "Runtime environment: 'mainnet' (Polygon/137) or 'testnet' (Amoy/80002)",
    )

    # ── Layer 1 — EIP-712 TypedDataSign ──────────────────────────────────────
    MAINNET_PK: str = Field(
        ...,
        description = "EOA private key (32 bytes / 64 hex chars) for L1 signing",
    )

    # ── Counterfactual proxy ──────────────────────────────────────────────────
    DEPOSIT_PROXY_ADDRESS: str = Field(
        ...,
        description = "EIP-1167 minimal proxy address; the 'funder' in CLOB parlance",
    )

    # ── Polymarket CLOB ───────────────────────────────────────────────────────
    CLOB_API_URL: str = Field(
        default     = "https://clob.polymarket.com",
        description = "Polymarket CLOB REST API base URL",
    )
    CHAIN_ID: int = Field(
        default     = _DEFAULT_CHAIN_ID,
        description = "137 = Polygon Mainnet  |  80002 = Polygon Amoy Testnet",
    )

    # ── On-chain execution (web3) ─────────────────────────────────────────────
    POLYGON_RPC_URL: str = Field(
        default     = "",
        description = (
            "Polygon Mainnet RPC endpoint for on-chain CTF split/merge calls. "
            "Leave blank to disable on-chain arb execution. "
            "Free options: https://polygon-rpc.com  "
            "or use a private key from Alchemy/Infura for reliability."
        ),
    )

    # ── Target market ─────────────────────────────────────────────────────────
    TARGET_CONDITION_ID: str = Field(
        default     = "",
        description = (
            "32-byte hex condition ID of the market to trade. "
            "Leave blank to auto-select the highest-volume active market."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("MAINNET_PK")
    @classmethod
    def validate_private_key(cls, v: str) -> str:
        raw = v.removeprefix("0x")
        if len(raw) != 64:
            raise ValueError(
                f"MAINNET_PK must be exactly 64 hex chars (32 bytes); got {len(raw)}"
            )
        try:
            int(raw, 16)
        except ValueError:
            raise ValueError("MAINNET_PK contains non-hex characters")
        return v

    @field_validator("DEPOSIT_PROXY_ADDRESS")
    @classmethod
    def validate_proxy_address(cls, v: str) -> str:
        raw = v.removeprefix("0x")
        if len(raw) != 40:
            raise ValueError(
                f"DEPOSIT_PROXY_ADDRESS must be exactly 40 hex chars (20 bytes); got {len(raw)}"
            )
        try:
            int(raw, 16)
        except ValueError:
            raise ValueError("DEPOSIT_PROXY_ADDRESS contains non-hex characters")
        return v

    @field_validator("TARGET_CONDITION_ID")
    @classmethod
    def validate_condition_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return v
        raw = v.removeprefix("0x")
        if len(raw) != 64:
            raise ValueError(
                f"TARGET_CONDITION_ID must be 64 hex chars; got {len(raw)}. "
                "Leave blank for auto-select."
            )
        try:
            int(raw, 16)
        except ValueError:
            raise ValueError("TARGET_CONDITION_ID contains non-hex characters")
        return v

    @field_validator("POLYGON_RPC_URL")
    @classmethod
    def validate_rpc_url(cls, v: str) -> str:
        v = v.strip()
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"POLYGON_RPC_URL must be an HTTP/HTTPS URL; got: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def enforce_testnet_chain_id(self) -> "Settings":
        if self.ENV_MODE.lower() == "testnet" and self.CHAIN_ID != 80002:
            raise ValueError(
                f"ENV_MODE=testnet requires CHAIN_ID=80002; "
                f"got CHAIN_ID={self.CHAIN_ID}. Check {_ENV_FILE}."
            )
        return self


def load_settings() -> Settings:
    return Settings()
