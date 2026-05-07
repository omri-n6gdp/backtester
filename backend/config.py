"""Configuration loader. Reads .env from the project root and exposes a settings singleton."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of the `backend/` directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    t212_demo_key: str
    t212_live_key: str  # intentionally allowed to be empty until Phase 13

    @classmethod
    def from_env(cls) -> "Settings":
        groq = os.environ.get("GROQ_API_KEY", "").strip()
        t212_demo = os.environ.get("T212_DEMO_KEY", "").strip()
        t212_live = os.environ.get("T212_LIVE_KEY", "").strip()

        if not groq:
            raise RuntimeError(
                f"GROQ_API_KEY missing or empty in {ENV_PATH}. "
                "Get a free key at console.groq.com."
            )
        if not t212_demo:
            raise RuntimeError(
                f"T212_DEMO_KEY missing or empty in {ENV_PATH}. "
                "Generate one in the Trading 212 app under Settings -> API (Beta) in Practice mode."
            )
        # T212_LIVE_KEY stays empty until Phase 13.

        return cls(
            groq_api_key=groq,
            t212_demo_key=t212_demo,
            t212_live_key=t212_live,
        )


settings = Settings.from_env()
