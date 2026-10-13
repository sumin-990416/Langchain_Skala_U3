"""Orchestrator-Workers 실행 진입점. 기존 투자 성향 모드는 legacy_main.py."""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
