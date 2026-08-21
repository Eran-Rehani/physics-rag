from __future__ import annotations

import re

from typer.testing import CliRunner

from physics_rag.cli import app

# Wide terminal so rich does not wrap option names mid-string.
runner = CliRunner(env={"COLUMNS": "200"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Rich styles option names character-by-character, so strip ANSI first."""
    return _ANSI_RE.sub("", text)


def test_app_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    for command in ("ingest", "ask", "stats"):
        assert command in output


def test_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    for option in ("--config", "--force", "--batch-size", "--collection"):
        assert option in output


def test_ask_help() -> None:
    result = runner.invoke(app, ["ask", "--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    for option in ("--config", "--top-k", "--collection", "--show-sources", "--threshold"):
        assert option in output


def test_stats_help() -> None:
    result = runner.invoke(app, ["stats", "--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    for option in ("--config", "--collection"):
        assert option in output
