"""Tests for the interactive REPL."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covenant.core.agent_loop import AgentLoop, StepResult
from covenant.core.executive import Action
from covenant.repl import (
    COMMANDS,
    EXIT_WORDS,
    CovenantREPL,
    _display_result,
    _make_completer,
    start_repl,
)


# --- Unit tests: _display_result ---


class TestDisplayResult:
    def test_respond_action(self, capsys):
        result = StepResult(action=Action.RESPOND, payload={"text": "Hello!"})
        _display_result(result)
        captured = capsys.readouterr()
        assert "Hello!" in captured.out

    def test_ask_action(self, capsys):
        result = StepResult(action=Action.ASK, payload={"text": "What next?"})
        _display_result(result)
        captured = capsys.readouterr()
        assert "What next?" in captured.out

    def test_tool_action(self, capsys):
        result = StepResult(
            action=Action.TOOL,
            payload={"tool": "calculator", "args": {"expr": "2+2"}},
        )
        _display_result(result)
        captured = capsys.readouterr()
        assert "calculator" in captured.out

    def test_stop_action(self, capsys):
        result = StepResult(action=Action.STOP, payload={})
        _display_result(result)
        captured = capsys.readouterr()
        assert "session ended" in captured.out

    def test_blocked_result(self, capsys):
        result = StepResult(
            action=Action.TOOL,
            payload={"tool": "rm"},
            blocked=True,
            block_reason="unsafe operation",
        )
        _display_result(result)
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.out
        assert "unsafe operation" in captured.out

    def test_reflect_action(self, capsys):
        result = StepResult(action=Action.REFLECT, payload={})
        _display_result(result)
        captured = capsys.readouterr()
        assert "reflect" in captured.out

    def test_retrieve_more_action(self, capsys):
        result = StepResult(action=Action.RETRIEVE_MORE, payload={})
        _display_result(result)
        captured = capsys.readouterr()
        assert "retrieve_more" in captured.out


# --- Unit tests: _make_completer ---


class TestMakeCompleter:
    def test_returns_word_completer(self):
        from prompt_toolkit.completion import WordCompleter

        completer = _make_completer()
        assert isinstance(completer, WordCompleter)

    def test_includes_all_commands(self):
        completer = _make_completer()
        for cmd in COMMANDS:
            assert cmd in completer.words

    def test_includes_exit_words(self):
        completer = _make_completer()
        assert "exit" in completer.words
        assert "quit" in completer.words


# --- Unit tests: commands ---


class TestCommands:
    def setup_method(self):
        self.repl = CovenantREPL(llm=False, max_steps=5)

    @pytest.mark.asyncio
    async def test_quit_returns_true(self):
        assert await self.repl._handle_command("/quit") is True

    @pytest.mark.asyncio
    async def test_exit_returns_true(self):
        assert await self.repl._handle_command("/exit") is True

    @pytest.mark.asyncio
    async def test_bare_exit_returns_true(self):
        assert await self.repl._handle_command("exit") is True

    @pytest.mark.asyncio
    async def test_bare_quit_returns_true(self):
        assert await self.repl._handle_command("quit") is True

    @pytest.mark.asyncio
    async def test_help_returns_false(self, capsys):
        assert await self.repl._handle_command("/help") is False
        captured = capsys.readouterr()
        assert "/help" in captured.out

    @pytest.mark.asyncio
    async def test_clear_without_agent(self, capsys):
        assert await self.repl._handle_command("/clear") is False
        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_clear_with_agent(self):
        self.repl._agent = AgentLoop()
        self.repl._agent.history.append(
            StepResult(action=Action.RESPOND, payload={"text": "hi"})
        )
        assert await self.repl._handle_command("/clear") is False
        assert len(self.repl._agent.history) == 0

    @pytest.mark.asyncio
    async def test_status_no_agent(self, capsys):
        assert await self.repl._handle_command("/status") is False
        captured = capsys.readouterr()
        assert "No active cortex" in captured.out

    @pytest.mark.asyncio
    async def test_status_with_agent(self, capsys):
        self.repl._agent = AgentLoop()
        self.repl._agent.wm.goal = "test goal"
        self.repl._agent.wm.budget = 5
        assert await self.repl._handle_command("/status") is False
        captured = capsys.readouterr()
        assert "test goal" in captured.out
        assert "5" in captured.out

    @pytest.mark.asyncio
    async def test_reset_clears_agent(self, capsys):
        self.repl._agent = AgentLoop()
        assert await self.repl._handle_command("/reset") is False
        assert self.repl._agent is None
        captured = capsys.readouterr()
        assert "reset" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_unknown_command(self, capsys):
        assert await self.repl._handle_command("/foo") is False
        captured = capsys.readouterr()
        assert "unknown command" in captured.out.lower()


# --- Integration: _run_agent_turn with rule-based agent ---


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_run_agent_turn_responds(self):
        """Rule-based agent immediately responds, so a single step should suffice."""
        repl = CovenantREPL(llm=False, max_steps=5)
        results = await repl._run_agent_turn("hello")
        assert len(results) >= 1
        last = results[-1]
        assert last.action in {Action.RESPOND, Action.ASK, Action.STOP}

    @pytest.mark.asyncio
    async def test_run_agent_turn_creates_agent_lazily(self):
        """Agent should be created on first turn."""
        repl = CovenantREPL(llm=False, max_steps=3)
        assert repl._agent is None
        await repl._run_agent_turn("test")
        assert repl._agent is not None

    @pytest.mark.asyncio
    async def test_run_agent_turn_respects_max_steps(self):
        """Should not exceed max_steps results."""
        repl = CovenantREPL(llm=False, max_steps=2)
        results = await repl._run_agent_turn("compute something")
        assert len(results) <= 2


# --- CLI integration: REPL is default command ---


class TestCLIIntegration:
    def test_no_args_invokes_repl(self):
        """Running `covenant` with no args should invoke start_repl."""
        from click.testing import CliRunner

        from covenant.cli import main

        runner = CliRunner()
        with patch("covenant.repl.start_repl") as mock_repl:
            result = runner.invoke(main, [])
            mock_repl.assert_called_once_with(llm=False, max_steps=10)

    def test_subcommands_still_work(self):
        """Existing subcommands should still function."""
        from click.testing import CliRunner

        from covenant.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["hello"])
        assert result.exit_code == 0
        assert "Hello from Covenant" in result.output

    def test_repl_with_llm_flag(self):
        """--llm flag should be passed to start_repl."""
        from click.testing import CliRunner

        from covenant.cli import main

        runner = CliRunner()
        with patch("covenant.repl.start_repl") as mock_repl:
            result = runner.invoke(main, ["--llm", "--max-steps", "5"])
            mock_repl.assert_called_once_with(llm=True, max_steps=5)

    def test_start_repl_entry_point(self):
        """start_repl should create REPL and call asyncio.run."""
        with patch("covenant.repl.asyncio.run") as mock_run:
            start_repl(llm=False, max_steps=3)
            mock_run.assert_called_once()
