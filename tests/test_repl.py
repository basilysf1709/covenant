"""Tests for the interactive REPL."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covenant.core.agent_loop import AgentLoop, StepResult
from covenant.core.executive import Action
from covenant.repl import (
    SLASH_COMMANDS,
    CovenantREPL,
    format_step_result,
    make_completer,
    start_repl,
)


# --- Unit tests: format_step_result ---


class TestFormatStepResult:
    def test_respond_action(self):
        result = StepResult(action=Action.RESPOND, payload={"text": "Hello!"})
        text, style = format_step_result(result)
        assert text == "Hello!"
        assert style == "class:response"

    def test_ask_action(self):
        result = StepResult(action=Action.ASK, payload={"text": "What next?"})
        text, style = format_step_result(result)
        assert text == "? What next?"
        assert style == "class:info"

    def test_tool_action(self):
        result = StepResult(
            action=Action.TOOL,
            payload={"tool": "calculator", "args": {"expr": "2+2"}},
        )
        text, style = format_step_result(result)
        assert "calculator" in text
        assert style == "class:tool"

    def test_stop_action(self):
        result = StepResult(action=Action.STOP, payload={})
        text, style = format_step_result(result)
        assert "stopped" in text.lower()
        assert style == "class:info"

    def test_blocked_result(self):
        result = StepResult(
            action=Action.TOOL,
            payload={"tool": "rm"},
            blocked=True,
            block_reason="unsafe operation",
        )
        text, style = format_step_result(result)
        assert "BLOCKED" in text
        assert "unsafe operation" in text
        assert style == "class:error"

    def test_reflect_action(self):
        result = StepResult(action=Action.REFLECT, payload={})
        text, style = format_step_result(result)
        assert "reflect" in text
        assert style == "class:info"

    def test_retrieve_more_action(self):
        result = StepResult(action=Action.RETRIEVE_MORE, payload={})
        text, style = format_step_result(result)
        assert "retrieve_more" in text
        assert style == "class:info"


# --- Unit tests: make_completer ---


class TestMakeCompleter:
    def test_returns_word_completer(self):
        from prompt_toolkit.completion import WordCompleter

        completer = make_completer()
        assert isinstance(completer, WordCompleter)

    def test_includes_all_slash_commands(self):
        completer = make_completer()
        for cmd in SLASH_COMMANDS:
            assert cmd in completer.words


# --- Unit tests: slash commands ---


class TestSlashCommands:
    def setup_method(self):
        self.repl = CovenantREPL(llm=False, max_steps=5)

    @patch.object(CovenantREPL, "_print_styled")
    def test_quit_returns_true(self, mock_print):
        assert self.repl._handle_slash_command("/quit") is True
        mock_print.assert_called_once()
        assert "Goodbye" in mock_print.call_args[0][0]

    @patch.object(CovenantREPL, "_print_styled")
    def test_help_shows_commands(self, mock_print):
        assert self.repl._handle_slash_command("/help") is False
        # Should print header + one line per command
        assert mock_print.call_count >= len(SLASH_COMMANDS) + 1

    @patch.object(CovenantREPL, "_print_styled")
    def test_clear_without_agent(self, mock_print):
        assert self.repl._handle_slash_command("/clear") is False
        assert "cleared" in mock_print.call_args[0][0].lower()

    @patch.object(CovenantREPL, "_print_styled")
    def test_clear_with_agent(self, mock_print):
        self.repl._agent = AgentLoop()
        self.repl._agent.history.append(
            StepResult(action=Action.RESPOND, payload={"text": "hi"})
        )
        assert self.repl._handle_slash_command("/clear") is False
        assert len(self.repl._agent.history) == 0

    @patch.object(CovenantREPL, "_print_styled")
    def test_status_no_agent(self, mock_print):
        assert self.repl._handle_slash_command("/status") is False
        assert "No active agent" in mock_print.call_args[0][0]

    @patch.object(CovenantREPL, "_print_styled")
    def test_status_with_agent(self, mock_print):
        self.repl._agent = AgentLoop()
        self.repl._agent.wm.goal = "test goal"
        self.repl._agent.wm.budget = 5
        assert self.repl._handle_slash_command("/status") is False
        all_text = " ".join(call[0][0] for call in mock_print.call_args_list)
        assert "test goal" in all_text
        assert "5" in all_text

    @patch.object(CovenantREPL, "_print_styled")
    def test_reset_clears_agent(self, mock_print):
        self.repl._agent = AgentLoop()
        assert self.repl._handle_slash_command("/reset") is False
        assert self.repl._agent is None
        assert "reset" in mock_print.call_args[0][0].lower()

    @patch.object(CovenantREPL, "_print_styled")
    def test_unknown_command(self, mock_print):
        assert self.repl._handle_slash_command("/foo") is False
        assert "Unknown" in mock_print.call_args[0][0]


# --- Integration: _run_agent_turn with rule-based agent ---


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_run_agent_turn_responds(self):
        """Rule-based agent immediately responds, so a single step should suffice."""
        repl = CovenantREPL(llm=False, max_steps=5)
        results = await repl._run_agent_turn("hello")
        assert len(results) >= 1
        # Rule-based agent with no LLM defaults to RESPOND
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
