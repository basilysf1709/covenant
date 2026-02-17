"""Interactive REPL for Covenant agent."""

from __future__ import annotations

import asyncio
import itertools
import sys
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from covenant import __version__
from covenant.core.agent_loop import AgentLoop, StepResult
from covenant.core.executive import Action

REPL_STYLE = Style.from_dict(
    {
        "prompt": "bold ansigreen",
        "response": "ansiwhite",
        "tool": "ansiblue",
        "error": "bold ansired",
        "info": "ansicyan",
    }
)

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/clear": "Clear conversation history",
    "/quit": "Exit the REPL",
    "/status": "Show agent working memory",
    "/reset": "Reset the agent to a fresh state",
}


def make_completer() -> WordCompleter:
    """Create a WordCompleter for slash commands."""
    return WordCompleter(list(SLASH_COMMANDS.keys()), sentence=True)


def format_step_result(result: StepResult) -> tuple[str, str]:
    """Format a StepResult into (text, style_class) for display."""
    if result.blocked:
        reason = result.block_reason or "blocked by safety gate"
        return (f"[BLOCKED] {reason}", "class:error")

    action = result.action

    if action == Action.RESPOND:
        text = result.payload.get("text", "")
        return (text, "class:response")

    if action == Action.ASK:
        text = result.payload.get("text", "")
        return (f"? {text}", "class:info")

    if action == Action.TOOL:
        tool = result.payload.get("tool", "unknown")
        args = result.payload.get("args", {})
        return (f"[tool: {tool}] {args}", "class:tool")

    if action == Action.STOP:
        return ("Agent stopped.", "class:info")

    # REFLECT, RETRIEVE_MORE, etc.
    return (f"[{action.value}]", "class:info")


class ThinkingSpinner:
    """Async context manager that shows a braille spinner on stderr."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "thinking..."):
        self._message = message
        self._task: asyncio.Task[None] | None = None

    async def _spin(self) -> None:
        try:
            for frame in itertools.cycle(self.FRAMES):
                sys.stderr.write(f"\r  {frame} {self._message}")
                sys.stderr.flush()
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            sys.stderr.write("\r" + " " * (len(self._message) + 6) + "\r")
            sys.stderr.flush()

    async def __aenter__(self) -> ThinkingSpinner:
        self._task = asyncio.create_task(self._spin())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class CovenantREPL:
    """Interactive REPL for the Covenant agent."""

    def __init__(self, llm: bool = False, max_steps: int = 10):
        self._llm = llm
        self._max_steps = max_steps
        self._agent: AgentLoop | None = None
        self._session = PromptSession(
            completer=make_completer(),
            style=REPL_STYLE,
        )

    def _create_agent(self) -> AgentLoop:
        """Create a fresh AgentLoop instance."""
        loop_kwargs: dict[str, Any] = {}
        if self._llm:
            from covenant.config import get_settings
            from covenant.llm.client import get_llm_client

            settings = get_settings()
            loop_kwargs["llm_client"] = get_llm_client(settings)
        return AgentLoop(**loop_kwargs)

    def _print_styled(self, text: str, style_class: str) -> None:
        """Print styled output via prompt-toolkit."""
        from prompt_toolkit import print_formatted_text

        print_formatted_text(
            FormattedText([(style_class, text)]),
            style=REPL_STYLE,
        )

    def _handle_slash_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        cmd = cmd.strip().lower()

        if cmd == "/quit":
            self._print_styled("Goodbye!", "class:info")
            return True

        if cmd == "/help":
            self._print_styled("Commands:", "class:info")
            for name, desc in SLASH_COMMANDS.items():
                self._print_styled(f"  {name:<10} {desc}", "class:info")
            return False

        if cmd == "/clear":
            if self._agent is not None:
                self._agent.history.clear()
            self._print_styled("History cleared.", "class:info")
            return False

        if cmd == "/status":
            if self._agent is None:
                self._print_styled("No active agent.", "class:info")
            else:
                wm = self._agent.wm
                self._print_styled("Working Memory:", "class:info")
                self._print_styled(f"  Goal:    {wm.goal or '(none)'}", "class:info")
                self._print_styled(f"  State:   {wm.state.value}", "class:info")
                self._print_styled(f"  Budget:  {wm.budget}", "class:info")
                if wm.plan:
                    self._print_styled(f"  Plan:    {wm.plan}", "class:info")
            return False

        if cmd == "/reset":
            self._agent = None
            self._print_styled("Agent reset.", "class:info")
            return False

        self._print_styled(f"Unknown command: {cmd}", "class:error")
        return False

    async def _run_agent_turn(self, user_input: str) -> list[StepResult]:
        """Run agent steps until RESPOND, ASK, STOP, or max steps."""
        if self._agent is None:
            self._agent = self._create_agent()

        results: list[StepResult] = []

        # First step with user input
        async with ThinkingSpinner():
            result = await self._agent.step(user_input)
        results.append(result)

        # Continue stepping until terminal action or budget
        terminal = {Action.RESPOND, Action.ASK, Action.STOP}
        for _ in range(self._max_steps - 1):
            if result.action in terminal or result.blocked:
                break
            async with ThinkingSpinner():
                result = await self._agent.step()
            results.append(result)

        return results

    async def run(self) -> None:
        """Main async REPL loop."""
        self._print_styled(
            f"Covenant v{__version__} - Interactive Mode",
            "class:info",
        )
        self._print_styled(
            "Type a message to chat with the agent. Use /help for commands.",
            "class:info",
        )
        print()  # blank line

        while True:
            try:
                user_input = await self._session.prompt_async(
                    FormattedText([("class:prompt", "covenant> ")]),
                )
            except (EOFError, KeyboardInterrupt):
                self._print_styled("\nGoodbye!", "class:info")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                should_exit = self._handle_slash_command(user_input)
                if should_exit:
                    break
                continue

            # Run agent
            try:
                results = await self._run_agent_turn(user_input)
                for r in results:
                    text, style = format_step_result(r)
                    if text:
                        self._print_styled(text, style)
            except Exception as exc:
                self._print_styled(f"Error: {exc}", "class:error")

            print()  # blank line between turns


def start_repl(llm: bool = False, max_steps: int = 10) -> None:
    """Entry point: create and run the REPL."""
    repl = CovenantREPL(llm=llm, max_steps=max_steps)
    asyncio.run(repl.run())
