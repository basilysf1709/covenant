"""Interactive REPL for Covenant agent — Neural Interface TUI."""

from __future__ import annotations

import asyncio
import itertools
import sys
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from covenant import __version__
from covenant.core.agent_loop import AgentLoop, StepResult
from covenant.core.executive import Action

# ── Neural color palette ──────────────────────────────────────────────────────

NEURAL_THEME = Theme(
    {
        "cortex": "bold magenta",
        "synapse": "cyan",
        "signal": "bold yellow",
        "dendrite": "dim white",
        "axon": "bright_white",
        "err": "bold red",
        "dim": "dim",
        "status.key": "bold cyan",
        "status.val": "white",
    }
)

console = Console(theme=NEURAL_THEME, highlight=False)

# prompt-toolkit style for the input line
INPUT_STYLE = Style.from_dict(
    {
        "prompt_glyph": "fg:ansimagenta bold",
        "prompt_sep": "fg:ansimagenta",
    }
)

# ── Commands ──────────────────────────────────────────────────────────────────

COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/memories": "Summarize collected episodic memories",
    "/clear": "Clear conversation history",
    "/status": "Show cortex state",
    "/reset": "Reset neural pathways",
    "/quit": "Disconnect",
    "/exit": "Disconnect",
}

EXIT_WORDS = frozenset({"exit", "quit", "/exit", "/quit"})


def _make_completer() -> WordCompleter:
    words = list(COMMANDS.keys()) + ["exit", "quit"]
    return WordCompleter(words, sentence=True)


# ── Banner ────────────────────────────────────────────────────────────────────

HELP_TEXT = """\
[dim]  Type naturally. Say [/dim][signal]exit[/signal][dim] or [/dim][signal]quit[/signal][dim] to disconnect.[/dim]
[dim]  Press [/dim][signal]Ctrl+D[/signal][dim] or [/dim][signal]Ctrl+C[/signal][dim] also works.[/dim]"""


def _print_banner() -> None:
    content = Text(justify="center")
    content.append("◉◉◉", style="bold magenta")
    content.append("  C O V E N A N T  ", style="bold bright_white")
    content.append("◉◉◉", style="bold magenta")
    content.append("\n")
    content.append("∿∿  neural  interface  ∿∿", style="cyan")
    content.append("\n")
    content.append(f"v{__version__}", style="dim")

    console.print()
    console.print(Panel(content, border_style="dim", padding=(0, 2)))
    console.print(HELP_TEXT)
    console.print()


# ── Neural pulse spinner ─────────────────────────────────────────────────────

PULSE_FRAMES = [
    "\x1b[35m◉\x1b[0m \x1b[90m○ ○ ○ ○\x1b[0m",
    "\x1b[90m○\x1b[0m \x1b[35m◉\x1b[0m \x1b[90m○ ○ ○\x1b[0m",
    "\x1b[90m○ ○\x1b[0m \x1b[35m◉\x1b[0m \x1b[90m○ ○\x1b[0m",
    "\x1b[90m○ ○ ○\x1b[0m \x1b[35m◉\x1b[0m \x1b[90m○\x1b[0m",
    "\x1b[90m○ ○ ○ ○\x1b[0m \x1b[35m◉\x1b[0m",
    "\x1b[90m○ ○ ○\x1b[0m \x1b[36m◉\x1b[0m \x1b[90m○\x1b[0m",
    "\x1b[90m○ ○\x1b[0m \x1b[36m◉\x1b[0m \x1b[90m○ ○\x1b[0m",
    "\x1b[90m○\x1b[0m \x1b[36m◉\x1b[0m \x1b[90m○ ○ ○\x1b[0m",
]


class NeuralSpinner:
    """Async context manager — neural pulse animation on stderr."""

    def __init__(self, label: str = "cortex processing"):
        self._label = label
        self._task: asyncio.Task[None] | None = None

    async def _spin(self) -> None:
        try:
            for frame in itertools.cycle(PULSE_FRAMES):
                sys.stderr.write(f"\r  {frame}  \x1b[90m{self._label}\x1b[0m")
                sys.stderr.flush()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # clear the line
            sys.stderr.write("\r" + " " * (len(self._label) + 30) + "\r")
            sys.stderr.flush()

    async def __aenter__(self) -> NeuralSpinner:
        self._task = asyncio.create_task(self._spin())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ── Output formatting ─────────────────────────────────────────────────────────

_RESPONSE_BORDER = "\x1b[35m│\x1b[0m"


def _display_result(result: StepResult) -> None:
    """Render a StepResult with brain-themed formatting."""
    if result.blocked:
        reason = result.block_reason or "blocked by safety gate"
        console.print(
            Panel(
                f"[err]{reason}[/err]",
                title="[err]⚠ BLOCKED[/err]",
                border_style="red",
                padding=(0, 1),
            )
        )
        return

    action = result.action

    if action == Action.RESPOND:
        text = result.payload.get("text", "")
        if text:
            console.print(
                Panel(
                    text,
                    border_style="magenta",
                    padding=(0, 2),
                )
            )
        return

    if action == Action.ASK:
        text = result.payload.get("text", "")
        console.print(
            Panel(
                f"[signal]?[/signal] {text}",
                border_style="yellow",
                padding=(0, 2),
            )
        )
        return

    if action == Action.TOOL:
        tool = result.payload.get("tool", "?")
        args = result.payload.get("args", {})
        console.print(f"  [synapse]⚡ tool:[/synapse] [axon]{tool}[/axon] [dim]{args}[/dim]")
        return

    if action == Action.STOP:
        console.print("  [dim]◉ session ended[/dim]")
        return

    # REFLECT, RETRIEVE_MORE, etc.
    console.print(f"  [dim]∿ {action.value}[/dim]")


# ── Status display ────────────────────────────────────────────────────────────


def _display_status(agent: AgentLoop | None) -> None:
    if agent is None:
        console.print("  [dim]No active cortex.[/dim]")
        return

    wm = agent.wm
    status = Text()
    status.append("  ◉ Goal     ", style="status.key")
    status.append(f"{wm.goal or '—'}\n", style="status.val")
    status.append("  ◉ State    ", style="status.key")
    status.append(f"{wm.state.value}\n", style="status.val")
    status.append("  ◉ Budget   ", style="status.key")
    status.append(f"{wm.budget}\n", style="status.val")
    if wm.plan:
        status.append("  ◉ Plan     ", style="status.key")
        status.append(f"{wm.plan}\n", style="status.val")
    obs_count = len(wm.recent_observations)
    status.append("  ◉ Signals  ", style="status.key")
    status.append(f"{obs_count} observation(s)", style="status.val")

    console.print(
        Panel(status, title="[cortex]cortex state[/cortex]", border_style="magenta", padding=(0, 1))
    )


# ── REPL ──────────────────────────────────────────────────────────────────────


class CovenantREPL:
    """Interactive neural interface for the Covenant agent."""

    def __init__(self, llm: bool = False, max_steps: int = 10):
        self._llm = llm
        self._max_steps = max_steps
        self._agent: AgentLoop | None = None
        self._session = PromptSession(
            completer=_make_completer(),
            style=INPUT_STYLE,
        )
        self._session_factory: Any = None
        self._db_session: Any = None

    async def _init_memory(self) -> None:
        """Initialize DB for persistent memory. Falls back to volatile on failure."""
        try:
            from covenant.config import get_settings
            from covenant.memory.database import get_session_factory
            from covenant.memory.database import init_db as _init_db

            settings = get_settings()
            await _init_db(settings)
            self._session_factory = get_session_factory()
        except Exception as exc:
            self._session_factory = None
            console.print(
                f"  [err]⚠ memory init failed:[/err] [dim]{exc}[/dim]\n"
                "  [dim]Running in volatile mode — memory will not persist.[/dim]"
            )

    async def _create_agent(self) -> AgentLoop:
        loop_kwargs: dict[str, Any] = {}
        if self._llm:
            from covenant.config import get_settings
            from covenant.llm.client import get_embed_fn, get_llm_client

            settings = get_settings()
            loop_kwargs["llm_client"] = get_llm_client(settings)
            loop_kwargs["embed_fn"] = get_embed_fn(settings)

        if self._session_factory is not None:
            self._db_session = self._session_factory()
            await self._db_session.__aenter__()
            loop_kwargs["session"] = self._db_session

        return AgentLoop(**loop_kwargs)

    async def _cleanup_db_session(self) -> None:
        if self._db_session is not None:
            await self._db_session.__aexit__(None, None, None)
            self._db_session = None

    async def _display_memories(self) -> None:
        """Show a summary of stored episodic memories."""
        if self._session_factory is None:
            console.print("  [dim]No persistent memory available (volatile mode).[/dim]")
            return

        from collections import Counter

        from sqlalchemy import func, select

        from covenant.memory.models import Episode

        async with self._session_factory() as session:
            # Total count
            total = (await session.execute(select(func.count(Episode.id)))).scalar() or 0
            if total == 0:
                console.print("  [dim]No episodic memories stored yet.[/dim]")
                return

            # Count with embeddings
            embedded = (
                await session.execute(
                    select(func.count(Episode.id)).where(Episode.embedding.is_not(None))
                )
            ).scalar() or 0

            # Recent episodes
            recent = (
                await session.execute(
                    select(Episode).order_by(Episode.ts.desc()).limit(10)
                )
            ).scalars().all()

            # Goal frequency across all episodes
            all_goals = (
                await session.execute(select(Episode.goal))
            ).scalars().all()
            goal_counts = Counter(g.strip() for g in all_goals if g and g.strip())
            top_goals = goal_counts.most_common(5)

        content = Text()
        content.append(f"  ◉ Total episodes   ", style="status.key")
        content.append(f"{total}\n", style="status.val")
        content.append(f"  ◉ With embeddings  ", style="status.key")
        content.append(f"{embedded}\n", style="status.val")

        if top_goals:
            content.append("\n")
            content.append("  ◉ Top goals\n", style="status.key")
            for goal, count in top_goals:
                label = goal if len(goal) <= 60 else goal[:57] + "..."
                content.append(f"    {count:>3}×  ", style="synapse")
                content.append(f"{label}\n", style="status.val")

        content.append("\n")
        content.append("  ◉ Recent memories\n", style="status.key")
        for ep in recent:
            ts = ep.ts.strftime("%m/%d %H:%M") if ep.ts else "?"
            emb_marker = "⬡" if ep.embedding else "○"
            goal = ep.goal if len(ep.goal) <= 50 else ep.goal[:47] + "..."
            outcome = ep.outcome if len(ep.outcome) <= 40 else ep.outcome[:37] + "..."
            content.append(f"    {emb_marker} ", style="cortex")
            content.append(f"[{ts}] ", style="dim")
            content.append(f"{goal}", style="axon")
            content.append(f" → {outcome}\n", style="dendrite")

        console.print(
            Panel(content, title="[cortex]episodic memory[/cortex]", border_style="magenta", padding=(0, 1))
        )

    async def _handle_command(self, cmd: str) -> bool:
        """Handle a command. Returns True if the REPL should exit."""
        cmd = cmd.strip().lower()

        if cmd in EXIT_WORDS:
            console.print("  [dim]◉ disconnecting from cortex...[/dim]")
            console.print()
            return True

        if cmd == "/help":
            console.print()
            for name, desc in COMMANDS.items():
                console.print(f"  [synapse]{name:<10}[/synapse] [dim]{desc}[/dim]")
            console.print(f"  [signal]{'exit':<10}[/signal] [dim]Disconnect[/dim]")
            console.print(f"  [signal]{'quit':<10}[/signal] [dim]Disconnect[/dim]")
            console.print()
            return False

        if cmd == "/memories":
            await self._display_memories()
            return False

        if cmd == "/clear":
            if self._agent is not None:
                self._agent.history.clear()
            console.print("  [dim]◉ signals cleared[/dim]")
            return False

        if cmd == "/status":
            _display_status(self._agent)
            return False

        if cmd == "/reset":
            await self._cleanup_db_session()
            self._agent = None
            console.print("  [dim]◉ neural pathways reset[/dim]")
            return False

        console.print(f"  [err]unknown command:[/err] {cmd}")
        return False

    async def _run_agent_turn(self, user_input: str) -> list[StepResult]:
        if self._agent is None:
            self._agent = await self._create_agent()

        results: list[StepResult] = []

        async with NeuralSpinner():
            result = await self._agent.step(user_input)
        results.append(result)

        terminal = {Action.RESPOND, Action.ASK, Action.STOP}
        for _ in range(self._max_steps - 1):
            if result.action in terminal or result.blocked:
                break
            async with NeuralSpinner():
                result = await self._agent.step()
            results.append(result)

        return results

    async def run(self) -> None:
        """Main async REPL loop."""
        await self._init_memory()
        _print_banner()

        prompt_text = FormattedText(
            [
                ("class:prompt_glyph", "◉ "),
                ("class:prompt_sep", "› "),
            ]
        )

        try:
            while True:
                try:
                    user_input = await self._session.prompt_async(prompt_text)
                except (EOFError, KeyboardInterrupt):
                    console.print("\n  [dim]◉ disconnecting from cortex...[/dim]")
                    console.print()
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Exit words (bare or slash)
                if user_input.lower() in EXIT_WORDS:
                    console.print("  [dim]◉ disconnecting from cortex...[/dim]")
                    console.print()
                    break

                # Slash commands
                if user_input.startswith("/"):
                    should_exit = await self._handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # Agent turn
                try:
                    results = await self._run_agent_turn(user_input)
                    for r in results:
                        _display_result(r)
                except Exception as exc:
                    console.print(f"  [err]⚠ error:[/err] {exc}")

                console.print()  # breathing room between turns
        finally:
            await self._cleanup_db_session()
            if self._session_factory is not None:
                from covenant.memory.database import close_db

                await close_db()


def start_repl(llm: bool = False, max_steps: int = 10) -> None:
    """Entry point: create and run the REPL."""
    repl = CovenantREPL(llm=llm, max_steps=max_steps)
    asyncio.run(repl.run())
