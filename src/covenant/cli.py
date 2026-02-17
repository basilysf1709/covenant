"""Covenant CLI."""

import asyncio

import click

from covenant import __version__


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option("--llm/--no-llm", default=False, help="Use LLM in REPL mode.")
@click.option("--max-steps", default=10, help="Max agent steps per turn in REPL mode.")
@click.pass_context
def main(ctx, llm: bool, max_steps: int):
    """Covenant: Cloud Intelligence System."""
    ctx.ensure_object(dict)
    ctx.obj["llm"] = llm
    ctx.obj["max_steps"] = max_steps
    if ctx.invoked_subcommand is None:
        from covenant.repl import start_repl

        start_repl(llm=llm, max_steps=max_steps)


@main.command()
def hello():
    """Say hello."""
    click.echo("Hello from Covenant!")


@main.command("init-db")
def init_db():
    """Initialize the database."""
    from covenant.config import get_settings
    from covenant.memory.database import init_db as _init_db

    settings = get_settings()
    asyncio.run(_init_db(settings))
    click.echo("Database initialized.")


@main.command()
@click.argument("goal")
@click.option("--llm/--no-llm", default=False, help="Use LLM for decision-making.")
@click.option("--memory/--no-memory", default=True, help="Enable persistent memory.")
@click.option("--max-steps", default=10, help="Max agent loop steps.")
def run(goal: str, llm: bool, memory: bool, max_steps: int):
    """Run the agent on a goal."""
    from covenant.core.agent_loop import AgentLoop

    async def _run():
        loop_kwargs: dict = {}
        if llm:
            from covenant.config import get_settings
            from covenant.llm.client import get_llm_client

            settings = get_settings()
            loop_kwargs["llm_client"] = get_llm_client(settings)

        session = None
        if memory:
            from covenant.config import get_settings
            from covenant.memory.database import close_db, get_session_factory
            from covenant.memory.database import init_db as _init_db

            settings = get_settings()
            await _init_db(settings)
            session_factory = get_session_factory()
            session = session_factory()
            await session.__aenter__()
            loop_kwargs["session"] = session

        try:
            agent = AgentLoop(**loop_kwargs)
            results = await agent.run(goal, max_steps=max_steps)
            for r in results:
                status = " [BLOCKED]" if r.blocked else ""
                text = r.payload.get("text", r.action.value)
                click.echo(f"[{r.action.value}]{status} {text}")
        finally:
            if session is not None:
                await session.__aexit__(None, None, None)
                await close_db()

    asyncio.run(_run())


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=8000, help="Bind port.")
def serve(host: str, port: int):
    """Start the API server."""
    import uvicorn

    from covenant.api.app import create_app

    app = create_app()
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--threshold", default=0.1, help="Prune episodes below this salience.")
def consolidate(threshold: float):
    """Run memory consolidation (sleep cycle)."""
    from covenant.config import get_settings
    from covenant.memory.consolidation import Consolidator
    from covenant.memory.database import close_db, get_session_factory, init_db

    async def _consolidate():
        settings = get_settings()
        await init_db(settings)
        session_factory = get_session_factory()
        async with session_factory() as session:
            consolidator = Consolidator(session)
            report = await consolidator.run(prune_threshold=threshold)
            await session.commit()
        await close_db()
        return report

    report = asyncio.run(_consolidate())
    click.echo("Consolidation complete:")
    click.echo(f"  Deduped: {report.deduped}")
    click.echo(f"  Summarized: {report.summarized}")
    click.echo(f"  Facts extracted: {report.facts_extracted}")
    click.echo(f"  Salience updated: {report.salience_updated}")
    click.echo(f"  Pruned: {report.pruned}")


@main.command("eval")
def eval_cmd():
    """Run evaluation scenarios."""
    from covenant.eval.harness import EvalHarness
    from covenant.eval.scenarios import SCENARIOS

    async def _eval():
        harness = EvalHarness()
        report = await harness.run_all(SCENARIOS)
        return report

    report = asyncio.run(_eval())
    click.echo(report.summary())


if __name__ == "__main__":
    main()
