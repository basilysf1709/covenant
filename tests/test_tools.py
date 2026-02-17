"""Tests for tool router and built-in tools."""

from covenant.core.executive import Action, Decision
from covenant.core.safety import SafetyGate
from covenant.tools.builtin import calculator, default_router, web_search
from covenant.tools.router import ToolRouter, ToolSpec


async def test_register_and_list():
    router = ToolRouter()

    @router.register(name="echo", description="Echo input", parameters={"text": "text to echo"})
    def echo(text: str) -> str:
        return text

    tools = router.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "echo"


async def test_execute_sync_tool():
    router = ToolRouter()

    @router.register(name="add", description="Add two numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    result = await router.execute("add", a=2, b=3)
    assert result == "5"


async def test_execute_async_tool():
    router = ToolRouter()

    @router.register(name="async_echo", description="Async echo")
    async def async_echo(text: str) -> str:
        return text

    result = await router.execute("async_echo", text="hello")
    assert result == "hello"


async def test_execute_unknown_tool():
    router = ToolRouter()
    result = await router.execute("nonexistent")
    assert "unknown tool" in result.lower()


async def test_calculator():
    assert calculator("2 + 3") == "5.0"
    assert calculator("10 * 5 - 3") == "47.0"
    assert calculator("2 ** 8") == "256.0"


async def test_calculator_via_router():
    result = await default_router.execute("calculator", expression="3 + 4")
    assert result == "7.0"


async def test_web_search_stub():
    result = await default_router.execute("web_search", query="test")
    assert "stub" in result.lower()


async def test_list_tools_text():
    text = default_router.list_tools_text()
    assert "calculator" in text
    assert "web_search" in text


async def test_safety_blocks_dangerous_tool():
    gate = SafetyGate()
    decision = Decision(action=Action.TOOL, payload={"tool": "shell"})
    result = gate.check(decision)
    assert not result.allowed


async def test_safety_allows_safe_tool():
    gate = SafetyGate()
    decision = Decision(action=Action.TOOL, payload={"tool": "calculator"})
    result = gate.check(decision)
    assert result.allowed


def test_add_tool():
    router = ToolRouter()
    spec = ToolSpec(name="manual", description="Manually added", fn=lambda: "ok")
    router.add_tool(spec)
    assert len(router.list_tools()) == 1
