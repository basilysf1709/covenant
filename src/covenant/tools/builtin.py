"""Built-in tools for the Covenant agent."""

from __future__ import annotations

import ast
import operator

from covenant.tools.router import ToolRouter

# Default tool router with built-in tools
default_router = ToolRouter()

# Safe math operators
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.expr) -> float:
    """Safely evaluate a math expression AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return _SAFE_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval_expr(node.operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@default_router.register(
    name="calculator",
    description="Evaluate a safe math expression",
    parameters={"expression": "A math expression like '2 + 3 * 4'"},
)
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval_expr(tree.body)
    return str(result)


@default_router.register(
    name="web_search",
    description="Search the web (stub - returns placeholder results)",
    parameters={"query": "The search query"},
)
async def web_search(query: str) -> str:
    """Stub web search tool."""
    return f"[web_search stub] No results for: {query}"
