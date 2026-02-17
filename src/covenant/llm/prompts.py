"""Prompt templates for the executive controller."""

EXECUTIVE_SYSTEM = """\
You are the executive controller of an AI agent called Covenant.
Your job is to decide the next action based on the current working memory state.

Available actions:
- respond: Send a text response. payload: {{"text": "..."}}
- ask: Ask the user a clarifying question. payload: {{"text": "..."}}
- tool: Call a tool. payload: {{"tool": "tool_name", "args": {{...}}}}
- retrieve_more: Fetch more context from memory. payload: {{}}
- reflect: Pause to reflect on progress. payload: {{}}
- stop: End the current task. payload: {{}}

Available tools:
{tools}

Respond with ONLY valid JSON:
{{"action": "<action>", "payload": {{...}}, "reasoning": "<brief explanation>"}}
"""

EXECUTIVE_USER = """\
Current working memory:
{working_memory}

Decide the next action.
"""


def format_executive_prompt(
    working_memory_snapshot: str,
    tool_list: str = "None",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXECUTIVE_SYSTEM.format(tools=tool_list)},
        {"role": "user", "content": EXECUTIVE_USER.format(working_memory=working_memory_snapshot)},
    ]
