"""
The Agentic Loop — the engine that ties Tools and Skills together.

This module implements the core loop that:
  1. Sends the conversation (with the Skill as system prompt) to the LLM.
  2. Checks if the LLM wants to call any Tools.
  3. If yes — executes the tool(s), feeds results back, and loops.
  4. If no  — returns the final text response to the user.

This is the same pattern used by production agent frameworks.  Understanding
this loop is the single most important thing for building AI agents.
"""

import json
from openai import OpenAI

from agent.tools import TOOLS, TOOL_FUNCTIONS
from agent.skills import DAY_TRIP_PLANNER_SKILL

# Initialize the OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI()

MODEL = "gpt-4o"


def run_agent(user_message: str, conversation_history: list) -> str:
    """Run one turn of the agentic loop.

    Args:
        user_message: The latest message from the user.
        conversation_history: A mutable list of message dicts that persists
            across turns.  This function appends to it in place.

    Returns:
        The agent's final text response.
    """
    # Append the new user message to the conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Build the full messages list: system prompt (Skill) + conversation
    messages = [
        {"role": "system", "content": DAY_TRIP_PLANNER_SKILL},
        *conversation_history,
    ]

    # --- Agentic Loop ---
    # Keep calling the LLM until it produces a final text response (no more
    # tool calls).  Each iteration may involve one or more tool calls.
    # Safety limit prevents runaway loops (and runaway API costs).
    MAX_ITERATIONS = 10

    for _iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # Check for tool calls by inspecting the message directly.
        # This is more reliable than checking finish_reason, because the
        # exact string varies across API versions.
        if assistant_message.tool_calls:
            # Append the assistant's message (contains tool_calls metadata)
            # to the conversation so the API can match tool results later.
            conversation_history.append(assistant_message.model_dump())
            messages.append(assistant_message.model_dump())

            # Execute each tool call (OpenAI may request several in parallel)
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Look up and execute the matching Python function
                tool_fn = TOOL_FUNCTIONS.get(function_name)
                if tool_fn is None:
                    result = {"error": f"Unknown tool: {function_name}"}
                else:
                    result = tool_fn(**arguments)

                # Package the result for the API.  The tool_call_id must
                # match so OpenAI knows which call this result belongs to.
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
                conversation_history.append(tool_result_message)
                messages.append(tool_result_message)

            # Loop back to let the LLM process the tool results
            continue

        # No tool calls — the model produced a final text response
        final_text = assistant_message.content or ""
        conversation_history.append({"role": "assistant", "content": final_text})
        return final_text

    # If we exhaust all iterations, return whatever we have
    return "Sorry, I had trouble completing that request. Please try again."
