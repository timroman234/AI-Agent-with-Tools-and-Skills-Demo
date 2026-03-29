# City Explorer — Educational AI Agent

An educational project that teaches the difference between **Tools** (what an agent *can do*) and **Skills** (how an agent *thinks*) by building a day-trip planning assistant powered by OpenAI's GPT-4o.

## What is a Tool vs. a Skill?

| | Tool | Skill |
|---|---|---|
| **What it is** | A function the agent can call | A system prompt that shapes behavior |
| **Analogy** | A hammer in a toolbox | Knowing *when* and *how* to use the hammer |
| **In code** | JSON schema + Python function | A string passed as the system message |
| **Example** | `get_weather("Paris")` returns data | "Always check weather before recommending activities" |
| **Can the agent create new ones at runtime?** | No | No |
| **Who defines them?** | The developer | The developer |

**Key insight:** You can swap the Skill (system prompt) without changing any Tools, and the agent will behave completely differently. The same `get_weather` and `get_points_of_interest` tools could power a "budget backpacker" agent or a "luxury concierge" agent — just by changing the skill prompt.

## Architecture

```
User Input
    |
    v
+----------------------------------------------+
|  main.py — CLI Loop                          |
|  Reads input, prints output                  |
+----------------------------------------------+
    |
    v
+----------------------------------------------+
|  agent.py — Agentic Loop                     |
|                                              |
|  1. Build messages:                          |
|     [system: SKILL] + conversation_history   |
|                                              |
|  2. Call OpenAI API with messages + TOOLS    |
|                                              |
|  3. If finish_reason == "tool_calls":        |
|     - Execute each tool via TOOL_FUNCTIONS   |
|     - Append results to messages             |
|     - Go back to step 2                      |
|                                              |
|  4. If finish_reason == "stop":              |
|     - Return the text response               |
+----------------------------------------------+
    |                        |
    v                        v
+------------------+  +------------------+
|  skills.py       |  |  tools.py        |
|                  |  |                  |
|  DAY_TRIP_       |  |  get_weather()   |
|  PLANNER_SKILL   |  |  (real API)      |
|  (system prompt) |  |                  |
|                  |  |  get_points_of_  |
|                  |  |  interest()      |
|                  |  |  (simulated)     |
+------------------+  +------------------+
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key

### Steps

```bash
# 1. Navigate to the project
cd city-explorer-agent

# 2. Add your API key
#    Edit .env and replace sk-your-key-here with your real key
#    e.g.: OPENAI_API_KEY=sk-proj-abc123...

# 3. Install dependencies
uv sync

# 4. Run the agent
uv run python src/agent/main.py
```

Then type something like: **"I want to visit Tokyo"**

The agent will:
1. Check real weather in Tokyo (via wttr.in)
2. Look up landmarks and restaurants
3. Create a morning/afternoon/evening itinerary factoring in the weather

## File Walkthrough

### `tools.py` — What the agent CAN DO

Contains two tools, each with a **schema** (JSON that tells GPT-4o what the tool does) and an **implementation** (Python function that executes the action). `get_weather` makes a real HTTP request to wttr.in for live weather data, with a fallback to simulated data if offline. `get_points_of_interest` returns simulated data from a hardcoded database covering Paris, Tokyo, New York, London, and Rome.

### `skills.py` — How the agent THINKS

Contains a single system prompt (`DAY_TRIP_PLANNER_SKILL`) that instructs the agent to always check weather first, fetch both landmarks and restaurants, factor weather into recommendations, and format output as a morning/afternoon/evening itinerary. This is the only file you need to edit to change the agent's personality.

### `agent.py` — The Engine

Implements the agentic loop: send messages to the API, check if the model wants to call tools, execute them, feed results back, and repeat until the model produces a final text response. This is the core pattern behind all AI agents.

### `main.py` — The Interface

A simple CLI loop that loads environment variables, reads user input, calls the agent, and prints responses. Handles quit commands and Ctrl+C gracefully.

## Exercises

Try these modifications to deepen your understanding:

### 1. Add a new Tool: `get_local_events`

Create a new tool that returns upcoming events for a city. You'll need:
- A JSON schema in `tools.py`
- A Python function (simulated data is fine)
- Add it to the `TOOLS` list and `TOOL_FUNCTIONS` dict
- Update the skill prompt in `skills.py` to mention it

This teaches you how to **expand what the agent can do**.

### 2. Write a new Skill: `BUDGET_TRAVELER_SKILL`

Write a new system prompt that makes the agent focus on free activities, cheap eats, and money-saving tips. Swap it in `agent.py` (replace `DAY_TRIP_PLANNER_SKILL`). Notice how the agent's behavior changes completely — same tools, different personality.

This teaches you how to **change how the agent thinks**.

### 3. Add verbose mode

Add a `--verbose` flag to `main.py` that prints the raw API requests and responses (messages list, tool calls, tool results). This helps you see exactly what's happening inside the agentic loop.
