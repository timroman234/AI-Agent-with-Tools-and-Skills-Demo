# Building Tools and Skills: A Detailed Guide

This guide walks through **exactly** how Tools and Skills work inside the City Explorer agent, line by line. By the end, you'll understand not just *what* the code does, but *why* every piece exists — and you'll be able to build your own.

---

## Table of Contents

1. [The Core Mental Model](#the-core-mental-model)
2. [Part 1 — Building a Tool](#part-1--building-a-tool)
   - [Step 1: The Schema (telling the LLM what the tool does)](#step-1-the-schema)
   - [Step 2: The Implementation (doing the actual work)](#step-2-the-implementation)
   - [Step 3: Registering the Tool](#step-3-registering-the-tool)
3. [Part 2 — Building a Skill](#part-2--building-a-skill)
   - [What makes a good skill prompt](#what-makes-a-good-skill-prompt)
   - [How the skill plugs into the agent](#how-the-skill-plugs-into-the-agent)
4. [Part 3 — How They Work Together (the Agentic Loop)](#part-3--how-they-work-together)
5. [Walkthrough: What Happens When You Type "Visit Tokyo"](#walkthrough-what-happens-when-you-type-visit-tokyo)
6. [Building Your Own Tool: Step-by-Step](#building-your-own-tool-step-by-step)
7. [Building Your Own Skill: Step-by-Step](#building-your-own-skill-step-by-step)
8. [Common Mistakes and How to Avoid Them](#common-mistakes-and-how-to-avoid-them)

---

## The Core Mental Model

Before touching any code, internalize this distinction:

| | Tool | Skill |
|---|---|---|
| **Question it answers** | "What can the agent do?" | "How should the agent behave?" |
| **What it is in code** | A JSON schema + a Python function | A system prompt string |
| **Where it lives** | `tools.py` | `skills.py` |
| **Who sees it** | The LLM sees the schema; the Python runtime runs the function | The LLM sees the prompt; the user sees the resulting behavior |
| **Analogy** | A power drill in a workshop | The blueprint that says when and where to drill |

A tool without a skill is a capability with no direction. A skill without tools is a personality with no capabilities. You need both.

---

## Part 1 — Building a Tool

Every tool has exactly **two parts** that must stay in sync. If they disagree, the agent breaks. Let's examine both parts using our `get_weather` tool from `src/agent/tools.py`.

### Step 1: The Schema

The schema is a JSON dictionary that you send to the OpenAI API. The LLM reads it to understand: *what is this tool called, what does it do, and what arguments does it need?*

```python
GET_WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Returns the current weather for a given city. "
            "Use this before making any activity recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris' or 'San Miguel de Allende'"
                }
            },
            "required": ["city"]
        }
    }
}
```

Let's break down **why each field matters**:

#### `"type": "function"`

This tells OpenAI that this tool is a function call. It's always `"function"` — this is a fixed part of the OpenAI API format. Every tool you build will start with this.

#### `"name": "get_weather"`

This is the identifier the LLM uses when it decides to call this tool. When you see the model respond with a tool call, this name is how it references the tool. **This must exactly match the key you use in the `TOOL_FUNCTIONS` dispatch dictionary** (covered in Step 3). If the schema says `"get_weather"` but your dispatch dict says `"getWeather"`, the agent will fail with an "Unknown tool" error.

#### `"description"`

This is the most important field for the LLM's decision-making. The model uses this text to decide **when** to call the tool. Notice our description says *"Use this before making any activity recommendations"* — this is an instruction to the model baked right into the tool definition. The LLM reads this and learns that weather should be checked first.

Writing good descriptions is an art:
- **Too vague**: `"Gets weather"` — the model won't know when to use it
- **Too verbose**: A 500-word essay — wastes tokens and confuses the model
- **Just right**: One sentence on what it returns + one sentence on when to use it

#### `"parameters"`

This follows [JSON Schema](https://json-schema.org/) format. It tells the LLM what arguments to pass. Each property has:
- `"type"` — The data type (`"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`)
- `"description"` — Helps the LLM understand what to pass. Including examples (like `'Paris' or 'San Miguel de Allende'`) significantly improves accuracy.

#### `"required"`

Lists which parameters must always be provided. If `"city"` is required but the user says "What's the weather?", the LLM will ask the user to specify a city rather than calling the tool with an empty argument.

#### A more complex schema: `get_points_of_interest`

Our second tool shows an additional schema feature — the `enum` constraint:

```python
"category": {
    "type": "string",
    "enum": ["landmarks", "restaurants"],
    "description": "The type of places to return"
}
```

The `"enum"` field restricts what values the LLM can pass. Without it, the model might call `get_points_of_interest(city="Paris", category="nightlife")` — and your function wouldn't know how to handle that. With the enum, the model will **only** pass `"landmarks"` or `"restaurants"`. This is how you keep the LLM within the bounds of what your code actually supports.

---

### Step 2: The Implementation

The schema tells the LLM *about* the tool. The Python function *is* the tool. This is the code that actually runs when the agent makes a tool call.

#### Real API tool: `get_weather`

```python
def get_weather(city: str) -> dict:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        return {
            "city": city,
            "temperature_f": current["temp_F"],
            "condition": current["weatherDesc"][0]["value"],
            "humidity": current["humidity"],
        }
    except Exception:
        return {
            "city": city,
            "temperature_f": "72",
            "condition": "Partly cloudy",
            "humidity": "55",
            "note": "simulated (wttr.in unavailable)",
        }
```

Key design decisions and why:

**1. The function signature must match the schema parameters.**

The schema declares one required parameter: `"city"` of type `"string"`. The Python function takes `city: str`. These must match because the agentic loop calls the function like this:

```python
arguments = json.loads(tool_call.function.arguments)  # {"city": "Paris"}
result = tool_fn(**arguments)                          # get_weather(city="Paris")
```

The `**arguments` unpacking means the JSON keys become keyword arguments. If the schema says `"city"` but your function expects `"city_name"`, you'll get a `TypeError`.

**2. Always return a dictionary (or list), never raw strings.**

The result gets serialized to JSON and sent back to the LLM:

```python
tool_result_message = {
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result),   # <-- must be a JSON string
}
```

By returning a structured dict, you give the LLM clear, parseable data. Returning a plain string like `"72 degrees and cloudy"` works but gives the model less structure to reason about.

**3. The return dict is your "contract" with the LLM.**

The keys you return (`city`, `temperature_f`, `condition`, `humidity`) are what the LLM sees. Choose descriptive key names because the model reads them. `"temperature_f"` tells the model the value is in Fahrenheit. If you named it `"temp"`, the model wouldn't know the unit.

**4. Always include a fallback.**

The `except Exception` block returns simulated data when the API is down. This is important because tool failures shouldn't crash the entire agent. The LLM can still produce a useful response with simulated data — it just won't be real-time.

**5. We use `httpx` instead of `requests`.**

`httpx` is already a dependency of the `openai` package, so it adds no extra weight. It also supports `async` if you want to upgrade later. The `timeout=10` prevents the agent from hanging indefinitely on a slow network.

#### Simulated data tool: `get_points_of_interest`

```python
def get_points_of_interest(city: str, category: str) -> list[dict]:
    city_key = next((k for k in _POI_DATABASE if k.lower() == city.lower()), None)

    if city_key and category in _POI_DATABASE[city_key]:
        return _POI_DATABASE[city_key][category]

    # Generic fallback for unknown cities
    if category == "landmarks":
        return [
            {"name": "City Center", "description": f"The historic heart of {city}"},
            ...
        ]
```

Design decisions:

**1. Case-insensitive lookup.**

The LLM might send `"paris"`, `"Paris"`, or `"PARIS"`. The `city.lower()` comparison handles all three. Always normalize LLM inputs — models aren't consistent with casing.

**2. Fallback for unknown cities.**

If someone asks about a city not in our database, we still return plausible generic data. The LLM can work with generic data to produce a useful itinerary. Returning an empty list or an error would force the agent to say "I don't know" — a worse user experience.

**3. Return type matches the schema's implied contract.**

The schema description says it *"Returns a list of notable places"* — and the function returns `list[dict]`. Consistency between what the schema promises and what the function delivers keeps the agent coherent.

---

### Step 3: Registering the Tool

After defining the schema and the function, you must register both so the agentic loop can find them:

```python
# List of all tool schemas — passed to the OpenAI API
TOOLS = [GET_WEATHER_SCHEMA, GET_POINTS_OF_INTEREST_SCHEMA]

# Maps tool name -> callable — used by the agentic loop to dispatch calls
TOOL_FUNCTIONS: dict[str, callable] = {
    "get_weather": get_weather,
    "get_points_of_interest": get_points_of_interest,
}
```

**`TOOLS`** is the list sent to the API:

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,          # <-- the LLM sees these schemas
)
```

**`TOOL_FUNCTIONS`** is the dispatch table used to run the right function:

```python
function_name = tool_call.function.name          # "get_weather"
tool_fn = TOOL_FUNCTIONS.get(function_name)      # get_weather function
result = tool_fn(**arguments)                     # execute it
```

The string key in `TOOL_FUNCTIONS` **must exactly match** the `"name"` field in the corresponding schema. This is the link between the LLM's request ("I want to call get_weather") and your code (the `get_weather` Python function).

---

## Part 2 — Building a Skill

### What makes a good skill prompt

Here is our complete skill from `src/agent/skills.py`:

```python
DAY_TRIP_PLANNER_SKILL = """\
You are City Explorer, a friendly and practical day-trip planning assistant.

When a user asks about visiting a city, you MUST:
1. Always call get_weather first to check current conditions.
2. Call get_points_of_interest for both "landmarks" and "restaurants".
3. Factor weather into every recommendation (e.g., suggest indoor activities
   if it is raining, outdoor ones if the weather is nice).
4. Respond with a structured itinerary broken into Morning, Afternoon, and
   Evening sections.

Keep your tone warm, concise, and practical.  Use the real weather data to
make your suggestions feel personal and timely.
"""
```

Let's examine each part and why it's there:

#### Line 1: Identity

> *"You are City Explorer, a friendly and practical day-trip planning assistant."*

This gives the model a **persona**. Without it, GPT-4o defaults to generic assistant behavior. With it, the model consistently adopts this identity. The adjectives "friendly" and "practical" influence tone throughout every response.

#### Lines 3-7: Tool usage instructions

> *"When a user asks about visiting a city, you MUST:"*
> *"1. Always call get_weather first..."*
> *"2. Call get_points_of_interest for both 'landmarks' and 'restaurants'..."*

This is where the Skill *directs* how Tools are used. Notice:

- **"MUST"** — Strong language produces more reliable compliance than "should" or "consider"
- **"first"** — Establishes ordering. Without this, the model might call POI before weather, and the itinerary wouldn't account for conditions.
- **"for both"** — Explicit instruction to make two calls. Without this, the model might only fetch landmarks and skip restaurants.
- **The tool names match exactly** — `get_weather` and `get_points_of_interest` match the schema names. The model connects these instructions to the available tools.

#### Lines 8-9: Reasoning instructions

> *"3. Factor weather into every recommendation..."*

This is pure reasoning guidance. The tool returns weather data, but the Skill tells the model **how to think about** that data. This is the difference between Tools and Skills in action: the tool provides the temperature, the skill says "if it's raining, suggest indoor activities."

#### Line 10: Output format

> *"4. Respond with a structured itinerary broken into Morning, Afternoon, and Evening sections."*

This controls the response structure. Without it, the model might write a freeform paragraph. With it, every response follows a consistent format that users can scan quickly.

#### Lines 12-13: Tone and style

> *"Keep your tone warm, concise, and practical."*

Three adjectives that shape every word the model produces. "Warm" prevents cold, robotic responses. "Concise" prevents rambling. "Practical" prevents flowery travel-brochure language.

### How the skill plugs into the agent

In `agent.py`, the skill becomes the system message:

```python
messages = [
    {"role": "system", "content": DAY_TRIP_PLANNER_SKILL},   # <-- Skill goes here
    *conversation_history,
]
```

The system message is special in OpenAI's API — it sets the "behind the scenes" instructions that the model follows throughout the conversation. Every API call in the agentic loop includes this same system prompt, so the skill influences every decision the model makes, including which tools to call and how to interpret their results.

### Why skills are model-agnostic

Notice there's nothing OpenAI-specific in the skill text. It's plain English. You could copy this exact string and use it with Claude, Gemini, Llama, or any other LLM that supports system prompts. This is a fundamental property of Skills: **they are portable across models** because they're just natural language instructions.

Tools, by contrast, have model-specific schema formats (OpenAI uses one JSON structure, Anthropic uses another). But Skills are universal.

---

## Part 3 — How They Work Together

The agentic loop in `src/agent/agent.py` is the engine that connects Skills and Tools. Here's the complete flow:

```
                    +------------------------------------------+
                    |  System Prompt (SKILL)                   |
                    |  "You are City Explorer..."              |
                    +------------------------------------------+
                                      |
                                      v
User: "Visit Tokyo"  --->  [ messages list ]  --->  OpenAI API
                                                        |
                                      +-----------------+
                                      |
                                      v
                              Does the response
                            contain tool_calls?
                              /            \
                            YES             NO
                            /                 \
                           v                   v
                    Execute each           Return the text
                    tool call:             to the user
                    - get_weather("Tokyo")
                    - get_points_of_interest("Tokyo", "landmarks")
                    - get_points_of_interest("Tokyo", "restaurants")
                           |
                           v
                    Append results to
                    messages list
                           |
                           v
                    Call OpenAI API again
                    (loop back to top)
```

The critical piece of code is the tool-call detection and dispatch:

```python
if assistant_message.tool_calls:
    # The model wants to use tools — execute them
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name           # e.g. "get_weather"
        arguments = json.loads(tool_call.function.arguments)  # e.g. {"city": "Tokyo"}
        tool_fn = TOOL_FUNCTIONS.get(function_name)       # look up the function
        result = tool_fn(**arguments)                      # call it

        tool_result_message = {
            "role": "tool",
            "tool_call_id": tool_call.id,         # must match the request
            "content": json.dumps(result),         # result as JSON string
        }
```

Three things about this code that are easy to miss:

**1. Parallel tool calls.**
OpenAI may return multiple tool calls in a single response. For example, the model might request `get_weather("Tokyo")` AND `get_points_of_interest("Tokyo", "landmarks")` in the same response. The `for tool_call in assistant_message.tool_calls` loop handles all of them before making the next API call.

**2. The `tool_call_id` must match.**
Each tool call has a unique ID (like `call_abc123`). When you send the result back, you must include the same ID so the API can match the result to the request. If you mix up IDs, the model gets confused about which result belongs to which call.

**3. The assistant message must be appended before tool results.**
The line `conversation_history.append(assistant_message.model_dump())` comes before the tool results are appended. OpenAI's API requires this ordering — the assistant's tool-calling message must precede the tool results in the message list. Reversing this order causes an API error.

---

## Walkthrough: What Happens When You Type "Visit Tokyo"

Here's the exact sequence, message by message:

**Round 1 — User sends message:**

```
messages = [
  {"role": "system",    "content": "You are City Explorer..."},
  {"role": "user",      "content": "Visit Tokyo"}
]
```

The API receives the skill (system), the user message, and the two tool schemas.

**Round 1 — Model responds with tool calls:**

The model, guided by the skill instruction *"Always call get_weather first"* and *"Call get_points_of_interest for both 'landmarks' and 'restaurants'"*, returns three tool calls:

```
tool_calls: [
  {id: "call_001", function: {name: "get_weather", arguments: '{"city":"Tokyo"}'}},
  {id: "call_002", function: {name: "get_points_of_interest", arguments: '{"city":"Tokyo","category":"landmarks"}'}},
  {id: "call_003", function: {name: "get_points_of_interest", arguments: '{"city":"Tokyo","category":"restaurants"}'}}
]
```

**Round 1 — Agent executes tools:**

- `get_weather("Tokyo")` hits wttr.in and returns `{"city": "Tokyo", "temperature_f": "58", "condition": "Light rain", "humidity": "89"}`
- `get_points_of_interest("Tokyo", "landmarks")` returns Senso-ji, Meiji Shrine, Tokyo Skytree
- `get_points_of_interest("Tokyo", "restaurants")` returns Ichiran Ramen, Tsukiji Outer Market, Gonpachi

**Round 2 — Agent sends results back:**

```
messages = [
  {"role": "system",    "content": "You are City Explorer..."},
  {"role": "user",      "content": "Visit Tokyo"},
  {"role": "assistant", "content": null, "tool_calls": [...]},      # model's request
  {"role": "tool",      "tool_call_id": "call_001", "content": '{"city":"Tokyo",...}'},
  {"role": "tool",      "tool_call_id": "call_002", "content": '[{"name":"Senso-ji",...}]'},
  {"role": "tool",      "tool_call_id": "call_003", "content": '[{"name":"Ichiran",...}]'}
]
```

**Round 2 — Model responds with final text:**

Now the model has weather data + landmarks + restaurants. Guided by the skill instruction *"Factor weather into every recommendation"* and *"Respond with a structured itinerary: Morning, Afternoon, Evening"*, it generates:

> **Morning**: Since it's a rainy 58°F in Tokyo, start indoors at **Senso-ji Temple**...
> **Afternoon**: Head to **Tokyo Skytree** for panoramic views...
> **Evening**: Warm up with a bowl at **Ichiran Ramen**...

The model chose indoor-friendly activities because the skill told it to factor in weather, and the weather tool reported rain.

---

## Building Your Own Tool: Step-by-Step

Let's say you want to add a `get_local_events` tool. Follow these three steps:

### Step 1: Define the schema in `tools.py`

```python
GET_LOCAL_EVENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_local_events",
        "description": "Returns upcoming events happening in a city today or this week.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["today", "this_week"],
                    "description": "Whether to return events for today or the full week"
                }
            },
            "required": ["city"]
        }
    }
}
```

Note that `timeframe` is **not** in the `"required"` list. This means the LLM can call `get_local_events(city="Tokyo")` without specifying a timeframe. Your Python function should handle that with a default value.

### Step 2: Write the implementation in `tools.py`

```python
def get_local_events(city: str, timeframe: str = "today") -> list[dict]:
    # Simulated data — in production, you'd call an events API
    events = {
        "Tokyo": [
            {"name": "Cherry Blossom Festival", "type": "festival", "time": "All day"},
            {"name": "Sumo Tournament", "type": "sports", "time": "1:00 PM"},
        ],
        # ... more cities
    }
    city_key = next((k for k in events if k.lower() == city.lower()), None)
    if city_key:
        return events[city_key]
    return [{"name": f"Local Market in {city}", "type": "market", "time": "Morning"}]
```

The `timeframe` parameter has a default value of `"today"` to match the fact that it's optional in the schema.

### Step 3: Register it

Add the schema to the `TOOLS` list and the function to the `TOOL_FUNCTIONS` dict:

```python
TOOLS = [GET_WEATHER_SCHEMA, GET_POINTS_OF_INTEREST_SCHEMA, GET_LOCAL_EVENTS_SCHEMA]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_points_of_interest": get_points_of_interest,
    "get_local_events": get_local_events,
}
```

That's it. The agentic loop in `agent.py` doesn't need any changes — it dynamically dispatches based on whatever is in `TOOL_FUNCTIONS`. The LLM will see the new tool in the `tools` parameter and start using it when relevant.

But the LLM won't use it *optimally* until you update the Skill to mention it.

---

## Building Your Own Skill: Step-by-Step

### Example: Budget Traveler Skill

Add this to `skills.py`:

```python
BUDGET_TRAVELER_SKILL = """\
You are Budget Explorer, a savvy travel assistant who helps people
experience cities without breaking the bank.

When a user asks about visiting a city, you MUST:
1. Call get_weather first — suggest free outdoor activities if weather is good.
2. Call get_points_of_interest for both "landmarks" and "restaurants".
3. Prioritize FREE attractions (parks, public squares, free museum days).
4. For restaurants, always recommend the cheapest option first.
5. Suggest money-saving tips specific to the weather (e.g., "Pack a rain
   jacket instead of buying an umbrella from a street vendor").
6. Format your response as a budget day plan with estimated costs.

Always mention the estimated total cost at the end. Your goal is to help
the user spend under $50 for the entire day.
"""
```

### Swapping it in

In `agent.py`, change one line:

```python
# Before:
from agent.skills import DAY_TRIP_PLANNER_SKILL

# After:
from agent.skills import BUDGET_TRAVELER_SKILL
```

And update the system message:

```python
messages = [
    {"role": "system", "content": BUDGET_TRAVELER_SKILL},
    *conversation_history,
]
```

Same tools, completely different agent behavior. The weather data is the same. The landmarks are the same. But the agent now prioritizes free activities, recommends cheap restaurants first, and adds cost estimates. **That's the power of Skills.**

### Skill writing guidelines

1. **Start with identity.** "You are [Name], a [adjective] [role]." This anchors all behavior.
2. **Use numbered instructions.** The model follows ordered lists more reliably than prose.
3. **Reference tools by name.** The model connects skill instructions to tool schemas when names match.
4. **Use "MUST" for critical behavior.** Weak language like "try to" or "consider" gets ignored under ambiguity.
5. **Specify output format.** "Morning/Afternoon/Evening" or "bullet points" or "table" — be explicit.
6. **Set tone in one sentence.** Pick 2-3 adjectives: "warm, concise, practical" or "witty, irreverent, brief."
7. **Keep it under 200 words.** Longer prompts waste tokens and dilute the important instructions.

---

## Common Mistakes and How to Avoid Them

### Tool Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Schema `"name"` doesn't match `TOOL_FUNCTIONS` key | Agent logs "Unknown tool: ..." | Copy-paste the name string to both places |
| Function parameter name doesn't match schema property | `TypeError: unexpected keyword argument` | Ensure `"city"` in schema = `city` in function signature |
| Returning a non-serializable object | `TypeError: Object of type X is not JSON serializable` | Always return dicts, lists, strings, numbers, booleans |
| No fallback for API failures | Agent crashes when external service is down | Wrap API calls in try/except with simulated fallback data |
| Missing `"required"` field in schema | LLM sometimes omits parameters, function crashes | List all critical params in `"required"`, add defaults for optional ones |

### Skill Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Not mentioning tool names | Model doesn't reliably use your tools | Reference tools by exact name: "Call get_weather first" |
| Vague instructions | Model behavior is inconsistent | Use "MUST", numbered lists, and specific actions |
| Conflicting instructions | Model picks one randomly | Review for contradictions; test with edge cases |
| Too long (500+ words) | Model ignores later instructions | Keep under 200 words; prioritize the most important behaviors |
| No output format specified | Response structure varies every time | Add explicit format: "Respond with Morning, Afternoon, Evening sections" |

### Agentic Loop Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Not appending assistant message before tool results | OpenAI API error | Always append the assistant's tool_calls message first |
| Wrong `tool_call_id` in result | Model misattributes results | Use `tool_call.id` directly from the request |
| No iteration limit | Infinite loop burns API credits | Use `for _ in range(MAX_ITERATIONS)` instead of `while True` |
| Checking `finish_reason` instead of `tool_calls` | Misses tool calls in some API versions | Check `if assistant_message.tool_calls:` directly |
