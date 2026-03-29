"""
Tools — what the agent CAN DO.

Each tool has two parts:
  1. A schema (JSON dict) that tells the LLM what the tool does, its name,
     and what arguments it accepts.  This is sent to the OpenAI API so the
     model can decide when to call the tool.
  2. A Python function that actually executes the action and returns a result.

The agent cannot invent new tools at runtime.  Tools are fixed capabilities
defined here by the developer.
"""

import json
import httpx

# ---------------------------------------------------------------------------
# Tool Schemas  (OpenAI function-calling format)
# ---------------------------------------------------------------------------

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
                    "description": "The city name, e.g. 'Paris' or 'San Miguel de Allende'",
                }
            },
            "required": ["city"],
        },
    },
}

GET_POINTS_OF_INTEREST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_points_of_interest",
        "description": "Returns a list of notable places to visit or eat at in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name",
                },
                "category": {
                    "type": "string",
                    "enum": ["landmarks", "restaurants"],
                    "description": "The type of places to return",
                },
            },
            "required": ["city", "category"],
        },
    },
}

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """Fetch real weather data from wttr.in (free, no API key needed).

    Falls back to simulated data if the request fails so the agent still
    works offline or when wttr.in is unavailable.
    """
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
        # Fallback: simulated weather so the agent works offline
        return {
            "city": city,
            "temperature_f": "72",
            "condition": "Partly cloudy",
            "humidity": "55",
            "note": "simulated (wttr.in unavailable)",
        }


# Simulated points-of-interest database
_POI_DATABASE: dict[str, dict[str, list[dict]]] = {
    "Paris": {
        "landmarks": [
            {"name": "Eiffel Tower", "description": "Iconic iron lattice tower on the Champ de Mars"},
            {"name": "Louvre Museum", "description": "World's largest art museum, home of the Mona Lisa"},
            {"name": "Notre-Dame Cathedral", "description": "Medieval Catholic cathedral on the Île de la Cité"},
        ],
        "restaurants": [
            {"name": "Le Bouillon Chartier", "cuisine": "Traditional French", "price": "$$"},
            {"name": "L'As du Fallafel", "cuisine": "Middle Eastern", "price": "$"},
            {"name": "Café de Flore", "cuisine": "French Café", "price": "$$$"},
        ],
    },
    "Tokyo": {
        "landmarks": [
            {"name": "Senso-ji Temple", "description": "Ancient Buddhist temple in Asakusa"},
            {"name": "Meiji Shrine", "description": "Shinto shrine surrounded by a large forested area"},
            {"name": "Tokyo Skytree", "description": "Tallest tower in Japan with observation decks"},
        ],
        "restaurants": [
            {"name": "Ichiran Ramen", "cuisine": "Ramen", "price": "$$"},
            {"name": "Tsukiji Outer Market", "cuisine": "Sushi & Seafood", "price": "$$"},
            {"name": "Gonpachi Nishi-Azabu", "cuisine": "Japanese Izakaya", "price": "$$$"},
        ],
    },
    "New York": {
        "landmarks": [
            {"name": "Statue of Liberty", "description": "Colossal neoclassical sculpture on Liberty Island"},
            {"name": "Central Park", "description": "843-acre urban park in Manhattan"},
            {"name": "Empire State Building", "description": "Art Deco skyscraper with observation deck"},
        ],
        "restaurants": [
            {"name": "Joe's Pizza", "cuisine": "New York Pizza", "price": "$"},
            {"name": "Katz's Delicatessen", "cuisine": "Jewish Deli", "price": "$$"},
            {"name": "Le Bernardin", "cuisine": "French Seafood", "price": "$$$$"},
        ],
    },
    "London": {
        "landmarks": [
            {"name": "Tower of London", "description": "Historic royal palace and fortress"},
            {"name": "British Museum", "description": "World-famous museum of art and antiquities"},
            {"name": "Buckingham Palace", "description": "Official London residence of the monarch"},
        ],
        "restaurants": [
            {"name": "Dishoom", "cuisine": "Bombay-style Indian", "price": "$$"},
            {"name": "Borough Market", "cuisine": "Various street food", "price": "$"},
            {"name": "The Ledbury", "cuisine": "Modern British", "price": "$$$$"},
        ],
    },
    "Rome": {
        "landmarks": [
            {"name": "Colosseum", "description": "Ancient amphitheatre in the centre of Rome"},
            {"name": "Vatican Museums", "description": "Art museums housing the Sistine Chapel"},
            {"name": "Trevi Fountain", "description": "Baroque fountain and iconic landmark"},
        ],
        "restaurants": [
            {"name": "Da Enzo al 29", "cuisine": "Roman Trattoria", "price": "$$"},
            {"name": "Pizzeria ai Marmi", "cuisine": "Roman Pizza", "price": "$"},
            {"name": "Roscioli", "cuisine": "Italian", "price": "$$$"},
        ],
    },
}


def get_points_of_interest(city: str, category: str) -> list[dict]:
    """Return points of interest for a city.

    Uses a hardcoded database for known cities.  For unknown cities, returns
    generic placeholder data so the agent can still produce a useful response.
    """
    city_key = next((k for k in _POI_DATABASE if k.lower() == city.lower()), None)

    if city_key and category in _POI_DATABASE[city_key]:
        return _POI_DATABASE[city_key][category]

    # Generic fallback for unknown cities
    if category == "landmarks":
        return [
            {"name": "City Center", "description": f"The historic heart of {city}"},
            {"name": "Main Museum", "description": f"The principal museum of {city}"},
            {"name": "Central Park", "description": f"The largest green space in {city}"},
        ]
    else:
        return [
            {"name": "Local Bistro", "cuisine": "Local cuisine", "price": "$$"},
            {"name": "Street Food Market", "cuisine": "Various", "price": "$"},
            {"name": "Fine Dining House", "cuisine": "International", "price": "$$$"},
        ]


# ---------------------------------------------------------------------------
# Exports used by agent.py
# ---------------------------------------------------------------------------

# List of all tool schemas — passed to the OpenAI API
TOOLS = [GET_WEATHER_SCHEMA, GET_POINTS_OF_INTEREST_SCHEMA]

# Maps tool name -> callable — used by the agentic loop to dispatch calls
TOOL_FUNCTIONS: dict[str, callable] = {
    "get_weather": get_weather,
    "get_points_of_interest": get_points_of_interest,
}
