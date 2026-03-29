"""
CLI Entry Point — the user-facing interface.

This is the simplest part of the project.  It:
  1. Loads the .env file so OPENAI_API_KEY is available.
  2. Prints a welcome banner.
  3. Runs a loop: read user input -> call the agent -> print the response.
  4. Handles quit commands and Ctrl+C gracefully.
"""

import sys
from pathlib import Path

# Add the src/ directory to Python's module search path so that
# `import agent.xxx` works regardless of how the script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load environment variables BEFORE importing agent (which creates the
# OpenAI client that reads OPENAI_API_KEY from the environment).
load_dotenv()

from agent.agent import run_agent  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("  City Explorer — AI Day-Trip Planner")
    print("=" * 60)
    print()
    print("Tell me a city and I'll plan your perfect day trip!")
    print("I'll check the real weather and suggest places to visit.")
    print()
    print('Type "quit" or "exit" to leave.  Press Ctrl+C anytime.')
    print("-" * 60)

    conversation_history: list = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Happy travels!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! Happy travels!")
            break

        try:
            response = run_agent(user_input, conversation_history)
            print(f"\nCity Explorer:\n{'-' * 40}")
            print(response)
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy travels!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your OPENAI_API_KEY in .env and try again.")


if __name__ == "__main__":
    main()
