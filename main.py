"""Entry point for the MHI/HHI pair-trading CrewAI workflow.

Usage:
    python main.py

Requires an LLM API key in the environment (e.g. ANTHROPIC_API_KEY or
OPENAI_API_KEY) matching the model set via the CREW_MODEL env var
(default: anthropic/claude-sonnet-5). See README.md for details.
"""
from dotenv import load_dotenv

from pair_crew import config
from pair_crew.crew import build_crew


def main() -> None:
    load_dotenv()
    config.ENTRY_POINT = "crewai_agents"
    crew = build_crew()
    result = crew.kickoff()
    print("\n\n===== FINAL DELIVERABLE =====\n")
    print(result)


if __name__ == "__main__":
    main()
