"""Pre-flight check for the LLM configured via CREW_MODEL / .env.

Builds the LLM exactly the way main.py does, then asks a minimal one-tool
agent to call a throwaway tool that returns a freshly generated, unguessable
code. If that exact code comes back in the final answer, the model didn't
just respond - it genuinely issued a structured tool call and relayed a real
tool result. That's the capability this crew depends on for every agent.

Usage:
    python check_llm_connection.py
"""
import os
import sys
import uuid

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

SECRET_CODE = uuid.uuid4().hex[:8]


@tool("get_secret_code")
def get_secret_code() -> str:
    """Return a one-time secret code. This tool must be called to learn the
    code - it cannot be guessed or computed."""
    return SECRET_CODE


def main() -> int:
    load_dotenv()
    from pair_crew.agents import get_llm  # imported after load_dotenv for a clean env

    model = os.getenv("CREW_MODEL", "anthropic/claude-sonnet-5")
    print(f"Testing CREW_MODEL={model!r} ...\n")

    try:
        llm = get_llm()
        agent = Agent(
            role="Connection Tester",
            goal="Call the get_secret_code tool and report exactly what it returns.",
            backstory="You always call the tool you are given rather than guessing an answer.",
            tools=[get_secret_code],
            llm=llm,
            allow_delegation=False,
            verbose=True,
        )
        task = Task(
            description=(
                "Call the get_secret_code tool (it takes no arguments). "
                "Reply with ONLY the exact code it returns - no other words."
            ),
            expected_output="The exact secret code string, nothing else.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
    except Exception as exc:
        print(f"\n❌ FAILED to get a response from {model!r}: {exc}")
        if model.startswith("ollama/"):
            model_name = model.split("/", 1)[1] if "/" in model else model
            print(
                "\nOllama troubleshooting:\n"
                "  - Is the server running? Try: ollama serve\n"
                f"  - Is the model pulled? Try: ollama pull {model_name}\n"
                "  - Using a non-default host? Set OLLAMA_HOST in .env.\n"
            )
        return 1

    output = str(result)
    if SECRET_CODE in output:
        print(
            f"\n✅ SUCCESS - {model!r} connected and correctly called the tool "
            f"(code {SECRET_CODE} confirmed)."
        )
        return 0

    print(
        f"\n❌ CONNECTED, but tool-calling failed - the model answered without "
        f"relaying the real tool result (expected {SECRET_CODE!r}, got: {output!r}).\n"
        "This model is likely too unreliable at tool-calling for this crew; try a different model."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
