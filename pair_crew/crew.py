import time

from crewai import Crew, Process

from pair_crew.agents import build_evaluator_agent, build_ml_agent, build_supervisor_agent, get_llm
from pair_crew.tasks import build_evaluator_task, build_ml_task, build_supervisor_task


def build_crew() -> Crew:
    # Captured before any task runs; task guardrails use it to verify each agent's
    # expected output artifacts were actually (re)written by real tool calls during
    # this run, rather than fabricated by the LLM without calling any tools.
    run_started_at = time.time()

    llm = get_llm()

    ml_agent = build_ml_agent(llm)
    supervisor_agent = build_supervisor_agent(llm)
    evaluator_agent = build_evaluator_agent(llm)

    ml_task = build_ml_task(ml_agent, run_started_at)
    supervisor_task = build_supervisor_task(supervisor_agent, [ml_task], run_started_at)
    evaluator_task = build_evaluator_task(evaluator_agent, [ml_task, supervisor_task], run_started_at)

    return Crew(
        agents=[ml_agent, supervisor_agent, evaluator_agent],
        tasks=[ml_task, supervisor_task, evaluator_task],
        process=Process.sequential,
        verbose=True,
    )
