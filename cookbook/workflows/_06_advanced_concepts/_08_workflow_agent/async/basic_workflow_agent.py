import asyncio

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat
from agno.workflow import WorkflowAgent
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


story_writer = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are tasked with writing a 100 word story based on a given topic",
)

story_formatter = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are tasked with breaking down a short story in prelogues, body and epilogue",
)


def add_references(step_input: StepInput):
    """Add references to the story"""

    previous_output = step_input.previous_step_content

    if isinstance(previous_output, str):
        return previous_output + "\n\nReferences: https://www.agno.com"


# Create a WorkflowAgent that will decide when to run the workflow
workflow_agent = WorkflowAgent(model=OpenAIChat(id="gpt-4o-mini"), num_history_runs=4)

# Create workflow with the WorkflowAgent
workflow = Workflow(
    name="Story Generation Workflow",
    description="A workflow that generates stories, formats them, and adds references",
    agent=workflow_agent,
    steps=[story_writer, story_formatter, add_references],
    db=PostgresDb(db_url),
    # debug_mode=True,
)


async def main():
    """Async main function"""
    # First call - will run the workflow
    print("\n" + "=" * 80)
    print("FIRST CALL (ASYNC): Tell me a story about a husky named Max")
    print("=" * 80)
    await workflow.aprint_response("Tell me a story about a husky named Max")

    # Second call - should answer from history without re-running workflow
    print("\n" + "=" * 80)
    print("SECOND CALL (ASYNC): What was Max like?")
    print("=" * 80)
    await workflow.aprint_response("What was Max like?")

    # Third call - new topic, should run workflow again
    print("\n" + "=" * 80)
    print("THIRD CALL (ASYNC): Now tell me about a cat named Luna")
    print("=" * 80)
    await workflow.aprint_response("Now tell me about a cat named Luna")

    # Fourth call - should answer from history
    print("\n" + "=" * 80)
    print("FOURTH CALL (ASYNC): Compare Max and Luna")
    print("=" * 80)
    await workflow.aprint_response("Compare Max and Luna")


if __name__ == "__main__":
    asyncio.run(main())
