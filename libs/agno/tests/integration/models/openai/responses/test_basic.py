from typing import Optional

import pytest
from pydantic import BaseModel, Field

from agno.agent import Agent, RunOutput  # noqa
from agno.db.sqlite import SqliteDb
from agno.exceptions import ModelProviderError
from agno.models.openai import OpenAIResponses


@pytest.fixture(scope="module")
def openai_responses_model():
    """Fixture that provides an OpenAI Responses model and reuses it across all tests in the module."""
    return OpenAIResponses(id="gpt-4o-mini")


def _assert_metrics(response: RunOutput):
    """
    Assert that the response metrics are valid and consistent.

    Args:
        response: The RunOutput to validate metrics for
    """
    assert response.metrics is not None
    input_tokens = response.metrics.input_tokens
    output_tokens = response.metrics.output_tokens
    total_tokens = response.metrics.total_tokens

    assert input_tokens > 0
    assert output_tokens > 0
    assert total_tokens > 0
    assert total_tokens == input_tokens + output_tokens


def test_basic(openai_responses_model):
    """Test basic functionality of the OpenAIResponses model."""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    # Run a simple query
    response: RunOutput = agent.run("Share a 2 sentence horror story")

    assert response.content is not None and response.messages is not None
    assert len(response.messages) == 3
    assert [m.role for m in response.messages] == ["system", "user", "assistant"]

    _assert_metrics(response)


def test_basic_stream(openai_responses_model):
    """Test basic streaming functionality of the OpenAIResponses model."""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    run_stream = agent.run("Say 'hi'", stream=True)
    for chunk in run_stream:
        assert chunk.content is not None or chunk.model_provider_data is not None


@pytest.mark.asyncio
async def test_async_basic(openai_responses_model):
    """Test basic async functionality of the OpenAIResponses model."""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    response = await agent.arun("Share a 2 sentence horror story")

    assert response.content is not None
    assert response.messages is not None
    assert len(response.messages) == 3
    assert [m.role for m in response.messages] == ["system", "user", "assistant"]
    _assert_metrics(response)


@pytest.mark.asyncio
async def test_async_basic_stream(openai_responses_model):
    """Test basic async streaming functionality of the OpenAIResponses model."""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    async for response in agent.arun("Share a 2 sentence horror story", stream=True):
        assert response.content is not None or response.model_provider_data is not None


def test_exception_handling():
    """Test proper error handling for invalid model IDs."""
    agent = Agent(model=OpenAIResponses(id="gpt-100"), markdown=True, telemetry=False)

    with pytest.raises(ModelProviderError) as exc:
        agent.run("Share a 2 sentence horror story")

    assert exc.value.model_name == "OpenAIResponses"
    assert exc.value.model_id == "gpt-100"
    assert exc.value.status_code == 400


def test_with_memory(openai_responses_model):
    """Test that the model retains context from previous interactions."""
    agent = Agent(
        model=openai_responses_model,
        db=SqliteDb(db_file="tmp/openai/responses/test_with_memory.db"),
        add_history_to_context=True,
        markdown=True,
        telemetry=False,
    )

    # First interaction
    response1 = agent.run("My name is John Smith")
    assert response1.content is not None

    # Second interaction should remember the name
    response2 = agent.run("What's my name?")
    assert response2.content is not None and "John Smith" in response2.content

    # Verify memories were created
    messages = agent.get_messages_for_session()
    assert len(messages) == 5
    assert [m.role for m in messages] == ["system", "user", "assistant", "user", "assistant"]

    # Test metrics structure and types
    _assert_metrics(response2)


def test_structured_output_json_mode(openai_responses_model):
    """Test structured output with Pydantic models."""

    class MovieScript(BaseModel):
        title: str = Field(..., description="Movie title")
        genre: str = Field(..., description="Movie genre")
        plot: str = Field(..., description="Brief plot summary")
        release_date: Optional[str] = Field(None, description="Release date of the movie")

    agent = Agent(
        model=openai_responses_model,
        output_schema=MovieScript,
        use_json_mode=True,
        telemetry=False,
    )

    response = agent.run("Create a movie about time travel")

    # Verify structured output
    assert isinstance(response.content, MovieScript)
    assert response.content.title is not None
    assert response.content.genre is not None
    assert response.content.plot is not None


def test_structured_output(openai_responses_model):
    """Test native structured output with the responses API."""

    class MovieScript(BaseModel):
        title: str = Field(..., description="Movie title")
        genre: str = Field(..., description="Movie genre")
        plot: str = Field(..., description="Brief plot summary")
        release_date: Optional[str] = Field(None, description="Release date of the movie")

    agent = Agent(
        model=openai_responses_model,
        output_schema=MovieScript,
        telemetry=False,
    )

    response = agent.run("Create a movie about time travel")

    # Verify structured output
    assert isinstance(response.content, MovieScript)
    assert response.content.title is not None
    assert response.content.genre is not None
    assert response.content.plot is not None


def test_history(openai_responses_model):
    """Test conversation history in the agent."""
    agent = Agent(
        model=openai_responses_model,
        db=SqliteDb(db_file="tmp/openai/responses/test_basic.db"),
        add_history_to_context=True,
        telemetry=False,
    )
    run_output = agent.run("Hello")
    assert run_output.messages is not None
    assert len(run_output.messages) == 2

    run_output = agent.run("Hello 2")
    assert run_output.messages is not None
    assert len(run_output.messages) == 4

    run_output = agent.run("Hello 3")
    assert run_output.messages is not None
    assert len(run_output.messages) == 6

    run_output = agent.run("Hello 4")
    assert run_output.messages is not None
    assert len(run_output.messages) == 8


def test_client_persistence(openai_responses_model):
    """Test that the same OpenAI Responses client instance is reused across multiple calls"""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    # First call should create a new client
    agent.run("Hello")
    first_client = openai_responses_model.client
    assert first_client is not None

    # Second call should reuse the same client
    agent.run("Hello again")
    second_client = openai_responses_model.client
    assert second_client is not None
    assert first_client is second_client, "Client should be persisted and reused"

    # Third call should also reuse the same client
    agent.run("Hello once more")
    third_client = openai_responses_model.client
    assert third_client is not None
    assert first_client is third_client, "Client should still be the same instance"


@pytest.mark.asyncio
async def test_async_client_persistence(openai_responses_model):
    """Test that the same async OpenAI Responses client instance is reused across multiple calls"""
    agent = Agent(model=openai_responses_model, markdown=True, telemetry=False)

    # First call should create a new async client
    await agent.arun("Hello")
    first_client = openai_responses_model.async_client
    assert first_client is not None

    # Second call should reuse the same async client
    await agent.arun("Hello again")
    second_client = openai_responses_model.async_client
    assert second_client is not None
    assert first_client is second_client, "Async client should be persisted and reused"

    # Third call should also reuse the same async client
    await agent.arun("Hello once more")
    third_client = openai_responses_model.async_client
    assert third_client is not None
    assert first_client is third_client, "Async client should still be the same instance"
