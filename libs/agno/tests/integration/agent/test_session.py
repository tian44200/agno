import uuid
from typing import Any, Dict, Optional

from agno.agent.agent import Agent
from agno.db.base import SessionType
from agno.models.openai.chat import OpenAIChat


def add_item(session_state: Dict[str, Any], item: str) -> str:
    """Add an item to the shopping list (sync version)."""
    session_state["shopping_list"].append(item)
    return f"The shopping list is now {session_state['shopping_list']}"


async def async_add_item(session_state: Dict[str, Any], item: str) -> str:
    """Add an item to the shopping list (async version)."""
    session_state["shopping_list"].append(item)
    return f"The shopping list is now {session_state['shopping_list']}"


def chat_agent_factory(shared_db, session_id: Optional[str] = None, session_state: Optional[Dict[str, Any]] = None):
    """Create an agent with storage and memory for testing."""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        db=shared_db,
        session_id=session_id or str(uuid.uuid4()),
        session_state=session_state or {},
    )


def test_agent_default_state(shared_db):
    session_id = "session_1"
    session_state = {"test_key": "test_value"}
    chat_agent = chat_agent_factory(shared_db, session_id, session_state)

    response = chat_agent.run("Hello, how are you?")

    assert response.run_id is not None

    assert chat_agent.session_id == session_id
    assert chat_agent.session_state == session_state

    session_from_storage = shared_db.get_session(session_id=session_id, session_type=SessionType.AGENT)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id
    assert session_from_storage.session_data["session_state"] == {
        "test_key": "test_value",
    }


def test_agent_set_session_name(shared_db):
    session_id = "session_1"
    chat_agent = chat_agent_factory(shared_db, session_id)

    chat_agent.run("Hello, how are you?")

    chat_agent.set_session_name(session_id=session_id, session_name="my_test_session")

    session_from_storage = shared_db.get_session(session_id=session_id, session_type=SessionType.AGENT)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id
    assert session_from_storage.session_data["session_name"] == "my_test_session"


def test_agent_get_session_name(shared_db):
    session_id = "session_1"
    chat_agent = chat_agent_factory(shared_db, session_id)
    chat_agent.run("Hello, how are you?")
    chat_agent.set_session_name(session_id=session_id, session_name="my_test_session")
    assert chat_agent.get_session_name() == "my_test_session"


def test_agent_get_session_state(shared_db):
    session_id = "session_1"
    chat_agent = chat_agent_factory(shared_db, session_id, session_state={"test_key": "test_value"})
    chat_agent.run("Hello, how are you?")
    assert chat_agent.get_session_state() == {"test_key": "test_value"}


def test_agent_get_session_metrics(shared_db):
    session_id = "session_1"
    chat_agent = chat_agent_factory(shared_db, session_id)
    chat_agent.run("Hello, how are you?")
    metrics = chat_agent.get_session_metrics()
    assert metrics is not None
    assert metrics.total_tokens > 0
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens


def test_agent_session_state_switch_session_id(shared_db):
    session_id_1 = "session_1"
    session_id_2 = "session_2"

    chat_agent = chat_agent_factory(shared_db, session_id_1, session_state={"test_key": "test_value"})

    # First run with a session ID (reset should not happen)
    chat_agent.run("What can you do?")
    session_from_storage = shared_db.get_session(session_id=session_id_1, session_type=SessionType.AGENT)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id_1
    assert session_from_storage.session_data["session_state"] == {"test_key": "test_value"}

    # Second run with different session ID, and override session state
    chat_agent.run("What can you do?", session_id=session_id_2, session_state={"test_key": "test_value_2"})
    session_from_storage = shared_db.get_session(session_id=session_id_2, session_type=SessionType.AGENT)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id_2
    assert session_from_storage.session_data["session_state"] == {"test_key": "test_value_2"}

    # Third run with the original session ID
    chat_agent.run("What can you do?", session_id=session_id_1)
    session_from_storage = shared_db.get_session(session_id=session_id_1, session_type=SessionType.AGENT)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id_1
    assert session_from_storage.session_data["session_state"] == {"test_key": "test_value"}


def test_agent_with_state_on_agent(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        session_state={"shopping_list": []},
        tools=[add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    agent.run("Add oranges to my shopping list")
    response = agent.run(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )
    assert (
        response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


def test_agent_with_state_on_agent_stream(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        session_state={"shopping_list": []},
        session_id=str(uuid.uuid4()),
        tools=[add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    for _ in agent.run("Add oranges to my shopping list", stream=True):
        pass

    session_from_storage = shared_db.get_session(session_id=agent.session_id, session_type=SessionType.AGENT)
    assert session_from_storage.session_data["session_state"] == {"shopping_list": ["oranges"]}

    for _ in agent.run(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```',
        stream=True,
    ):
        pass

    run_response = agent.get_last_run_output()
    assert (
        run_response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


def test_agent_with_state_on_run(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        tools=[add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    agent.run("Add oranges to my shopping list", session_id="session_1", session_state={"shopping_list": []})

    session_from_storage = shared_db.get_session(session_id="session_1", session_type=SessionType.AGENT)
    assert session_from_storage.session_data["session_state"] == {"shopping_list": ["oranges"]}

    response = agent.run(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```',
        session_id="session_1",
    )

    assert (
        response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


def test_agent_with_state_on_run_stream(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        tools=[add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    for response in agent.run(
        "Add oranges to my shopping list", session_id="session_1", session_state={"shopping_list": []}, stream=True
    ):
        pass

    session_from_storage = shared_db.get_session(session_id="session_1", session_type=SessionType.AGENT)
    assert session_from_storage.session_data["session_state"] == {"shopping_list": ["oranges"]}

    for response in agent.run(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```',
        session_id="session_1",
        stream=True,
    ):
        pass

    run_response = agent.get_last_run_output(session_id="session_1")
    assert (
        run_response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


async def test_agent_with_state_on_run_async(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        tools=[async_add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    await agent.arun("Add oranges to my shopping list", session_id="session_1", session_state={"shopping_list": []})

    session_from_storage = shared_db.get_session(session_id="session_1", session_type=SessionType.AGENT)
    assert session_from_storage.session_data["session_state"] == {"shopping_list": ["oranges"]}

    response = await agent.arun(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```',
        session_id="session_1",
    )

    assert (
        response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


async def test_agent_with_state_on_run_stream_async(shared_db):
    # Create an Agent that maintains state
    agent = Agent(
        db=shared_db,
        tools=[async_add_item],
        instructions="Current state (shopping list) is: {shopping_list}",
        markdown=True,
    )
    async for response in agent.arun(
        "Add oranges to my shopping list", session_id="session_1", session_state={"shopping_list": []}, stream=True
    ):
        pass

    session_from_storage = shared_db.get_session(session_id="session_1", session_type=SessionType.AGENT)
    assert session_from_storage.session_data["session_state"] == {"shopping_list": ["oranges"]}

    async for response in agent.arun(
        'Current shopping list: {shopping_list}. Other random json ```json { "properties": { "title": { "title": "a" } } }```',
        session_id="session_1",
        stream=True,
    ):
        pass

    run_response = agent.get_last_run_output(session_id="session_1")
    assert (
        run_response.messages[1].content
        == 'Current shopping list: [\'oranges\']. Other random json ```json { "properties": { "title": { "title": "a" } } }```'
    )


def test_add_session_state_to_context(shared_db):
    agent = Agent(
        db=shared_db,
        session_state={"shopping_list": ["oranges"]},
        markdown=True,
        add_session_state_to_context=True,
    )
    response = agent.run("What is in my shopping list?")
    assert response is not None
    assert response.messages is not None

    # Check the system message
    assert "'shopping_list': ['oranges']" in response.messages[0].content

    assert "oranges" in response.content.lower()


def test_session_state_in_run_output(shared_db):
    """Test that RunOutput contains the updated session_state in non-streaming mode."""
    session_id = str(uuid.uuid4())
    agent = Agent(
        db=shared_db,
        session_id=session_id,
        session_state={"shopping_list": []},
        tools=[add_item],
        instructions="You help manage shopping lists.",
        markdown=True,
    )

    response = agent.run("Add apples to my shopping list")

    # Verify RunOutput has session_state field
    assert response.session_state is not None, "RunOutput should have session_state"
    assert isinstance(response.session_state, dict), "session_state should be a dict"
    assert "shopping_list" in response.session_state, "shopping_list key should be present"
    assert isinstance(response.session_state["shopping_list"], list), "shopping_list should be a list"

    # Verify state was updated by the tool
    assert len(response.session_state.get("shopping_list", [])) == 1, "Shopping list should have 1 item"
    assert "apples" in response.session_state["shopping_list"], "Shopping list should contain apples"


def test_session_state_in_run_completed_event_stream(shared_db):
    """Test that RunCompletedEvent contains session_state in streaming mode."""
    session_id = str(uuid.uuid4())
    agent = Agent(
        db=shared_db,
        session_id=session_id,
        session_state={"shopping_list": ["bananas"]},
        tools=[add_item],
        instructions="You help manage shopping lists.",
        markdown=True,
    )

    run_completed_event = None

    for event in agent.run("Add oranges to my shopping list", stream=True, stream_intermediate_steps=True):
        if hasattr(event, "event") and event.event == "RunCompleted":
            run_completed_event = event
            break

    # Verify RunCompletedEvent structure
    assert run_completed_event is not None, "Should receive RunCompleted event"
    assert run_completed_event.session_state is not None, "RunCompletedEvent should have session_state"
    assert isinstance(run_completed_event.session_state, dict), "session_state should be a dict"
    assert "shopping_list" in run_completed_event.session_state, "shopping_list key should be present"
    assert "bananas" in run_completed_event.session_state.get("shopping_list", []), "Initial item should be preserved"

    # Verify state was updated by the tool
    assert len(run_completed_event.session_state.get("shopping_list", [])) == 2, "Shopping list should have 2 items"
    assert "oranges" in run_completed_event.session_state["shopping_list"], "Shopping list should contain oranges"


async def test_session_state_in_run_output_async(shared_db):
    """Test that RunOutput contains session_state in async non-streaming mode."""
    session_id = str(uuid.uuid4())
    agent = Agent(
        db=shared_db,
        session_id=session_id,
        session_state={"shopping_list": []},
        tools=[async_add_item],
        instructions="You help manage shopping lists.",
        markdown=True,
    )

    response = await agent.arun("Add apples to my shopping list")

    # Verify RunOutput has session_state
    assert response.session_state is not None, "RunOutput should have session_state"
    assert isinstance(response.session_state, dict), "session_state should be a dict"
    assert "shopping_list" in response.session_state, "shopping_list key should be present"
    assert isinstance(response.session_state["shopping_list"], list), "shopping_list should be a list"

    # Verify state was updated by the tool
    assert len(response.session_state.get("shopping_list", [])) == 1, "Shopping list should have 1 item"
    assert "apples" in response.session_state["shopping_list"], "Shopping list should contain apples"


async def test_session_state_in_run_completed_event_stream_async(shared_db):
    """Test that RunCompletedEvent contains session_state in async streaming mode."""
    session_id = str(uuid.uuid4())
    agent = Agent(
        db=shared_db,
        session_id=session_id,
        session_state={"shopping_list": ["bananas"]},
        tools=[async_add_item],
        instructions="You help manage shopping lists.",
        markdown=True,
    )

    run_completed_event = None

    async for event in agent.arun("Add oranges to my shopping list", stream=True, stream_intermediate_steps=True):
        if hasattr(event, "event") and event.event == "RunCompleted":
            run_completed_event = event
            break

    # Verify RunCompletedEvent structure
    assert run_completed_event is not None, "Should receive RunCompleted event"
    assert run_completed_event.session_state is not None, "RunCompletedEvent should have session_state"
    assert isinstance(run_completed_event.session_state, dict), "session_state should be a dict"
    assert "shopping_list" in run_completed_event.session_state, "shopping_list key should be present"
    assert "bananas" in run_completed_event.session_state.get("shopping_list", []), "Initial item should be preserved"

    # Verify state was updated by the tool
    assert len(run_completed_event.session_state.get("shopping_list", [])) == 2, "Shopping list should have 2 items"
    assert "oranges" in run_completed_event.session_state["shopping_list"], "Shopping list should contain oranges"
