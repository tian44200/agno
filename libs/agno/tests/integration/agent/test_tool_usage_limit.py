import pytest

from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.agent import RunEvent
from agno.tools.yfinance import YFinanceTools


def test_tool_usage_limit_single_tool():
    """Test that a tool with usage_limit=1 is only called once."""
    # Create YFinanceTools and set usage_limit on specific tools
    yfinance_tools = YFinanceTools(cache_results=True)

    # Set usage_limit on the get_current_stock_price function
    for tool in yfinance_tools.functions.values():
        if tool.name == "get_current_stock_price":
            tool.usage_limit = 1
            break

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[yfinance_tools],
        markdown=True,
        telemetry=False,
    )

    response = agent.run("Find me the current price of TSLA twice. Call it once then later again.")

    # Verify that get_current_stock_price was only called once due to usage_limit
    stock_price_calls = [t for t in response.tools if t.tool_name == "get_current_stock_price"]
    assert len(stock_price_calls) == 1
    assert stock_price_calls[0].tool_args == {"symbol": "TSLA"}
    assert stock_price_calls[0].result is not None
    assert response.content is not None


def test_tool_usage_limit_stream():
    """Test that tool usage limits work with streaming."""
    # Create YFinanceTools and set usage_limit on specific tools
    yfinance_tools = YFinanceTools(cache_results=True)

    # Set usage_limit on the get_current_stock_price function
    for tool in yfinance_tools.functions.values():
        if tool.name == "get_current_stock_price":
            tool.usage_limit = 1
            break

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[yfinance_tools],
        markdown=True,
        telemetry=False,
    )

    response_stream = agent.run(
        "Find me the current price of TSLA twice. Call it once then later again.", stream=True, stream_events=True
    )

    tools = []
    for chunk in response_stream:
        if chunk.event == RunEvent.tool_call_completed:
            tools.append(chunk.tool)

    stock_price_calls = [t for t in tools if t.tool_name == "get_current_stock_price"]
    assert len(stock_price_calls) == 1, f"Expected 1 stock price call, got {len(stock_price_calls)}"
    assert stock_price_calls[0].tool_args == {"symbol": "TSLA"}
    assert stock_price_calls[0].result is not None


@pytest.mark.asyncio
async def test_tool_usage_limit_async():
    """Test that tool usage limits work with async."""
    # Create YFinanceTools and set usage_limit on specific tools
    yfinance_tools = YFinanceTools(cache_results=True)

    # Set usage_limit on the get_current_stock_price function
    for tool in yfinance_tools.functions.values():
        if tool.name == "get_current_stock_price":
            tool.usage_limit = 1
            break

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[yfinance_tools],
        markdown=True,
        telemetry=False,
    )

    response = await agent.arun("Find me the current price of TSLA twice. Call it once then later again.")

    # Verify that get_current_stock_price was only called once due to usage_limit
    stock_price_calls = [t for t in response.tools if t.tool_name == "get_current_stock_price"]
    assert len(stock_price_calls) == 1
    assert stock_price_calls[0].tool_args == {"symbol": "TSLA"}
    assert stock_price_calls[0].result is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_tool_usage_limit_stream_async():
    """Test that tool usage limits work with async streaming."""
    # Create YFinanceTools and set usage_limit on specific tools
    yfinance_tools = YFinanceTools(cache_results=True)

    # Set usage_limit on the get_current_stock_price function
    for tool in yfinance_tools.functions.values():
        if tool.name == "get_current_stock_price":
            tool.usage_limit = 1
            break

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[yfinance_tools],
        markdown=True,
        telemetry=False,
    )

    tools = []
    async for chunk in agent.arun(
        "Find me the current price of TSLA twice. Call it once then later again.", stream=True, stream_events=True
    ):
        if chunk.event == RunEvent.tool_call_completed:
            tools.append(chunk.tool)

    stock_price_calls = [t for t in tools if t.tool_name == "get_current_stock_price"]
    assert len(stock_price_calls) == 1, f"Expected 1 stock price call, got {len(stock_price_calls)}"
    assert stock_price_calls[0].tool_args == {"symbol": "TSLA"}
    assert stock_price_calls[0].result is not None
