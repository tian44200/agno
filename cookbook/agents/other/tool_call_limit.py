"""
This cookbook shows how to use tool call limit to control the number of tool calls an agent can make.
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.yfinance import YFinanceTools

# Example 1: Using tool_call_limit (global limit for all tools)
agent = Agent(
    model=Claude(id="claude-3-5-haiku-20241022"),
    tools=[YFinanceTools(company_news=True, cache_results=True)],
    tool_call_limit=1,
)

# It should only call the first tool and fail to call the second tool.
agent.print_response(
    "Find me the current price of TSLA, then after that find me the latest news about Tesla.",
    stream=True,
)

# Example 2: Using tool_call_limits (per-tool limits)
# This allows you to set different call limits for different tools
agent_with_limits = Agent(
    model=Claude(id="claude-3-5-haiku-20241022"),
    tools=[YFinanceTools(company_news=True, cache_results=True)],
    tool_call_limits={
        "get_stock_price": 1,
        "get_company_news": 1,
    },
)

# This will respect the per-tool limits defined above, allowing one call to each tool.
agent_with_limits.print_response(
    "Find me the current price of TSLA and AAPL, then get the latest news about Tesla and Google.",
    stream=True,
)
