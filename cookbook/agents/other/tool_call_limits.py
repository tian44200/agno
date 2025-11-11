"""
This cookbook shows how to use tool_call_limits to control the number of calls per tool.
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.yfinance import YFinanceTools

# Example 1: Using tool_call_limits with YFinanceTools
agent = Agent(
    model=Claude(id="claude-3-5-haiku-20241022"),
    tools=[YFinanceTools(company_news=True, cache_results=True)],
    tool_call_limits={
        "get_current_stock_price": 1,
        "get_company_news": 1,
    },
)

# It should only call each tool once.
agent.print_response(
    "Find me the current price of TSLA and AAPL, then get the latest news about Tesla and Google.",
    stream=True,
)

# Example 2: Using tool_call_limits with custom function tools
def format_text(text: str) -> str:
    """Format text to uppercase."""
    return text.upper()

def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

agent_with_custom = Agent(
    model=Claude(id="claude-3-5-haiku-20241022"),
    tools=[format_text, reverse_string],
    tool_call_limits={
        "format_text": 1,
        "reverse_string": 1,
    },
)

# It should only call each tool once.
agent_with_custom.print_response(
    "Format apple and pear to uppercase, then reverse the string banana and cherry.",
    stream=True,
)
