from langchain.agents import Tool

# Basic calculator tool
calculator_tool = Tool(
    name="Calculator",
    func=lambda query: str(eval(query)),
    description="Useful for doing basic math calculations. Input should be a valid arithmetic expression."
)

def get_tools():
    return [calculator_tool]
