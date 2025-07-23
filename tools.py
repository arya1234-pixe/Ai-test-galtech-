from langchain.agents import Tool

# Example function for a tool
def simple_calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define the tool using LangChain's Tool class
calculator_tool = Tool(
    name="Simple Calculator",
    func=simple_calculator,
    description="Useful for evaluating basic math expressions. Input should be like '2 + 2', '10 / 5', etc."
)

# This function is what your code is trying to import
def get_tools():
    return [calculator_tool]
