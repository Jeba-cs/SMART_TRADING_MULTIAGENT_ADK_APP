"""ADK Smart Trader Agents Package"""

# Import base classes with proper fallback
try:
    from google.adk.agents import Agent
    from google.adk.tools import Tool
    print("Using Google ADK classes")
except ImportError:
    print("Google ADK not available, using fallback classes")

    class Agent:
        def __init__(self, model=None, name=None, description=None, instructions=None, tools=None, **kwargs):
            self.model = model
            self.name = name
            self.description = description
            self.instructions = instructions
            self.tools = tools or []
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Tool:
        def __init__(self, name=None, description=None, **kwargs):
            self.name = name
            self.description = description
            for key, value in kwargs.items():
                setattr(self, key, value)

        async def call(self, *args, **kwargs):
            raise NotImplementedError("Subclasses must implement call method")

__all__ = ['Agent', 'Tool']
