# Import all capability modules so their @register decorators fire on package import.
from research_assistant.capabilities import search, fetch, extract, compare, synthesize  # noqa: F401
