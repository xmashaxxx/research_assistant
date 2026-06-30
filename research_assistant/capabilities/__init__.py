# Import all capability modules so their @register decorators fire on package import.
from research_assistant.capabilities import search, fetch_paper, extract, compare, synthesize, relate_to_project  # noqa: F401
