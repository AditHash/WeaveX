"""Central ID generation.

One place decides what an ID looks like, so every model stays consistent
and the format can change later without touching six files.
"""

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Generate a readable, unique ID like 'entity_a3f9c1d2'.

    Short uuid4 hex (8 chars) is enough entropy for this project's scale
    and stays short enough to read in logs/debuggers, unlike a full uuid4.
    """
    return f"{prefix}_{uuid4().hex[:8]}"
