from typing import List
from langchain.agents import tool

CONTACTS = [{"name": "shai", "phone": "+972545579687"}]


@tool("get_all_contacts")
def get_all_contacts(placeholder: str) -> List[dict]:
    """Get contacts."""
    return CONTACTS
