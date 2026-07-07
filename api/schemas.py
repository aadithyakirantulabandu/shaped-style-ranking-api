"""Generic interaction event schema shared across the API, rankers, and eval."""
from pydantic import BaseModel


class Event(BaseModel):
    user_id: str
    item_id: str
    timestamp: int
    event_type: str = "view"
