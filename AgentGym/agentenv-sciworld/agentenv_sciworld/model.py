from pydantic import BaseModel
from typing import Optional


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: int
    # Optional: ask ScienceWorld to generate a gold action sequence ("gold path")
    # Note: gold path is not guaranteed to be optimal, and generation may fail for some tasks.
    generate_gold_path: bool = False
    # Optional: simplification string, e.g. "easy" or "teleportAction,openDoors"
    simplification_str: str = ""

class CloseRequestBody(BaseModel):
    id: int