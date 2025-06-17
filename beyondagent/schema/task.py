from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(default=...)

    env_type: str = Field(default="")

    metadata: dict = Field(default_factory=dict)