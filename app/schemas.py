from pydantic import BaseModel, Field

class ClientData(BaseModel):
    age: int = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    balance: int
    housing: str
    loan: str
    campaign: int = Field(ge=1)