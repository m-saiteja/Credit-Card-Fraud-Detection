from pydantic import (
    BaseModel, 
    Field,
    ConfigDict,
    create_model
)

from src.config import PCA_COLS

class TransactionBase(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 10.0,
                "Amount": 123.45,
                **{col: 0.0 for col in PCA_COLS}
            }
        }
    )

TransactionRequest = create_model(
    "TransactionRequest",
    __base__=TransactionBase,
    **{col: (float, ...) for col in PCA_COLS}
)

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold: float
    model_version: str
