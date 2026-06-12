from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import Optional

app = FastAPI()

df= pd.read_csv("dataset.csv")
df =df.fillna(0)

class ModelItem(BaseModel):
    model_name: str
    provider: str
    mmlu_score: float
    cost_per_1M_combined: float
    open_weights: bool

@app.get("/models")
def get_models(limit: int= 10, provider: Optional[str]=None):
    res =df
    if provider:
        res = res[res['provider']== provider]
    return res.head(limit).to_dict(orient="records")

@app.post("/models")
def add_model(item: ModelItem):
    global df
    new_row =pd.DataFrame([item.dict()])
    df = pd.concat([df, new_row], ignore_index=True)
    df= df.fillna(0)
    df.to_csv("dataset.csv", index=False)
    return {"status": "ok"}