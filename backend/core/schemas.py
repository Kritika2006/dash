# cfo_dashboard/backend/core/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import pandas as pd

# Schema for incoming data from the frontend
class FinancialInput(BaseModel):
    revenue: float
    expenses: float
    liabilities: float
    burn_rate: float
    cash: Optional[float] = None

# Schema for the /ask endpoint payload
class AskPayload(BaseModel):
    question: str
    finance: FinancialInput

# Schema for the full analysis response to the frontend
class AnalysisResult(BaseModel):
    kpis: Dict[str, Any]
    advisory: str
    expense_chart_data: List[Dict[str, Any]]
    runway_chart_data: List[Dict[str, Any]]
    data_preview: List[Dict[str, Any]]
    risk: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class TimeSeriesData(BaseModel):
    records: List[Dict[str, Any]]

class ForecastRequest(BaseModel):
    model: Literal["arima", "prophet", "random_forest", "gradient_boosting"] = "arima"
    records: List[Dict[str, Any]]

class RiskResult(BaseModel):
    score: float
    level: Literal["low", "medium", "high"]
    factors: Dict[str, float]

class MonteCarloRequest(BaseModel):
    initial_cash: float
    monthly_revenue_mean: float
    monthly_revenue_std: float
    monthly_expenses_mean: float
    monthly_expenses_std: float
    months: int = 12
    simulations: int = 10000

class StressTestRequest(BaseModel):
    base_revenue: float
    base_expenses: float
    base_cash: float
    scenarios: List[Dict[str, float]]

class ESGRequest(BaseModel):
    environmental_score: float = 0.5
    social_score: float = 0.5
    governance_score: float = 0.5