# cfo_dashboard/frontend/utils/api_client.py
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000" # Or your Docker service name later

def get_dashboard_data(uploaded_file):
    """Posts a file to the backend and gets all dashboard data."""
    if uploaded_file is None:
        return None
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/process-file", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
        return None

def ask_ai_assistant(question: str, financial_context: dict):
    """Posts a question to the AI assistant."""
    payload = {"question": question, "finance": financial_context}
    try:
        response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get response from AI: {e}")
        return {"answer": "Error connecting to the AI model."}
    
# cfo_dashboard/frontend/utils/api_client.py
# ... (keep your existing get_dashboard_data and ask_ai_assistant functions) ...

# cfo_dashboard/frontend/utils/api_client.py
# ... (keep other functions) ...

def get_forecast_data(dataframe_records: list, model: str = "arima"):
    """Posts the cleaned dataframe records to the backend for forecasting."""
    if not dataframe_records:
        return None
    payload = {"model": model, "records": dataframe_records}
    try:
        response = requests.post(f"{BACKEND_URL}/forecast", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not generate forecast: {e}")
        return None
    # cfo_dashboard/frontend/utils/api_client.py
# ... (keep other functions as they are) ...

def ask_ai_assistant(question: str, financial_context: dict):
    """Posts a question to the AI assistant."""
    payload = {"question": question, "finance": financial_context}
    try:
        # FIX: Increase the timeout from 30 seconds to 120 seconds
        response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get response from AI: {e}")
        return {"answer": f"Error communicating with the AI model: {e}"}


def run_stress_test(base_revenue: float, base_expenses: float, base_cash: float, scenarios: list):
    """Run stress test scenarios."""
    payload = {
        "base_revenue": base_revenue,
        "base_expenses": base_expenses,
        "base_cash": base_cash,
        "scenarios": scenarios
    }
    try:
        response = requests.post(f"{BACKEND_URL}/stress-test", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not run stress test: {e}")
        return {}


def run_monte_carlo(initial_cash: float, monthly_revenue_mean: float, monthly_revenue_std: float,
                   monthly_expenses_mean: float, monthly_expenses_std: float, simulations: int = 10000):
    """Run Monte Carlo simulation."""
    payload = {
        "initial_cash": initial_cash,
        "monthly_revenue_mean": monthly_revenue_mean,
        "monthly_revenue_std": monthly_revenue_std,
        "monthly_expenses_mean": monthly_expenses_mean,
        "monthly_expenses_std": monthly_expenses_std,
        "simulations": simulations
    }
    try:
        response = requests.post(f"{BACKEND_URL}/monte-carlo", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not run Monte Carlo simulation: {e}")
        return {}


def calculate_esg_score(environmental_score: float, social_score: float, governance_score: float):
    """Calculate ESG risk score."""
    payload = {
        "environmental_score": environmental_score,
        "social_score": social_score,
        "governance_score": governance_score
    }
    try:
        response = requests.post(f"{BACKEND_URL}/esg-score", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not calculate ESG score: {e}")
        return {}


def get_financial_ratios(revenue: float, expenses: float, liabilities: float, cash: float):
    """Get comprehensive financial ratios."""
    params = {
        "revenue": revenue,
        "expenses": expenses,
        "liabilities": liabilities,
        "cash": cash
    }
    try:
        response = requests.post(f"{BACKEND_URL}/financial-ratios", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get financial ratios: {e}")
        return {}


def get_multiple_forecasts(dataframe_records: list):
    """Get forecasts from multiple models for comparison."""
    payload = {"records": dataframe_records}
    try:
        response = requests.post(f"{BACKEND_URL}/multiple-forecasts", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get multiple forecasts: {e}")
        return {}


def load_sample_data():
    """Loads the sample dataset for testing."""
    sample_data_path = "c:\\Users\\kriti\\Downloads\\cfo_dashboard\\cfo_dashboard\\data\\Zomato_Annual_Report_2023-24.pdf"
    # Implement logic to extract and process the sample data
    # For now, return a placeholder dictionary
    return {
        "data_preview": [{"Date": "2023-01-01", "Revenue": 1000}, {"Date": "2023-02-01", "Revenue": 1200}],
        "kpis": {"runway_months": 12, "burn_rate": 100},
        "advisory": "Focus on reducing expenses.",
        "expense_chart_data": [{"Category": "Marketing", "Amount": 500}],
        "runway_chart_data": [{"Month": "Current", "Runway": 12}]
    }