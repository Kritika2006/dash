# cfo_dashboard/backend/services/analysis_service.py
import pandas as pd
from ..core.schemas import FinancialInput

def calculate_kpis(data: FinancialInput) -> dict:
    """Calculates key financial metrics."""
    profit = data.revenue - data.expenses
    debt_ratio = round(data.liabilities / data.revenue, 2) if data.revenue else 0
    runway = round((data.cash if data.cash else max(profit, 0)) / data.burn_rate, 2) if data.burn_rate else 0
    return {
        "profit": profit,
        "debt_ratio": debt_ratio,
        "burn_rate": data.burn_rate,
        "runway_months": runway
    }

def generate_advisory(kpis: dict) -> str:
    """Generates simple text-based financial advice."""
    messages = []
    if kpis.get("profit", 0) < 0:
        messages.append("⚠ Your expenses exceed revenue. Focus on reducing costs or boosting sales.")
    else:
        messages.append("✅ Profitable operations achieved. Well done!")

    if kpis.get("debt_ratio", 0) > 0.5:
        messages.append("⚠ High debt ratio suggests financial risk. Consider deleveraging.")
    else:
        messages.append("✅ Debt levels appear manageable.")

    if kpis.get("runway_months", 0) < 6:
        messages.append("⚠ Runway is short. Secure funding or significantly cut burn rate immediately.")
    else:
        messages.append("✅ You have a healthy runway.")

    return " ".join(messages)

# cfo_dashboard/backend/services/analysis_service.py

# ... (keep other functions like calculate_kpis and generate_advisory)

def process_dataframe(df: pd.DataFrame):
    """
    Processes the cleaned DataFrame from the Zomato report to extract 
    chart data and financial inputs for KPI calculation.
    """
    # Convert relevant columns to numeric, coercing errors
    for col in ['GOV', 'Adjusted Revenue', 'Adjusted EBITDA']:
        if col in df.columns:
            # Clean the string by removing commas, symbols, etc.
            df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Adjusted Revenue', 'Adjusted EBITDA'], inplace=True)

    # --- Create FinancialInput model from the Zomato table data ---
    # We sum the values for all business segments to get the total
    total_adjusted_revenue = float(df['Adjusted Revenue'].sum())
    total_adjusted_ebitda = float(df['Adjusted EBITDA'].sum())

    # We can approximate Expenses from Revenue and EBITDA
    # Expenses = Adjusted Revenue - Adjusted EBITDA
    total_expenses = total_adjusted_revenue - total_adjusted_ebitda
    
    financial_input = FinancialInput(
        revenue=total_adjusted_revenue,
        expenses=total_expenses,
        # These columns are not in the page 5 table, so we set them to 0 as placeholders.
        liabilities=0.0,
        burn_rate=total_expenses / 12 if total_expenses > 0 else 0.0, # Approximate monthly burn
        cash=0.0 
    )

    # --- Prepare chart data ---
    # For the expense chart, we can use the 'Business' column as the category
    expense_data = df[['Business', 'Adjusted Revenue']].copy()
    expense_data.rename(columns={"Adjusted Revenue": "Amount"}, inplace=True)

    # Runway data is a placeholder as we don't have a cash balance from this table
    kpis = calculate_kpis(financial_input)
    runway_data = pd.DataFrame([
        {"Month": "Current", "Runway": kpis.get('runway_months', 0)}
    ])

    return financial_input, expense_data.to_dict(orient='records'), runway_data.to_dict(orient='records')