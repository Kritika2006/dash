import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_risk_from_financials(revenue: float, expenses: float, liabilities: float, cash: float) -> Dict[str, float]:
    """Compute simple heuristic risk factors from aggregate financials.

    - liquidity_risk: low cash relative to monthly expenses
    - leverage_risk: high liabilities relative to revenue
    - burn_risk: expenses exceeding revenue
    """
    monthly_expenses = expenses / 12 if expenses else 0.0
    liquidity_risk = 1.0 if monthly_expenses == 0 else max(0.0, 1.0 - (cash / (monthly_expenses * 6)))
    leverage_risk = 0.0 if revenue == 0 else min(1.0, liabilities / max(revenue, 1e-6))
    burn_risk = 0.0 if revenue == 0 else min(1.0, max(0.0, (expenses - revenue) / max(revenue, 1e-6)))
    return {
        "liquidity_risk": round(liquidity_risk, 3),
        "leverage_risk": round(leverage_risk, 3),
        "burn_risk": round(burn_risk, 3),
    }


def summarize_risk(factors: Dict[str, float]) -> Dict[str, object]:
    """Summarize factor scores into an overall score and qualitative level."""
    weights = {
        "liquidity_risk": 0.4,
        "leverage_risk": 0.3,
        "burn_risk": 0.3,
    }
    score = sum(factors[name] * weight for name, weight in weights.items())
    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"
    return {"score": round(score, 3), "level": level, "factors": factors}


def compute_risk_from_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Try to derive aggregate numbers from a generic dataframe and compute risk."""
    revenue = float(df.get("Revenue", pd.Series(dtype=float)).sum()) if "Revenue" in df.columns else 0.0
    expenses = float(df.get("Expenses", pd.Series(dtype=float)).sum()) if "Expenses" in df.columns else 0.0
    liabilities = float(df.get("Liabilities", pd.Series(dtype=float)).sum()) if "Liabilities" in df.columns else 0.0
    cash = float(df.get("Cash", pd.Series(dtype=float)).sum()) if "Cash" in df.columns else 0.0

    factors = compute_risk_from_financials(revenue, expenses, liabilities, cash)
    return summarize_risk(factors)


def monte_carlo_cash_flow_simulation(
    initial_cash: float,
    monthly_revenue_mean: float,
    monthly_revenue_std: float,
    monthly_expenses_mean: float,
    monthly_expenses_std: float,
    months: int = 12,
    simulations: int = 10000
) -> Dict[str, any]:
    """
    Monte Carlo simulation for cash flow forecasting with risk assessment.
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate random scenarios
    revenue_scenarios = np.random.normal(monthly_revenue_mean, monthly_revenue_std, (simulations, months))
    expense_scenarios = np.random.normal(monthly_expenses_mean, monthly_expenses_std, (simulations, months))
    
    # Ensure non-negative values
    revenue_scenarios = np.maximum(revenue_scenarios, 0)
    expense_scenarios = np.maximum(expense_scenarios, 0)
    
    # Calculate cash flow for each scenario
    cash_flow_scenarios = np.zeros((simulations, months))
    cash_flow_scenarios[:, 0] = initial_cash + revenue_scenarios[:, 0] - expense_scenarios[:, 0]
    
    for month in range(1, months):
        cash_flow_scenarios[:, month] = (
            cash_flow_scenarios[:, month-1] + 
            revenue_scenarios[:, month] - 
            expense_scenarios[:, month]
        )
    
    # Calculate statistics
    final_cash_distribution = cash_flow_scenarios[:, -1]
    
    # Value at Risk (VaR) calculations
    var_95 = np.percentile(final_cash_distribution, 5)  # 95% VaR
    var_99 = np.percentile(final_cash_distribution, 1)  # 99% VaR
    
    # Probability of running out of cash
    bankruptcy_prob = np.mean(np.any(cash_flow_scenarios < 0, axis=1))
    
    # Expected value and confidence intervals
    expected_final_cash = np.mean(final_cash_distribution)
    ci_95_lower = np.percentile(final_cash_distribution, 2.5)
    ci_95_upper = np.percentile(final_cash_distribution, 97.5)
    
    return {
        "expected_final_cash": round(expected_final_cash, 2),
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "bankruptcy_probability": round(bankruptcy_prob, 4),
        "confidence_interval_95": [round(ci_95_lower, 2), round(ci_95_upper, 2)],
        "cash_flow_scenarios": cash_flow_scenarios.tolist(),
        "final_cash_distribution": final_cash_distribution.tolist()
    }


def calculate_financial_ratios(revenue: float, expenses: float, liabilities: float, cash: float) -> Dict[str, float]:
    """Calculate comprehensive financial ratios for risk assessment."""
    ratios = {}
    
    # Liquidity ratios
    monthly_expenses = expenses / 12 if expenses > 0 else 1
    ratios["cash_runway_months"] = cash / monthly_expenses if monthly_expenses > 0 else 0
    
    # Leverage ratios
    ratios["debt_to_revenue"] = liabilities / revenue if revenue > 0 else 0
    ratios["debt_to_equity"] = liabilities / max(cash, 1) if cash > 0 else 0
    
    # Profitability ratios
    profit = revenue - expenses
    ratios["profit_margin"] = profit / revenue if revenue > 0 else 0
    ratios["expense_ratio"] = expenses / revenue if revenue > 0 else 0
    
    # Efficiency ratios
    ratios["revenue_per_month"] = revenue / 12
    ratios["expense_per_month"] = expenses / 12
    
    return {k: round(v, 4) for k, v in ratios.items()}


def stress_test_scenarios(
    base_revenue: float,
    base_expenses: float,
    base_cash: float,
    scenarios: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Stress test different economic scenarios.
    scenarios: List of dicts with 'revenue_change' and 'expense_change' as percentages
    """
    results = {}
    
    for i, scenario in enumerate(scenarios):
        scenario_name = scenario.get('name', f'Scenario_{i+1}')
        revenue_multiplier = 1 + scenario.get('revenue_change', 0) / 100
        expense_multiplier = 1 + scenario.get('expense_change', 0) / 100
        
        adjusted_revenue = base_revenue * revenue_multiplier
        adjusted_expenses = base_expenses * expense_multiplier
        
        # Calculate runway under this scenario
        monthly_burn = adjusted_expenses / 12
        runway_months = base_cash / monthly_burn if monthly_burn > 0 else float('inf')
        
        # Calculate profit/loss
        profit_loss = adjusted_revenue - adjusted_expenses
        
        results[scenario_name] = {
            "adjusted_revenue": round(adjusted_revenue, 2),
            "adjusted_expenses": round(adjusted_expenses, 2),
            "runway_months": round(runway_months, 2),
            "profit_loss": round(profit_loss, 2),
            "revenue_change_pct": scenario.get('revenue_change', 0),
            "expense_change_pct": scenario.get('expense_change', 0)
        }
    
    return results


def calculate_esg_risk_score(
    environmental_score: float = 0.5,
    social_score: float = 0.5,
    governance_score: float = 0.5
) -> Dict[str, any]:
    """
    Calculate ESG (Environmental, Social, Governance) risk score.
    Scores should be between 0-1, where 1 is best.
    """
    # Weighted ESG score
    weights = {"environmental": 0.3, "social": 0.3, "governance": 0.4}
    overall_esg_score = (
        environmental_score * weights["environmental"] +
        social_score * weights["social"] +
        governance_score * weights["governance"]
    )
    
    # Convert to risk level
    if overall_esg_score >= 0.8:
        risk_level = "low"
    elif overall_esg_score >= 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return {
        "overall_esg_score": round(overall_esg_score, 3),
        "risk_level": risk_level,
        "environmental_score": round(environmental_score, 3),
        "social_score": round(social_score, 3),
        "governance_score": round(governance_score, 3),
        "recommendations": _get_esg_recommendations(overall_esg_score)
    }


def _get_esg_recommendations(score: float) -> List[str]:
    """Generate ESG improvement recommendations based on score."""
    recommendations = []
    
    if score < 0.4:
        recommendations.extend([
            "Implement comprehensive ESG framework",
            "Conduct ESG risk assessment",
            "Develop sustainability reporting standards"
        ])
    elif score < 0.6:
        recommendations.extend([
            "Enhance governance structures",
            "Improve stakeholder engagement",
            "Strengthen environmental policies"
        ])
    elif score < 0.8:
        recommendations.extend([
            "Optimize ESG metrics tracking",
            "Enhance transparency reporting",
            "Consider ESG-linked financing"
        ])
    else:
        recommendations.extend([
            "Maintain ESG leadership position",
            "Share best practices with industry",
            "Explore ESG innovation opportunities"
        ])
    
    return recommendations


