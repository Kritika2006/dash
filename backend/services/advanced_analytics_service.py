# cfo_dashboard/backend/services/advanced_analytics_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def detect_anomalies(data: pd.DataFrame, columns: List[str], contamination: float = 0.1) -> Dict[str, any]:
    """
    Detect anomalies in financial data using Isolation Forest.
    """
    if len(data) < 10:
        return {"anomalies": [], "anomaly_score": 0, "message": "Insufficient data for anomaly detection"}
    
    # Prepare data for anomaly detection
    numeric_data = data[columns].select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return {"anomalies": [], "anomaly_score": 0, "message": "No numeric columns found"}
    
    # Fill missing values
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_data)
    anomaly_scores = iso_forest.decision_function(scaled_data)
    
    # Get anomaly indices
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    
    # Prepare results
    anomalies = []
    for idx in anomaly_indices:
        anomaly_data = {
            "index": int(idx),
            "score": float(anomaly_scores[idx]),
            "values": numeric_data.iloc[idx].to_dict()
        }
        anomalies.append(anomaly_data)
    
    # Calculate overall anomaly score
    overall_score = len(anomalies) / len(data)
    
    return {
        "anomalies": anomalies,
        "anomaly_score": round(overall_score, 4),
        "total_anomalies": len(anomalies),
        "total_records": len(data)
    }


def calculate_correlations(data: pd.DataFrame, target_column: str) -> Dict[str, any]:
    """
    Calculate correlations between target column and other numeric columns.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if target_column not in numeric_data.columns:
        return {"correlations": {}, "message": f"Target column '{target_column}' not found in numeric data"}
    
    correlations = {}
    target_series = numeric_data[target_column]
    
    for col in numeric_data.columns:
        if col != target_column:
            try:
                corr_coef, p_value = stats.pearsonr(target_series, numeric_data[col])
                correlations[col] = {
                    "correlation": round(corr_coef, 4),
                    "p_value": round(p_value, 4),
                    "strength": _get_correlation_strength(abs(corr_coef)),
                    "significant": p_value < 0.05
                }
            except:
                correlations[col] = {
                    "correlation": 0,
                    "p_value": 1,
                    "strength": "no correlation",
                    "significant": False
                }
    
    # Sort by absolute correlation strength
    sorted_correlations = dict(sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    ))
    
    return {
        "correlations": sorted_correlations,
        "target_column": target_column,
        "total_variables": len(sorted_correlations)
    }


def _get_correlation_strength(abs_corr: float) -> str:
    """Determine correlation strength based on absolute value."""
    if abs_corr >= 0.8:
        return "very strong"
    elif abs_corr >= 0.6:
        return "strong"
    elif abs_corr >= 0.4:
        return "moderate"
    elif abs_corr >= 0.2:
        return "weak"
    else:
        return "very weak"


def analyze_trends(data: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, any]:
    """
    Analyze trends in time series data.
    """
    try:
        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column)
        
        # Calculate trend using linear regression
        x = np.arange(len(data))
        y = data[value_column].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate trend direction and strength
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        trend_strength = _get_trend_strength(abs(r_value))
        
        # Calculate percentage change
        first_value = y[0]
        last_value = y[-1]
        percentage_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
        
        # Calculate volatility (standard deviation)
        volatility = np.std(y)
        
        # Detect seasonality (simplified)
        if len(data) >= 12:  # At least 12 months of data
            monthly_data = data.set_index(date_column)[value_column].resample('M').mean()
            if len(monthly_data) >= 12:
                # Calculate coefficient of variation for seasonality
                cv = monthly_data.std() / monthly_data.mean() if monthly_data.mean() != 0 else 0
                seasonality_present = cv > 0.1  # Threshold for seasonality
            else:
                seasonality_present = False
        else:
            seasonality_present = False
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": round(slope, 4),
            "r_squared": round(r_value**2, 4),
            "p_value": round(p_value, 4),
            "percentage_change": round(percentage_change, 2),
            "volatility": round(volatility, 4),
            "seasonality_present": seasonality_present,
            "data_points": len(data),
            "time_span_days": (data[date_column].max() - data[date_column].min()).days
        }
        
    except Exception as e:
        return {"error": str(e), "message": "Failed to analyze trends"}


def _get_trend_strength(r_squared: float) -> str:
    """Determine trend strength based on R-squared value."""
    if r_squared >= 0.8:
        return "very strong"
    elif r_squared >= 0.6:
        return "strong"
    elif r_squared >= 0.4:
        return "moderate"
    elif r_squared >= 0.2:
        return "weak"
    else:
        return "very weak"


def perform_clustering_analysis(data: pd.DataFrame, columns: List[str], n_clusters: int = 3) -> Dict[str, any]:
    """
    Perform clustering analysis on financial data.
    """
    if len(data) < n_clusters:
        return {"clusters": [], "message": f"Insufficient data for {n_clusters} clusters"}
    
    # Prepare numeric data
    numeric_data = data[columns].select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return {"clusters": [], "message": "No numeric columns found for clustering"}
    
    # Fill missing values and standardize
    numeric_data = numeric_data.fillna(numeric_data.mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate cluster statistics
    clusters = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_data = numeric_data[cluster_mask]
        
        cluster_info = {
            "cluster_id": i,
            "size": int(cluster_mask.sum()),
            "percentage": round(cluster_mask.sum() / len(data) * 100, 2),
            "centroid": kmeans.cluster_centers_[i].tolist(),
            "mean_values": cluster_data.mean().to_dict(),
            "std_values": cluster_data.std().to_dict()
        }
        clusters.append(cluster_info)
    
    # Calculate silhouette score if possible
    try:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_score_val = round(silhouette_avg, 4)
    except:
        silhouette_score_val = None
    
    return {
        "clusters": clusters,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette_score_val,
        "total_records": len(data)
    }


def calculate_financial_health_score(
    revenue: float,
    expenses: float,
    liabilities: float,
    cash: float,
    growth_rate: float = 0.0
) -> Dict[str, any]:
    """
    Calculate a comprehensive financial health score.
    """
    scores = {}
    
    # Profitability Score (0-100)
    profit = revenue - expenses
    profit_margin = profit / revenue if revenue > 0 else 0
    profitability_score = min(100, max(0, (profit_margin + 0.2) * 250))  # Scale to 0-100
    
    # Liquidity Score (0-100)
    monthly_expenses = expenses / 12 if expenses > 0 else 1
    cash_runway_months = cash / monthly_expenses if monthly_expenses > 0 else 0
    liquidity_score = min(100, max(0, cash_runway_months * 10))  # Scale to 0-100
    
    # Leverage Score (0-100, inverted - lower leverage is better)
    debt_ratio = liabilities / revenue if revenue > 0 else 0
    leverage_score = max(0, 100 - (debt_ratio * 200))  # Scale to 0-100
    
    # Growth Score (0-100)
    growth_score = min(100, max(0, (growth_rate + 0.1) * 500))  # Scale to 0-100
    
    # Overall Health Score (weighted average)
    weights = {
        "profitability": 0.3,
        "liquidity": 0.25,
        "leverage": 0.25,
        "growth": 0.2
    }
    
    overall_score = (
        profitability_score * weights["profitability"] +
        liquidity_score * weights["liquidity"] +
        leverage_score * weights["leverage"] +
        growth_score * weights["growth"]
    )
    
    # Determine health level
    if overall_score >= 80:
        health_level = "excellent"
    elif overall_score >= 60:
        health_level = "good"
    elif overall_score >= 40:
        health_level = "fair"
    elif overall_score >= 20:
        health_level = "poor"
    else:
        health_level = "critical"
    
    return {
        "overall_score": round(overall_score, 2),
        "health_level": health_level,
        "component_scores": {
            "profitability": round(profitability_score, 2),
            "liquidity": round(liquidity_score, 2),
            "leverage": round(leverage_score, 2),
            "growth": round(growth_score, 2)
        },
        "recommendations": _get_health_recommendations(overall_score, health_level)
    }


def _get_health_recommendations(score: float, level: str) -> List[str]:
    """Generate recommendations based on financial health score."""
    recommendations = []
    
    if level == "critical":
        recommendations.extend([
            "Immediate cost reduction required",
            "Seek emergency funding",
            "Consider restructuring operations",
            "Implement strict cash management"
        ])
    elif level == "poor":
        recommendations.extend([
            "Focus on improving profitability",
            "Reduce unnecessary expenses",
            "Consider debt restructuring",
            "Improve cash flow management"
        ])
    elif level == "fair":
        recommendations.extend([
            "Optimize operational efficiency",
            "Strengthen financial controls",
            "Consider growth investments",
            "Monitor cash flow closely"
        ])
    elif level == "good":
        recommendations.extend([
            "Maintain current performance",
            "Consider strategic investments",
            "Optimize capital structure",
            "Plan for sustainable growth"
        ])
    else:  # excellent
        recommendations.extend([
            "Maintain excellence standards",
            "Consider expansion opportunities",
            "Optimize capital allocation",
            "Share best practices"
        ])
    
    return recommendations


def benchmark_analysis(
    company_metrics: Dict[str, float],
    industry_benchmarks: Dict[str, Dict[str, float]]
) -> Dict[str, any]:
    """
    Compare company metrics against industry benchmarks.
    """
    comparisons = {}
    
    for metric, company_value in company_metrics.items():
        if metric in industry_benchmarks:
            benchmark = industry_benchmarks[metric]
            percentile = _calculate_percentile(company_value, benchmark)
            
            comparisons[metric] = {
                "company_value": company_value,
                "industry_median": benchmark.get("median", 0),
                "industry_75th_percentile": benchmark.get("75th_percentile", 0),
                "industry_25th_percentile": benchmark.get("25th_percentile", 0),
                "percentile_rank": percentile,
                "performance": _get_performance_level(percentile)
            }
    
    return {
        "comparisons": comparisons,
        "overall_performance": _calculate_overall_performance(comparisons)
    }


def _calculate_percentile(value: float, benchmark: Dict[str, float]) -> float:
    """Calculate percentile rank of company value against industry benchmark."""
    median = benchmark.get("median", 0)
    p75 = benchmark.get("75th_percentile", median)
    p25 = benchmark.get("25th_percentile", median)
    
    if value >= p75:
        return 75 + ((value - p75) / (p75 - median)) * 25 if p75 > median else 100
    elif value >= median:
        return 50 + ((value - median) / (p75 - median)) * 25 if p75 > median else 50
    elif value >= p25:
        return 25 + ((value - p25) / (median - p25)) * 25 if median > p25 else 25
    else:
        return max(0, (value / p25) * 25) if p25 > 0 else 0


def _get_performance_level(percentile: float) -> str:
    """Determine performance level based on percentile."""
    if percentile >= 90:
        return "excellent"
    elif percentile >= 75:
        return "above average"
    elif percentile >= 50:
        return "average"
    elif percentile >= 25:
        return "below average"
    else:
        return "poor"


def _calculate_overall_performance(comparisons: Dict[str, Dict[str, any]]) -> str:
    """Calculate overall performance level."""
    if not comparisons:
        return "unknown"
    
    percentiles = [comp["percentile_rank"] for comp in comparisons.values()]
    avg_percentile = np.mean(percentiles)
    
    return _get_performance_level(avg_percentile)
