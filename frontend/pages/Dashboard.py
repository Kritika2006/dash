# cfo_dashboard/frontend/pages/1_📊_Dashboard.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# --- PATH HACK (FIXES IMPORT ERROR) ---
# Add the project root to the Python path
# This allows us to import from the `utils` directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from frontend.utils.ui_components import render_sidebar
from frontend.utils import api_client
# ------------------------------------

st.set_page_config(page_title="Dashboard", layout="wide")

# --- Render the common sidebar ---
render_sidebar()

st.title("📊 Advanced Financial Dashboard")
st.markdown("### Your AI-Powered Financial Command Center")

# Check if data exists in session state
if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
    st.warning("📂 No financial data loaded.")
    st.info("""
    **To get started:**
    1. Go to the **Home** page to auto-load the sample Zomato data, or
    2. Upload your own financial document below
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📂 Upload Financial Report",
        type=["csv", "xlsx", "xls", "pdf"],
        help="Upload your financial data file"
    )
    
    if uploaded_file:
        with st.spinner("AI is analyzing your document..."):
            try:
                st.session_state.analysis_results = api_client.get_dashboard_data(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_object = uploaded_file
                st.success(f"✅ Successfully analyzed: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Go to Home Page", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
    
    st.stop()

# Get analysis results
results = st.session_state.analysis_results
kpis = results.get('kpis', {})
risk = results.get('risk', {})

# Enhanced metrics display with better styling
st.subheader("🎯 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    profit = kpis.get('profit', 0)
    profit_delta = "positive" if profit > 0 else "negative"
    st.metric(
        "💰 Profit", 
        f"${profit:,.0f}",
        delta=f"{profit_delta}",
        help="Net profit after all expenses"
    )

with col2:
    debt_ratio = kpis.get('debt_ratio', 0)
    debt_status = "Low" if debt_ratio < 0.3 else "Medium" if debt_ratio < 0.5 else "High"
    st.metric(
        "📊 Debt Ratio", 
        f"{debt_ratio:.2f}",
        delta=f"{debt_status} Risk",
        help="Total liabilities divided by revenue"
    )

with col3:
    burn_rate = kpis.get('burn_rate', 0)
    st.metric(
        "🔥 Burn Rate", 
        f"${burn_rate:,.0f}/mo",
        help="Monthly cash consumption rate"
    )

with col4:
    runway = kpis.get('runway_months', 0)
    runway_status = "Healthy" if runway > 12 else "Concerning" if runway > 6 else "Critical"
    st.metric(
        "⏰ Runway", 
        f"{runway:.1f} months",
        delta=f"{runway_status}",
        help="Months of cash remaining at current burn rate"
    )

with col5:
    if risk:
        risk_level = risk.get('level', 'unknown').title()
        risk_score = risk.get('score', 0)
        risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
        st.metric(
            "⚠️ Risk Level", 
            risk_level,
            delta=f"Score: {risk_score:.2f}",
            help="Overall financial risk assessment"
        )

# Risk visualization
if risk:
    st.subheader("🎯 Risk Analysis")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        # Risk factors radar chart
        factors = risk.get('factors', {})
        if factors:
            fig = go.Figure(data=go.Scatterpolar(
                r=list(factors.values()),
                theta=list(factors.keys()),
                fill='toself',
                name='Risk Factors'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Risk Factor Analysis",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with risk_col2:
        # Risk level indicator
        risk_level = risk.get('level', 'unknown')
        risk_score = risk.get('score', 0)
        
        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Risk Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Enhanced charts section
st.subheader("📈 Financial Analysis")
tabs = st.tabs(["💰 Revenue & Expenses", "📊 Runway Analysis", "🔍 Data Insights", "📋 Raw Data"])

with tabs[0]:
    expense_records = results.get('expense_chart_data', [])
    if expense_records:
        exp_df = pd.DataFrame(expense_records)
        
        # Try to standardize column names
        name_col = 'Category' if 'Category' in exp_df.columns else ('Business' if 'Business' in exp_df.columns else None)
        value_col = 'Amount' if 'Amount' in exp_df.columns else None
        
        if name_col and value_col:
            # Create a more sophisticated chart
            fig = px.bar(
                exp_df, 
                x=name_col, 
                y=value_col,
                title="Revenue by Business Segment",
                color=value_col,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                xaxis_title=name_col,
                yaxis_title="Amount ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a pie chart for better visualization
            fig_pie = px.pie(
                exp_df, 
                values=value_col, 
                names=name_col,
                title="Revenue Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Expense chart data not in expected format.")
    else:
        st.info("No expense data available.")

with tabs[1]:
    runway_records = results.get('runway_chart_data', [])
    if runway_records:
        rw_df = pd.DataFrame(runway_records)
        name_col = 'Month' if 'Month' in rw_df.columns else None
        value_col = 'Runway' if 'Runway' in rw_df.columns else None
        
        if name_col and value_col:
            # Create an enhanced runway chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rw_df[name_col],
                y=rw_df[value_col],
                mode='lines+markers',
                name='Runway',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            
            # Add threshold lines
            fig.add_hline(y=12, line_dash="dash", line_color="green", 
                         annotation_text="Healthy (12+ months)")
            fig.add_hline(y=6, line_dash="dash", line_color="orange", 
                         annotation_text="Concerning (6 months)")
            fig.add_hline(y=3, line_dash="dash", line_color="red", 
                         annotation_text="Critical (3 months)")
            
            fig.update_layout(
                title="Cash Runway Analysis",
                xaxis_title=name_col,
                yaxis_title="Runway (months)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Runway data not in expected format.")
    else:
        st.info("No runway data available.")

with tabs[2]:
    st.subheader("🔍 Data Insights")
    
    # Financial health summary
    st.markdown("### 📊 Financial Health Summary")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        profit_margin = (kpis.get('profit', 0) / kpis.get('revenue', 1)) * 100 if kpis.get('revenue', 0) > 0 else 0
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    with health_col2:
        efficiency_ratio = (kpis.get('expenses', 0) / kpis.get('revenue', 1)) * 100 if kpis.get('revenue', 0) > 0 else 0
        st.metric("Expense Ratio", f"{efficiency_ratio:.1f}%")
    
    with health_col3:
        cash_ratio = (kpis.get('cash', 0) / kpis.get('expenses', 1)) * 100 if kpis.get('expenses', 0) > 0 else 0
        st.metric("Cash Ratio", f"{cash_ratio:.1f}%")
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    if profit_margin > 20:
        insights.append("✅ Excellent profit margin - above industry standards")
    elif profit_margin > 10:
        insights.append("📊 Good profit margin - room for improvement")
    else:
        insights.append("⚠️ Low profit margin - focus on cost optimization")
    
    if runway > 18:
        insights.append("✅ Strong cash position - consider growth investments")
    elif runway > 12:
        insights.append("📊 Adequate cash runway - monitor closely")
    else:
        insights.append("🚨 Short cash runway - immediate action required")
    
    if debt_ratio < 0.3:
        insights.append("✅ Low debt levels - financial flexibility maintained")
    elif debt_ratio < 0.5:
        insights.append("📊 Manageable debt levels - monitor leverage")
    else:
        insights.append("⚠️ High debt levels - consider deleveraging")
    
    for insight in insights:
        st.write(insight)

with tabs[3]:
    preview = results.get('data_preview', [])
    if preview:
        st.subheader("📋 Raw Data Preview")
        df_preview = pd.DataFrame(preview)
        
        # Add data summary
        st.markdown("### 📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_preview))
        with col2:
            st.metric("Columns", len(df_preview.columns))
        with col3:
            st.metric("Data Types", len(df_preview.dtypes.unique()))
        
        # Display the dataframe with enhanced styling
        st.dataframe(
            df_preview, 
            use_container_width=True, 
            height=400,
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    help=f"Values in {col}",
                    format="$%d" if "revenue" in col.lower() or "amount" in col.lower() else None
                ) for col in df_preview.columns if df_preview[col].dtype in ['int64', 'float64']
            }
        )
    else:
        st.info("No data preview available.")

# Footer with additional information
st.markdown("---")
st.markdown("### 🚀 Quick Actions")
action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("🔮 Forecasting", use_container_width=True):
        st.switch_page("pages/Forecasting.py")

with action_col2:
    if st.button("🎯 Scenario Simulator", use_container_width=True):
        st.switch_page("pages/Scenario_Simulator.py")

with action_col3:
    if st.button("🤖 AI Advisory", use_container_width=True):
        st.switch_page("pages/Advisory_Chat.py")

with action_col4:
    if st.button("📊 Export Report", use_container_width=True):
        st.success("Report export feature coming soon!")