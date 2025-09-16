# cfo_dashboard/frontend/pages/Scenario_Simulator.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# --- PATH HACK and IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from frontend.utils.ui_components import render_sidebar
from frontend.utils import api_client

st.set_page_config(page_title="Scenario Simulator", layout="wide")
render_sidebar()

st.title("🎯 Advanced Scenario Simulator")
st.markdown("Test different financial scenarios and see their impact on your business with advanced analytics.")

# Check if data exists in session state
if "analysis_results" not in st.session_state or not st.session_state.analysis_results:
    st.warning("📂 No financial data loaded.")
    st.info("""
    **To get started with scenario simulation:**
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

# Get current financial data
results = st.session_state.analysis_results
kpis = results.get('kpis', {})

# Current metrics display
st.subheader("📊 Current Financial Position")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Revenue", f"${kpis.get('revenue', 0):,.0f}")
with col2:
    st.metric("Expenses", f"${kpis.get('expenses', 0):,.0f}")
with col3:
    st.metric("Profit", f"${kpis.get('profit', 0):,.0f}")
with col4:
    st.metric("Runway", f"{kpis.get('runway_months', 0):.1f} months")
with col5:
    st.metric("Debt Ratio", f"{kpis.get('debt_ratio', 0):.2f}")

# Scenario configuration
st.subheader("🎛️ Scenario Configuration")
scenario_tabs = st.tabs(["📈 Revenue Scenarios", "💰 Expense Scenarios", "🔄 Combined Scenarios", "🎲 Monte Carlo", "📊 ESG Analysis"])

with scenario_tabs[0]:
    st.markdown("### Revenue Scenario Testing")
    
    col1, col2 = st.columns(2)
    with col1:
        optimistic_revenue = st.slider("🚀 Optimistic Growth (%)", -50, 100, 25, help="Best case revenue growth")
        moderate_revenue = st.slider("📊 Moderate Growth (%)", -30, 50, 8, help="Realistic revenue growth")
    with col2:
        pessimistic_revenue = st.slider("📉 Pessimistic Decline (%)", -80, 20, -12, help="Worst case revenue decline")
        recession_revenue = st.slider("💥 Recession Impact (%)", -90, 10, -25, help="Economic downturn impact")
    
    revenue_scenarios = [
        {"name": "Optimistic Growth", "revenue_change": optimistic_revenue, "expense_change": 0},
        {"name": "Moderate Growth", "revenue_change": moderate_revenue, "expense_change": 0},
        {"name": "Pessimistic Decline", "revenue_change": pessimistic_revenue, "expense_change": 0},
        {"name": "Recession Impact", "revenue_change": recession_revenue, "expense_change": 0}
    ]

with scenario_tabs[1]:
    st.markdown("### Expense Scenario Testing")
    
    col1, col2 = st.columns(2)
    with col1:
        cost_cutting = st.slider("✂️ Cost Cutting (%)", -50, 0, -15, help="Aggressive cost reduction")
        efficiency_gains = st.slider("⚡ Efficiency Gains (%)", -30, 0, -8, help="Operational efficiency improvements")
    with col2:
        inflation_pressure = st.slider("📈 Inflation Pressure (%)", 0, 50, 12, help="Rising costs due to inflation")
        expansion_costs = st.slider("🏗️ Expansion Costs (%)", 0, 100, 30, help="Costs for business expansion")
    
    expense_scenarios = [
        {"name": "Cost Cutting", "revenue_change": 0, "expense_change": cost_cutting},
        {"name": "Efficiency Gains", "revenue_change": 0, "expense_change": efficiency_gains},
        {"name": "Inflation Pressure", "revenue_change": 0, "expense_change": inflation_pressure},
        {"name": "Expansion Investment", "revenue_change": 0, "expense_change": expansion_costs}
    ]

with scenario_tabs[2]:
    st.markdown("### Combined Scenario Testing")
    
    col1, col2 = st.columns(2)
    with col1:
        growth_scenario_rev = st.slider("🚀 Growth Revenue (%)", -20, 100, 35, help="Revenue growth in expansion scenario")
        growth_scenario_exp = st.slider("💰 Growth Expenses (%)", -10, 50, 20, help="Expense increase for growth")
    with col2:
        crisis_scenario_rev = st.slider("💥 Crisis Revenue (%)", -80, 20, -35, help="Revenue decline in crisis")
        crisis_scenario_exp = st.slider("✂️ Crisis Expenses (%)", -30, 20, -20, help="Expense reduction in crisis")
    
    combined_scenarios = [
        {"name": "Growth Scenario", "revenue_change": growth_scenario_rev, "expense_change": growth_scenario_exp},
        {"name": "Crisis Scenario", "revenue_change": crisis_scenario_rev, "expense_change": crisis_scenario_exp},
        {"name": "Stable Scenario", "revenue_change": 0, "expense_change": 0}
    ]

with scenario_tabs[3]:
    st.markdown("### Monte Carlo Simulation")
    st.info("Monte Carlo simulation uses random sampling to model thousands of possible outcomes based on your financial parameters.")
    
    col1, col2 = st.columns(2)
    with col1:
        initial_cash = st.number_input("💰 Initial Cash", value=float(kpis.get('cash', 1000000)), min_value=0.0, format="%.0f")
        monthly_revenue_mean = st.number_input("📈 Monthly Revenue Mean", value=float(kpis.get('revenue', 1000000)) / 12, min_value=0.0, format="%.0f")
        monthly_revenue_std = st.number_input("📊 Revenue Volatility", value=float(kpis.get('revenue', 1000000)) / 12 * 0.2, min_value=0.0, format="%.0f")
    with col2:
        monthly_expenses_mean = st.number_input("💸 Monthly Expenses Mean", value=float(kpis.get('expenses', 800000)) / 12, min_value=0.0, format="%.0f")
        monthly_expenses_std = st.number_input("📊 Expense Volatility", value=float(kpis.get('expenses', 800000)) / 12 * 0.1, min_value=0.0, format="%.0f")
        simulations = st.number_input("🎲 Number of Simulations", value=10000, min_value=1000, max_value=50000, step=1000)

with scenario_tabs[4]:
    st.markdown("### ESG Risk Analysis")
    st.info("Environmental, Social, and Governance (ESG) factors can significantly impact financial performance and risk.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        environmental_score = st.slider("🌱 Environmental Score", 0.0, 1.0, 0.6, 0.1, help="Environmental sustainability score (0-1)")
    with col2:
        social_score = st.slider("👥 Social Score", 0.0, 1.0, 0.7, 0.1, help="Social responsibility score (0-1)")
    with col3:
        governance_score = st.slider("⚖️ Governance Score", 0.0, 1.0, 0.8, 0.1, help="Corporate governance score (0-1)")

# Run scenarios
if st.button("🚀 Run Advanced Scenario Analysis", type="primary"):
    with st.spinner("Running comprehensive scenario analysis..."):
        # Combine all scenarios
        all_scenarios = revenue_scenarios + expense_scenarios + combined_scenarios
        
        # Run stress test
        try:
            stress_test_results = api_client.run_stress_test(
                base_revenue=kpis.get('revenue', 1000000),
                base_expenses=kpis.get('expenses', 800000),
                base_cash=kpis.get('cash', 1000000),
                scenarios=all_scenarios
            )
        except Exception as e:
            st.error(f"Stress test failed: {e}")
            stress_test_results = {}
        
        # Run Monte Carlo simulation
        try:
            monte_carlo_results = api_client.run_monte_carlo(
                initial_cash=initial_cash,
                monthly_revenue_mean=monthly_revenue_mean,
                monthly_revenue_std=monthly_revenue_std,
                monthly_expenses_mean=monthly_expenses_mean,
                monthly_expenses_std=monthly_expenses_std,
                simulations=simulations
            )
        except Exception as e:
            st.error(f"Monte Carlo simulation failed: {e}")
            monte_carlo_results = {}
        
        # Run ESG analysis
        try:
            esg_results = api_client.calculate_esg_score(
                environmental_score=environmental_score,
                social_score=social_score,
                governance_score=governance_score
            )
        except Exception as e:
            st.error(f"ESG analysis failed: {e}")
            esg_results = {}
        
        # Display results
        st.subheader("📊 Scenario Analysis Results")
        
        # Stress test results
        if stress_test_results:
            st.markdown("### 🎯 Stress Test Results")
            stress_df = pd.DataFrame([
                {
                    "Scenario": name,
                    "Revenue": f"${data['adjusted_revenue']:,.0f}",
                    "Expenses": f"${data['adjusted_expenses']:,.0f}",
                    "Profit/Loss": f"${data['profit_loss']:,.0f}",
                    "Runway (months)": f"{data['runway_months']:.1f}",
                    "Revenue Change": f"{data['revenue_change_pct']:+.1f}%",
                    "Expense Change": f"{data['expense_change_pct']:+.1f}%"
                }
                for name, data in stress_test_results.items()
            ])
            
            st.dataframe(stress_df, use_container_width=True)
            
            # Visualize stress test results
            fig = go.Figure()
            
            scenarios = list(stress_test_results.keys())
            profits = [stress_test_results[s]['profit_loss'] for s in scenarios]
            runways = [stress_test_results[s]['runway_months'] for s in scenarios]
            
            fig.add_trace(go.Bar(
                name='Profit/Loss',
                x=scenarios,
                y=profits,
                yaxis='y',
                offsetgroup=1
            ))
            
            fig.add_trace(go.Bar(
                name='Runway (months)',
                x=scenarios,
                y=runways,
                yaxis='y2',
                offsetgroup=2
            ))
            
            fig.update_layout(
                title='Scenario Impact Analysis',
                xaxis_title='Scenarios',
                yaxis=dict(title='Profit/Loss ($)', side='left'),
                yaxis2=dict(title='Runway (months)', side='right', overlaying='y'),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Monte Carlo results
        if monte_carlo_results:
            st.markdown("### 🎲 Monte Carlo Simulation Results")
col1, col2, col3, col4 = st.columns(4)
            
with col1:
                st.metric("Expected Final Cash", f"${monte_carlo_results['expected_final_cash']:,.0f}")
with col2:
                st.metric("95% VaR", f"${monte_carlo_results['var_95']:,.0f}")
with col3:
                st.metric("99% VaR", f"${monte_carlo_results['var_99']:,.0f}")
with col4:
                st.metric("Bankruptcy Probability", f"{monte_carlo_results['bankruptcy_probability']:.2%}")
            
            # Confidence interval
            ci = monte_carlo_results['confidence_interval_95']
            st.info(f"95% Confidence Interval: ${ci[0]:,.0f} - ${ci[1]:,.0f}")
            
            # Visualization
            st.markdown("### 📈 Cash Flow Distribution")
            final_cash_dist = monte_carlo_results['final_cash_distribution']
            
            fig = px.histogram(
                x=final_cash_dist,
                nbins=50,
                title='Distribution of Final Cash Positions',
                labels={'x': 'Final Cash ($)', 'y': 'Frequency'}
            )
            fig.add_vline(x=monte_carlo_results['expected_final_cash'], line_dash="dash", line_color="red", 
                         annotation_text="Expected Value")
            fig.add_vline(x=monte_carlo_results['var_95'], line_dash="dash", line_color="orange", 
                         annotation_text="95% VaR")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ESG results
        if esg_results:
            st.markdown("### 🌱 ESG Risk Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall ESG Score", f"{esg_results['overall_esg_score']:.3f}")
                st.metric("Risk Level", esg_results['risk_level'].title())
            
            with col2:
                # ESG component scores
                fig = go.Figure(data=go.Scatterpolar(
                    r=[esg_results['environmental_score'], esg_results['social_score'], esg_results['governance_score']],
                    theta=['Environmental', 'Social', 'Governance'],
                    fill='toself',
                    name='ESG Scores'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="ESG Component Scores"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ESG recommendations
            st.markdown("### 💡 ESG Recommendations")
            for i, rec in enumerate(esg_results['recommendations'], 1):
                st.write(f"{i}. {rec}")

# Additional insights
st.subheader("💡 Key Insights")
st.info("""
**Scenario Analysis Best Practices:**
- Always test both optimistic and pessimistic scenarios
- Consider multiple time horizons (3, 6, 12 months)
- Factor in external economic conditions
- Monitor key risk indicators regularly
- Use Monte Carlo simulations for probabilistic outcomes
- Integrate ESG factors into risk assessment
""")