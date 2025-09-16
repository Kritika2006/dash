# cfo_dashboard/frontend/pages/Forecasting.py
import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# --- PATH HACK and IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from frontend.utils.ui_components import render_sidebar
from frontend.utils import api_client

st.set_page_config(page_title="Forecasting", layout="wide")
render_sidebar()

st.title("🔮 Financial Forecasting")
st.markdown("Predict future trends for key metrics like Revenue and Expenses.")

# Check if data exists in session state
if "analysis_results" not in st.session_state or not st.session_state.analysis_results:
    st.warning("📂 No financial data loaded.")
    st.info("""
    **To get started with forecasting:**
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

col1, col2 = st.columns([1,3])
with col1:
    model = st.selectbox(
        "Model",
        options=["arima", "random_forest", "gradient_boosting"],
        index=0,
        help="ARIMA for classical time series; RF/GB for ML-based forecasting",
    )

# --- FORECAST GENERATION ---
cleaned_data_records = st.session_state.analysis_results.get("data_preview")

if cleaned_data_records:
    with st.spinner("Generating forecast... This might take a minute."):
        forecast_data = api_client.get_forecast_data(cleaned_data_records, model=model)

    if forecast_data:
        historical = pd.DataFrame(forecast_data['historical'])
        forecast = pd.DataFrame(forecast_data['forecast'])
        
        # Convert date columns to datetime objects
        historical['date'] = pd.to_datetime(historical['date'])
        forecast['date'] = pd.to_datetime(forecast['date'])

        # --- CHART VISUALIZATION ---
        st.subheader("Metric Forecast (Next 12 Months)")

        base = alt.Chart(historical).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('actual:Q', title='Value'),
            tooltip=['date:T', 'actual:Q']
        ).properties(
            title='Historical Data vs. Forecast'
        )

        forecast_line = alt.Chart(forecast).mark_line(strokeDash=[5,5], color='green', point=True).encode(
            x='date:T',
            y=alt.Y('forecast:Q', title='Value'),
            tooltip=['date:T', 'forecast:Q']
        )

        confidence_area = alt.Chart(forecast).mark_area(opacity=0.3, color='green').encode(
            x='date:T',
            y='lower_bound:Q',
            y2='upper_bound:Q'
        )

        final_chart = (base + confidence_area + forecast_line).interactive()
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")
    else:
        st.error("Could not retrieve forecast data from the backend.")
else:
    st.error("No clean data available to generate a forecast.")