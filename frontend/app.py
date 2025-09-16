# cfo_dashboard/frontend/app.py
import streamlit as st
import os
import sys
from utils.ui_components import render_sidebar
from utils.api_client import load_sample_data  # Ensure this import is correct

# --- Page Config ---
st.set_page_config(page_title="AI CFO Dashboard", layout="wide")

# --- Session State Initialization ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = load_sample_data()  # Use sample data directly
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = "Sample Data"

# --- Render the common sidebar ---
render_sidebar()

# --- Main Page Content ---
st.title("Welcome to the AI CFO Dashboard 📈")
st.markdown("""
### Your AI-Powered Co-Pilot for Financial Strategy

**To get started:**
1.  **Analyze the sample financial data** preloaded into the dashboard.
2.  **Navigate** to the `Dashboard` or `Advisory Chat` pages to see the results.
""")

# Show current status
if st.session_state.get("analysis_results"):
    st.success(f"📊 **Data Ready:** {st.session_state.get('uploaded_file_name', 'Unknown file')}")
    st.info("You can now navigate to Dashboard, Forecasting, Scenario Simulator, or Advisory Chat to explore the data.")
else:
    st.warning("No data loaded. Please upload a file using the sidebar.")