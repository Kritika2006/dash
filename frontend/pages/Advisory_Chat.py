# cfo_dashboard/frontend/pages/Advisory_Chat.py
import streamlit as st
import sys
import os

# --- PATH HACK and IMPORTS ---
# This ensures we can import from the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from frontend.utils import api_client
from frontend.utils.ui_components import render_sidebar
# ------------------------------------

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advisory Chat", layout="centered")
render_sidebar()

# --- PAGE CONTENT ---
st.title("💬 Chat with Your AI Financial Advisor")

# Check if data exists in session state
if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
    st.warning("📂 No financial data loaded.")
    st.info("""
    **To get the best AI advice:**
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
    
    st.info("💡 **You can still chat with the AI**, but it won't have context about your specific financial data.")
    st.markdown("---")

# --- Initialize chat history in session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show data status
if st.session_state.get("analysis_results"):
    st.success(f"📊 **Using data from:** {st.session_state.get('uploaded_file_name', 'Unknown file')}")
else:
    st.warning("⚠️ **No financial data loaded** - AI responses will be general advice only.")

st.markdown("---")

# --- Display prior chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main chat input and response logic ---
if prompt := st.chat_input("Ask a question about your financial data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if financial data is available to provide context
            financial_context = {}
            if "analysis_results" in st.session_state and st.session_state.analysis_results:
                 kpis = st.session_state.analysis_results.get("kpis", {})
                 # Reconstruct the payload the backend expects
                 financial_context = {
                     "revenue": kpis.get('profit', 0) + kpis.get('burn_rate', 0),
                     "expenses": kpis.get('burn_rate', 0),
                     "liabilities": 0, # Placeholder, as this is not in the Zomato table
                     "burn_rate": kpis.get('burn_rate', 0),
                     "cash": kpis.get('runway_months', 0) * kpis.get('burn_rate', 0) # Placeholder
                 }
                 st.info("Using financial data from your uploaded report as context.", icon="ℹ️")
            else:
                st.warning("No financial file uploaded. The AI will have no context for its answers.", icon="⚠️")
                # Provide a default empty context if no file is uploaded
                financial_context = {"revenue": 0, "expenses": 0, "liabilities": 0, "burn_rate": 0, "cash": 0}

            # Get the AI's response from the backend
            response_data = api_client.ask_ai_assistant(prompt, financial_context)
            response = response_data.get("answer", "Sorry, I encountered an error connecting to the AI model.")
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})