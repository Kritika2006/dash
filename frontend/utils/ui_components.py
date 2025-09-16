# cfo_dashboard/frontend/utils/ui_components.py
import streamlit as st
from . import api_client

def render_sidebar():
    """
    Renders the sidebar with the logo, file uploader, and status messages.
    """
    # --- Custom CSS for Sidebar Reordering and Styling ---
    st.markdown("""
    <style>
        [data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; }
        [data-testid="stSidebar"] nav { order: 2; }
        .sidebar-logo { order: 1; padding-bottom: 20px; }
        .sidebar-logo h2 { text-align: center; font-size: 3em; font-weight: bold; }
        [data-testid="stFileUploader"] { order: 3; }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar Content ---
    st.sidebar.markdown("<div class='sidebar-logo'><h2>C<sup>3</sup></h2></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Temporarily disable file uploader
    # uploaded_file = st.sidebar.file_uploader(
    #     "📂 Upload Financial Report",
    #     type=["csv", "xlsx", "xls", "pdf"]
    # )

    # if uploaded_file:
    #     if st.session_state.get("uploaded_file_name") != uploaded_file.name:
    #         st.session_state.uploaded_file_name = uploaded_file.name
    #         st.session_state['uploaded_file_object'] = uploaded_file
    #         with st.spinner("AI is analyzing your document..."):
    #             st.session_state.analysis_results = api_client.get_dashboard_data(uploaded_file)
    #         st.switch_page("pages/Dashboard.py")

    # --- Sidebar Status Message ---
    if st.session_state.get("analysis_results"):
        st.sidebar.success(f"Successfully analyzed:\n**{st.session_state.uploaded_file_name}**")
    else:
        st.sidebar.info("Awaiting file upload...")