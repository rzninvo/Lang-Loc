import streamlit as st
import pandas as pd

def render_admin_tables(annotations, uninterpretable_images):
    col1, col2 = st.columns(2)
    with col1:
        if annotations:
            with st.expander("📊 Recent Annotations (Admin View)"):
                st.dataframe(pd.DataFrame(annotations).tail(10))
    with col2:
        if uninterpretable_images:
            with st.expander("⚠️ Uninterpretable Images (Admin View)"):
                st.dataframe(pd.DataFrame(uninterpretable_images).tail(10))