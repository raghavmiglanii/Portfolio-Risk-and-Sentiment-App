import streamlit as st
import pandas as pd


def read_log_from_memory():
    """Read suggestions log from session state"""
    return st.session_state.suggestions_log.copy()


def write_log_to_memory(new_suggestions_df):
    """Write suggestions log to session state"""
    try:
        current_log = read_log_from_memory()
        combined = pd.concat([current_log, new_suggestions_df])
        st.session_state.suggestions_log = combined.drop_duplicates(
            subset=['Ticker', 'Suggestion Date'], keep='last'
        ).reset_index(drop=True)
        return True
    except Exception as e:
        st.error(f"Failed to save to memory: {e}")
        return False