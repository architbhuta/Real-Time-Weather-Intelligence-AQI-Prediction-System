import streamlit as st

from utils.config import DEFAULT_LOCATION, LOCATIONS


def render_location_selector() -> str:
    if "location" not in st.session_state:
        st.session_state["location"] = DEFAULT_LOCATION

    options = list(LOCATIONS.keys())
    current = st.session_state["location"]
    index = options.index(current) if current in options else 0

    location = st.sidebar.selectbox("Location", options=options, index=index, key="location_selector")
    st.session_state["location"] = location
    return location
