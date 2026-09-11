import streamlit as st

# Streamlit dialogs are fragments. The app uses a startup language dialog,
# and fragment state can become stale across mobile reconnects/redeploys.
# Render the same chooser inline instead, without changing inference logic.
def _inline_dialog(_title):
    def decorator(func):
        return func
    return decorator

st.dialog = _inline_dialog

import frontend.app  # noqa: E402,F401
