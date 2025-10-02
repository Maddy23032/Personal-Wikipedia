"""History page for Streamlit app.
Shows question/answer history with copy/download options and a clear button.
"""
from __future__ import annotations
import io
import json
import re
import streamlit as st
from streamlit.components.v1 import html


def _download_bytes(filename: str, content: str) -> bytes:
    buf = io.BytesIO()
    buf.write(content.encode("utf-8"))
    buf.seek(0)
    return buf.getvalue()


def main():
    st.set_page_config(page_title="History - Personal Wikipedia", page_icon="🕘", layout="centered")
    st.title("🕘 Prompt History")

    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet. Ask a question on the main page.")
        return

    for i, item in enumerate(hist, start=1):
        with st.expander(f"Q{i}: {item['question']}"):
            st.markdown("**Answer:**")
            # Render mixed text and code blocks
            ans = item["answer"]
            pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)```", re.DOTALL)
            pos = 0
            for m in pattern.finditer(ans):
                start, end = m.span()
                if start > pos:
                    pre = ans[pos:start]
                    if pre.strip():
                        st.write(pre.strip())
                lang = (m.group(1) or '').strip() or None
                code = m.group(2).rstrip()
                st.code(code, language=lang)
                pos = end
            if pos < len(ans):
                tail = ans[pos:]
                if tail.strip():
                    st.write(tail.strip())
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download (.txt)",
                    data=_download_bytes(f"answer_{i}.txt", item["answer"]),
                    file_name=f"answer_{i}.txt",
                    mime="text/plain",
                )
            st.caption("Use the sidebar to navigate back to the main page.")

    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared.")


if __name__ == "__main__":
    main()
