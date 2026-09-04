import sys
from pathlib import Path

# Same fix as translation.py needed earlier — Streamlit's page runner
# doesn't reliably add rag/ to the import path on its own.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from rag.rag_manager import search_history

st.set_page_config(page_title="Search History", page_icon="🔍")
st.title("🔍 Search Your Caption History")

query = st.text_input("Search past captions by meaning (e.g. 'photos with dogs')")

if query:
    with st.spinner("Searching..."):
        results = search_history(query, top_k=5)
    count = 0
    if not results:
        st.info("No caption history yet — generate some captions first.")
    else:
        for r in results:
            count += 1
            st.markdown(f"Result {count} -> **{r['caption']}**")
            st.write(r['metadata']['translated_caption'])
            st.caption(
                f"{r['metadata']['language']} · "
                f"{r['metadata']['media_type']} · "
                f"{r['metadata']['timestamp'][:10]} · "
                f"similarity distance: {r['distance']:.3f}"
            )
            st.markdown("---")