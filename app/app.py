import streamlit as st

# Define pages using relative file paths
home = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
image = st.Page("pages/image_section.py", title="Image Captioning", icon="🖼️")
search_history = st.Page("pages/search_history.py", title="Search History", icon="🔍")
agent = st.Page("pages/agent.py", title="AI Agent", icon="🤖")

# Render navigation widget in the sidebar
pg = st.navigation([home, image, search_history, agent])
pg.run()