import streamlit as st
import requests
import time
import re
import os
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="NexusQuery Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Backend API URL ---
API_URL = "http://127.0.0.1:8000"

# --- Session State Initialization ---
# This is key for switching between "pages"
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "kb_id" not in st.session_state:
    st.session_state.kb_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_loaded_for" not in st.session_state:
    st.session_state.history_loaded_for = None

# --- Page 1: Dashboard ---
def render_dashboard():
    st.title("🧠 NexusQuery Dashboard")
    st.markdown("Manage your AI Knowledge Bases.")

    # --- Section 1: Create New Knowledge Base ---
    st.header("1. Create New Knowledge Base")
    with st.form("create_kb_form"):
        url_input = st.text_input("Add a source from a URL (optional)")
        uploaded_files = st.file_uploader(
            "Upload documents (required)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        submit_button = st.form_submit_button("Create and Process")

        if submit_button:
            if not uploaded_files and not url_input:
                st.error("Please add at least one URL or upload one file.")
            else:
                with st.spinner("Creating Knowledge Base..."):
                    # 1. Create the KB
                    response = requests.post(f"{API_URL}/knowledge-bases")
                    if response.status_code == 201:
                        kb_id = response.json().get("knowledge_base_id")
                        st.success(f"Knowledge Base created with ID: {kb_id}")
                        
                        # 2. Upload files and URLs
                        with st.spinner(f"Uploading and processing sources for {kb_id}..."):
                            endpoint = f"{API_URL}/knowledge-bases/{kb_id}/upload"
                            files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                            url_data = {"urls": [url_input]} if url_input else {}

                            response = requests.post(endpoint, files=files_to_upload, data=url_data)
                            if response.status_code == 202:
                                st.success("Processing started! Refresh the list below to see status.")
                                st.rerun()
                            else:
                                st.error(f"Error uploading: {response.text}")
                    else:
                        st.error("Failed to create knowledge base.")

    st.divider()

    # --- Section 2: Manage Existing KBs ---
    st.header("2. Manage Existing Knowledge Bases")
    
    # Button to refresh the list
    if st.button("Refresh List"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/knowledge-bases")
        if response.status_code == 200:
            kbs = response.json()
            if not kbs:
                st.info("No Knowledge Bases found. Create one above!")
                return

            # Display each KB in a card-like format
            for kb in sorted(kbs, key=lambda x: x['id'], reverse=True):
                kb_id = kb['id']
                with st.container(border=True):
                    st.subheader(f"ID: {kb_id}")
                    st.text(f"Status: {kb['status']}")
                    st.text(f"File Count: {len(kb['files'])}")

                    col1, col2 = st.columns(2)
                    with col1:
                        # LOAD BUTTON
                        if st.button("Load Chat", key=f"load_{kb_id}", type="primary", disabled=(kb['status'] != 'ready')):
                            st.session_state.kb_id = kb_id
                            st.session_state.page = "chat" # Switch page
                            st.session_state.messages = [] # Clear old chat
                            st.session_state.history_loaded_for = None
                            st.rerun()
                    with col2:
                        # DELETE BUTTON
                        if st.button("Delete", key=f"del_{kb_id}"):
                            with st.spinner("Deleting..."):
                                del_response = requests.delete(f"{API_URL}/knowledge-bases/{kb_id}")
                                if del_response.status_code == 200:
                                    st.success(f"KB {kb_id} deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {del_response.text}")
        else:
            st.error(f"Failed to fetch Knowledge Bases: {response.text}")
    except requests.ConnectionError:
        st.error("Could not connect to backend. Is the FastAPI server running?")


# --- Page 2: Chat Interface ---
def render_chat_page():
    kb_id = st.session_state.kb_id
    st.title(f"🧠 Chat with KB: `{kb_id}`")

    # --- Back Button ---
    if st.button("<- Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.session_state.kb_id = None
        st.session_state.messages = []
        st.session_state.history_loaded_for = None
        st.rerun()
    
    st.divider()

    # --- Create Tabs ---
    tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Explore Graph"])

    # --- Tab 1: Chat ---
    with tab1:
        # --- Load Chat History ---
        if st.session_state.history_loaded_for != kb_id:
            try:
                history_endpoint = f"{API_URL}/knowledge-bases/{kb_id}/history"
                response = requests.get(history_endpoint)
                if response.status_code == 200:
                    st.session_state.messages = response.json()
                    st.session_state.history_loaded_for = kb_id
                else:
                    st.session_state.messages = []
                    st.warning("Could not load chat history.")
            except requests.ConnectionError:
                st.error("Connection to backend failed.")
                st.session_state.messages = []

        # --- Display existing messages ---
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --- Handle new user input ---
        if prompt := st.chat_input("Ask questions about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    query_endpoint = f"{API_URL}/knowledge-bases/{kb_id}/query"
                    history = [
                        {"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]
                    ][-4:]
                    
                    try:
                        response = requests.post(
                            query_endpoint, 
                            json={"query": prompt, "history": history}
                        )
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            raw_answer = response_data.get("answer")
                            sources = response_data.get("sources", {})

                            answer_part = ""
                            citations_part = ""
                            
                            if "CITATIONS" in raw_answer:
                                parts = raw_answer.split("CITATIONS", 1)
                                answer_part = parts[0].strip()
                                citations_part = parts[1].strip()
                            else:
                                answer_part = raw_answer.strip()

                            st.markdown(answer_part)
                            
                            if citations_part and sources:
                                cited_ids = re.findall(r'\[(SOURCE_\d+)\]', citations_part)
                                if cited_ids:
                                    with st.expander("Show Sources"):
                                        for source_id in set(cited_ids):
                                            if source_id in sources:
                                                st.subheader(f"Source: {source_id}")
                                                st.markdown(f"> {sources[source_id]}")
                            
                            st.session_state.messages.append({"role": "assistant", "content": raw_answer}) # Save raw answer
                        
                        else:
                            st.error(f"Error from API: {response.status_code} - {response.text}")
                    
                    except requests.ConnectionError:
                        st.error("Failed to connect to backend.")

    # --- Tab 2: Knowledge Graph Explorer ---
    with tab2:
        st.subheader("Interactive Knowledge Graph")

        # Load graph data from session state if it exists
        if f"graph_data_{kb_id}" not in st.session_state:
            st.session_state[f"graph_data_{kb_id}"] = None

        if st.button("Load/Refresh Graph"):
            with st.spinner("Fetching graph data..."):
                try:
                    graph_endpoint = f"{API_URL}/knowledge-bases/{kb_id}/graph"
                    response = requests.get(graph_endpoint)
                    if response.status_code == 200:
                        st.session_state[f"graph_data_{kb_id}"] = response.json()
                    else:
                        st.error(f"Failed to fetch graph: {response.text}")
                except requests.ConnectionError:
                    st.error("Failed to connect to backend.")

        # If data is loaded, display it
        graph_data = st.session_state[f"graph_data_{kb_id}"]
        if graph_data:
            if not graph_data["nodes"]:
                st.warning("No graph data found for this Knowledge Base.")
            else:
                # --- NEW PYVIS LOGIC ---
                net = Network(height="800px", width="1100px", heading="")

                # Add nodes
                for node in graph_data["nodes"]:
                    net.add_node(node["id"], label=node["label"], size=node["size"])

                # Add edges
                for edge in graph_data["edges"]:
                    net.add_edge(edge["source"], edge["target"], label=edge["label"])

                # Generate the graph as an HTML file
                try:
                    # Save the graph to a temporary HTML file
                    html_file = f"graph_{kb_id}.html"
                    net.save_graph(html_file)

                    # Read the HTML file
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_data = f.read()

                    # Display the HTML file in Streamlit
                    components.html(html_data, height=800, width=1100, scrolling=True)

                    # Clean up the file
                    if os.path.exists(html_file):
                        os.remove(html_file)

                except Exception as e:
                    st.error(f"Error generating graph: {e}")
                # --- END OF PYVIS LOGIC ---

# --- Main Router ---
if st.session_state.page == "dashboard":
    render_dashboard()
else:
    render_chat_page()