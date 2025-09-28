import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="NexusQuery", 
    page_icon="🧠", 
    layout="wide"
)

# --- Backend API URL ---
API_URL = "http://127.0.0.1:8000"

# --- Main Application ---
st.title("🧠 NexusQuery v2.0")
st.markdown("""
Welcome to NexusQuery! Build a smart knowledge base from your documents.
1.  Use the sidebar to create a new knowledge base and upload your files.
2.  Wait for the status to show 'Ready'.
3.  Ask questions in the main chat area.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    st.subheader("1. Create Knowledge Base")
    if st.button("Create New Knowledge Base"):
        with st.spinner("Creating..."):
            response = requests.post(f"{API_URL}/knowledge-bases")
            if response.status_code == 201:
                kb_id = response.json().get("knowledge_base_id")
                st.session_state.kb_id = kb_id
                st.success(f"Knowledge Base created with ID: {kb_id}")
            else:
                st.error("Failed to create knowledge base.")

    if "kb_id" in st.session_state:
        st.info(f"Current KB ID: {st.session_state.kb_id}")
        
        st.subheader("2. Add Sources")
        url_input = st.text_input("Add a source from a URL")
        uploaded_files = st.file_uploader(
            "Or upload documents",
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )

        if st.button("Process Sources"):
            if uploaded_files or url_input:
                with st.spinner("Uploading and starting processing..."):
                    kb_id = st.session_state.kb_id
                    endpoint = f"{API_URL}/knowledge-bases/{kb_id}/upload"
                    
                    # Prepare files for multipart upload
                    files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                    # Prepare URL data
                    url_data = {"urls": [url_input]} if url_input else {}

                    response = requests.post(endpoint, files=files_to_upload, data=url_data)

                    if response.status_code == 202:
                        st.success(response.json().get("message"))
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            else:
                st.warning("Please add a URL or upload at least one file.")

        st.subheader("3. Check Status")
        if st.button("Check Processing Status"):
            with st.spinner("Checking status..."):
                kb_id = st.session_state.kb_id
                response = requests.get(f"{API_URL}/knowledge-bases/{kb_id}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    st.info(f"Status: {status_data.get('status')}, Files: {status_data.get('file_count')}")
                else:
                    st.error("Could not retrieve status.")

# --- Main Chat Area ---
st.header("Chat with your Knowledge Base")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load chat history from the API when the app loads or kb_id changes
if "kb_id" in st.session_state and st.session_state.kb_id:
    # flag to load history only once per KB
    if "history_loaded_for" not in st.session_state or st.session_state.history_loaded_for != st.session_state.kb_id:
        history_endpoint = f"{API_URL}/knowledge-bases/{st.session_state.kb_id}/history"
        response = requests.get(history_endpoint)
        if response.status_code == 200:
            st.session_state.messages = response.json()
            st.session_state.history_loaded_for = st.session_state.kb_id
        else:
            st.session_state.messages = []


# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask questions about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.kb_id:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_endpoint = f"{API_URL}/knowledge-bases/{st.session_state.kb_id}/query"
                
                # Get the last 4 messages to send as history
                # We exclude the current prompt itself from the history
                history = st.session_state.messages[:-1][-4:]
                
                # Send the query AND the history to the backend
                response = requests.post(
                    query_endpoint, 
                    json={"query": prompt, "history": history}
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get("answer")
                    # ... (rest of the response handling)
                    st.markdown(answer)
                    # Add AI response to the history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    # ... (error handling)
                    pass