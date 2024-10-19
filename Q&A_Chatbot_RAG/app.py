import streamlit as st
from chain import loading, creation_of_vectorDB_in_local, creation_FAQ_chain

# Set page configuration
st.set_page_config(page_title="QA Bot", layout="wide", page_icon="🤖")

# App header with a title and description
st.title("🤖 QA Bot - Your AI Assistant")
st.markdown("Ask questions based on your uploaded documents and get instant answers!")

# Initialize session state for chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = []

# Sidebar for document uploads
with st.sidebar:
    st.header("Settings")
    st.subheader("Upload your Documents")
    docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if st.button("Process Documents"):
        with st.spinner("Processing your documents..."):
            loaders = loading(docs)
            creation_of_vectorDB_in_local(loaders)
            st.success("Documents have been successfully processed!")

# Main chat interface
st.write("### Chat Interface")

# Display chat history with GPT-like message bubbles
for chat in st.session_state.chatHistory:
    if chat['role'] == 'user':
        st.chat_message("user", avatar="👤").markdown(chat['content'])
    else:
        st.chat_message("assistant", avatar="🤖").markdown(chat['content'])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user input to chat history
    st.session_state.chatHistory.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").markdown(prompt)

    # Generate response using the chain model
    if st.session_state.conversation:
        con = st.session_state.conversation
    else:
        con = None

    with st.spinner("Generating response..."):
        ch = creation_FAQ_chain()
        res = ch({
            "context": "You are a Q/A chatbot",
            "question": 'You should respond only with the information in the question. Please respond in a crisp way, within 60 words.',
            "query": prompt
        })
        cleaned_response = res['answer']

    # Add assistant response to chat history
    st.session_state.chatHistory.append({"role": "assistant", "content": cleaned_response})
    st.chat_message("assistant", avatar="🤖").markdown(cleaned_response)

    # Save conversation context
    st.session_state.conversation = con
