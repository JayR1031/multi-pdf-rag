import streamlit as st
import html
from rag_backend import (
    build_vectordb_from_uploaded_files,
    get_llm,
    answer_question,
    stream_llm_answer,
)

st.set_page_config(page_title="Chat with Your PDFs",
                   page_icon="📄", layout="wide")

# CSS
st.markdown("""
<style>
.chat-message {
    padding: 10px 15px;
    border-radius: 10px;
    margin-bottom: 12px;
    word-wrap: break-word;
    line-height: 1.5;
}
.user-message {
    background-color: #DCF8C6;
    margin-left: auto;
    color: #000000 !important;
    border: 1px solid #B8D99A;
}
.assistant-message {
    background-color: #E8E8E8;
    color: #000000 !important;
    border: 1px solid #C0C0C0;
}
/* Ensure text is always visible with high contrast */
.chat-message, .chat-message *, .chat-message p, .chat-message div, .chat-message span {
    color: #000000 !important;
}
/* Override any Streamlit theme colors */
.chat-message {
    background-color: inherit;
}
.user-message {
    background-color: #DCF8C6 !important;
}
.assistant-message {
    background-color: #E8E8E8 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📄 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload your files", type=["pdf"], accept_multiple_files=True
)


@st.cache_resource
def cached_llm():
    return get_llm()


@st.cache_resource
def cached_vectordb(files):
    return build_vectordb_from_uploaded_files(files)


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 Chat with Your PDFs")

if not uploaded_files:
    st.info("Upload PDFs to begin.")
    st.stop()

vectordb = cached_vectordb(uploaded_files)
llm = cached_llm()

# Show chat history
for role, content in st.session_state.messages:
    bubble = "user-message" if role == "user" else "assistant-message"
    # Escape HTML to prevent invalid tag errors
    escaped_content = html.escape(content)
    st.markdown(
        f"<div class='chat-message {bubble}'>{escaped_content}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Ask something about your documents...")

if user_input:
    # Display user msg
    st.session_state.messages.append(("user", user_input))
    escaped_user_input = html.escape(user_input)
    st.markdown(
        f"<div class='chat-message user-message'>{escaped_user_input}</div>", unsafe_allow_html=True)

    # RAG prompt with conversation history
    # Get previous messages (excluding the current one we just added)
    conversation_history = st.session_state.messages[:-1] if len(
        st.session_state.messages) > 1 else []
    prompt, docs = answer_question(
        vectordb, llm, user_input, conversation_history=conversation_history)

    # Stream result
    streamed = ""
    placeholder = st.empty()

    try:
        # Show loading indicator
        with placeholder.container():
            st.info("Generating answer...")

        # Stream tokens from the model
        for token in stream_llm_answer(llm, prompt, max_new_tokens=256):
            if token:  # Only add non-empty tokens
                streamed += token
                # Escape HTML to prevent invalid tag errors
                escaped_streamed = html.escape(streamed)
                placeholder.markdown(
                    f"<div class='chat-message assistant-message'>{escaped_streamed}</div>",
                    unsafe_allow_html=True
                )

        # If no output was generated, show a message
        if not streamed.strip():
            streamed = "I'm sorry, I couldn't generate a response. Please try again."
            escaped_streamed = html.escape(streamed)
            placeholder.markdown(
                f"<div class='chat-message assistant-message'>{escaped_streamed}</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        streamed = error_msg
        escaped_error = html.escape(error_msg)
        placeholder.markdown(
            f"<div class='chat-message assistant-message'>{escaped_error}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append(("assistant", streamed))

    with st.expander("Retrieved Chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(d.page_content)
