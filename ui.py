import streamlit as st
from typing import Callable, Dict, Any
from datetime import datetime

class BibleChatUI:
    def __init__(self):
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def get_greeting(self) -> str:
        """Return appropriate greeting based on time of day."""
        hour = datetime.now().hour
        if hour < 12:
            return "Good Morning"
        elif hour < 17:
            return "Good Afternoon"
        else:
            return "Good Evening"

    def setup_ui(self):
        """Setup the main UI components."""
        # Main title and greeting
        st.title("Bible Assistant")
        st.markdown("")
        st.markdown(f"{self.get_greeting()}! How can I help you today?")
        
        # Sidebar with subheader, New Chat button, and user guide
        with st.sidebar:
            st.markdown("<h2 style='font-size:24px;'>Talk with Holy Bible!</h2>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
            
            # User Guide Section
            st.markdown("---")
            st.markdown("### 📖 User Guide")
            with st.expander("How to use"):
                st.markdown("""
                **Getting Started:**
                1. Type your question in the chat box
                2. Press Enter or click Send
                3. Wait for the response
                
                **Tips for better results:**
                - Be specific in your questions
                - Include Bible verse references if known
                - Ask one question at a time
                - Use clear, simple language
                
                **Example questions:**
                - What does John 3:16 mean?
                - Who was Moses?
                - What is the story of creation?
                - Explain the Lord's Prayer
                """)

    def display_chat_history(self):
        """Display chat messages from history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_user_input(self) -> str:
        """Get user input from chat interface."""
        return st.chat_input("Ask a question about the Bible...")

    def update_chat(self, role: str, content: str):
        """Update chat history with new message."""
        st.session_state.messages.append({"role": role, "content": content})

    def display_response(self, response: str):
        """Display assistant response in chat."""
        with st.chat_message("assistant"):
            st.markdown(response)

    def show_error(self, error: str):
        """Display error message."""
        st.error(error)

    def show_spinner(self, text: str) -> st.spinner:
        """Create a spinner context manager."""
        return st.spinner(text)