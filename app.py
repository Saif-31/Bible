from typing import Dict, List, Optional, Union
import os
from pathlib import Path

# Third-party imports
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from ui import BibleChatUI

# Configure torch
import torch
torch.set_default_tensor_type(torch.FloatTensor)

# Constants
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
PINECONE_INDEX_NAME = "data"
PINECONE_NAMESPACE = "text_chunks"
RETRIEVER_K = 4

# Load environment variables
load_dotenv()

def init_environment() -> None:
    """Initialize environment variables."""
    required_vars = [
        "MISTRAL_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT"
    ]
    
    # Check for variables in Streamlit secrets
    for var in required_vars:
        if var not in st.secrets:
            st.error(f"Missing required secret: {var}")
            st.info("Please add the required secrets in Streamlit dashboard.")
            st.stop()

    # Set environment variables from Streamlit secrets
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

def init_pinecone() -> PineconeVectorStore:
    """Initialize Pinecone vector store."""
    try:
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="us-east-1"
        )
        index = pc.Index(PINECONE_INDEX_NAME)
        
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        return PineconeVectorStore(
            index=index,
            embedding=embedding_function,
            text_key='text',
            namespace=PINECONE_NAMESPACE
        )
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        st.stop()

def init_llm(system_prompt: str) -> ChatMistralAI:
    """Initialize the language model."""
    return ChatMistralAI(
        model="mistral-large-latest",
        system_message=system_prompt
    )

@st.cache_data(show_spinner=False)
def process_query(query: str, 
                 _refinement_chain: RunnableLambda,  # Added underscore to prevent hashing
                 _retrieval_chain: RetrievalQA) -> str:  # Added underscore to prevent hashing
    """
    Process a user query and return the bot's response.
    
    Args:
        query: User's question
        _refinement_chain: Chain for query refinement (not hashed)
        _retrieval_chain: Chain for retrieving answers (not hashed)
        
    Returns:
        str: Bot's formatted response
    """
    try:
        refined_query_msg = _refinement_chain.invoke({"original_question": query})
        refined_query = (
            refined_query_msg.get("text", "").strip()
            if isinstance(refined_query_msg, dict)
            else getattr(refined_query_msg, 'content', str(refined_query_msg)).strip()
        )
        
        response_msg = _retrieval_chain.invoke(refined_query)
        response = (
            response_msg.get("result", "")
            if isinstance(response_msg, dict)
            else getattr(response_msg, 'content', str(response_msg))
        )
        
        return response
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I apologize, but I encountered an error processing your question. Please try again."

def main():
    """Main application entry point."""
    # Initialize UI
    ui = BibleChatUI()
    ui.setup_ui()
    
    try:
        # Initialize components and system prompt
        vectorstore = init_pinecone()
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        
        # Load system prompt
        try:
            system_prompt = Path("system_prompt.txt").read_text()
        except FileNotFoundError:
            ui.show_error("System prompt file not found!")
            st.stop()
        
        # Initialize LLM and cache
        llm = init_llm(system_prompt)
        set_llm_cache(InMemoryCache())
        
        # Initialize chains
        refinement_chain, retrieval_chain = initialize_chains(llm, retriever)
        
        # Display chat history
        ui.display_chat_history()
        
        # Get user input and process
        if prompt := ui.get_user_input():
            ui.update_chat("user", prompt)
            
            with ui.show_spinner("Finding answer..."):
                response = process_query(prompt, refinement_chain, retrieval_chain)
                ui.update_chat("assistant", response)
                st.rerun()
                
    except Exception as e:
        ui.show_error(f"An error occurred: {str(e)}")
        st.stop()

def initialize_chains(llm, retriever):
    """Initialize the refinement and retrieval chains."""
    refinement_prompt = PromptTemplate(
        input_variables=["original_question"],
        template="Create a focused Bible search query based on: {original_question}"
    )
    refinement_chain = refinement_prompt | llm
    
    retrieval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Bible assistant."),
        ("human", (
            "Based on this context:\n"
            "{context}\n\n"
            "Please answer this question:\n"
            "{question}"
        ))
    ])
    
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": retrieval_prompt}
    )
    
    return refinement_chain, retrieval_chain

if __name__ == "__main__":
    main()
