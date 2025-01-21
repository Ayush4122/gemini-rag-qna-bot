# app.py
import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from typing import List, Dict, Union
import os
from urllib.parse import urlparse
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def clean_text(self, text: str) -> str:
        """Remove special characters and extra whitespace."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def extract_from_website(self, url: str) -> str:
        """Extract text content from a website."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            return self.clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from website: {str(e)}")
    
    def extract_from_pdf(self, pdf_file) -> str:
        """Extract text content from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return self.clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.text_splitter.split_text(text)

class VectorStore:
    def __init__(self, embedding_dimension: int = 768):
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.texts = []
        
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        """Add texts and their embeddings to the vector store."""
        self.index.add(embeddings)
        self.texts.extend(texts)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Search for most similar texts given a query embedding."""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.texts[i] for i in indices[0]]

class RAGChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = genai.GenerativeModel('models/embedding-001')
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings for a text using Gemini."""
        try:
            embedding = self.embedding_model.embed_content(
                text,
                task_type="retrieval_query"
            )
            return np.array(embedding.values)
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def process_source(self, source: Union[str, io.BytesIO], source_type: str):
        """Process either website URL or PDF file."""
        try:
            # Extract text based on source type
            if source_type == "website":
                text = self.doc_processor.extract_from_website(source)
            else:  # PDF
                text = self.doc_processor.extract_from_pdf(source)
                
            # Split text into chunks
            chunks = self.doc_processor.split_text(text)
            
            # Get embeddings for all chunks
            embeddings = np.vstack([self.get_embedding(chunk) for chunk in chunks])
            
            # Add to vector store
            self.vector_store.add_texts(chunks, embeddings)
            
            return len(chunks)
            
        except Exception as e:
            raise Exception(f"Error processing source: {str(e)}")
    
    def generate_response(self, query: str, max_tokens: int = 1024) -> str:
        """Generate response using RAG approach."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get relevant context
            relevant_texts = self.vector_store.similarity_search(query_embedding)
            
            # Construct prompt with context
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say so.
            
            Context:
            {' '.join(relevant_texts)}
            
            Question: {query}
            
            Answer:"""
            
            # Generate response
            response = self.model.generate_content(prompt, max_output_tokens=max_tokens)
            return response.text
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    
    st.title("RAG-based Q&A Chatbot")
    initialize_session_state()
    
    # Sidebar for uploading documents
    with st.sidebar:
        st.header("Upload Content")
        
        # Website URL input
        website_url = st.text_input("Enter website URL:")
        if st.button("Process Website") and website_url:
            with st.spinner("Processing website content..."):
                try:
                    num_chunks = st.session_state.chatbot.process_source(website_url, "website")
                    st.success(f"Successfully processed website into {num_chunks} chunks!")
                except Exception as e:
                    st.error(f"Error processing website: {str(e)}")
        
        # PDF upload
        pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
        if pdf_file is not None:
            with st.spinner("Processing PDF..."):
                try:
                    num_chunks = st.session_state.chatbot.process_source(pdf_file, "pdf")
                    st.success(f"Successfully processed PDF into {num_chunks} chunks!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the content:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
