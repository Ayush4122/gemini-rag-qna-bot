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

# Configure environment variables
if 'GOOGLE_API_KEY' not in os.environ:
    st.error("Please set your GOOGLE_API_KEY environment variable")
    st.stop()

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for better processing
            chunk_overlap=50,
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            return self.clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from website: {str(e)}")
    
    def extract_from_pdf(self, pdf_file) -> str:
        """Extract text content from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
            return self.clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text.strip():
            raise ValueError("Empty text provided for splitting")
        return self.text_splitter.split_text(text)

class VectorStore:
    def __init__(self, embedding_dimension: int = 768):
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.texts = []
        
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        """Add texts and their embeddings to the vector store."""
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts and embeddings must match")
        self.index.add(embeddings)
        self.texts.extend(texts)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Search for most similar texts given a query embedding."""
        if len(self.texts) == 0:
            raise ValueError("No texts in the vector store")
        k = min(k, len(self.texts))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.texts[i] for i in indices[0]]

class RAGChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = genai.GenerativeModel('models/embedding-001')
        self.vector_store = VectorStore(embedding_dimension=768)
        self.doc_processor = DocumentProcessor()
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings for a text using Gemini."""
        try:
            result = self.embedding_model.generate_content(
                text,
                generation_config={"temperature": 0.0}
            )
            
            if not hasattr(result, 'text'):
                raise ValueError("No embedding generated")
                
            # Convert text to numerical values and normalize
            text_values = [ord(c) for c in result.text]
            embedding = np.array(text_values, dtype=np.float32)
            
            # Pad or truncate to match embedding dimension
            target_dim = 768
            if len(embedding) < target_dim:
                embedding = np.pad(embedding, (0, target_dim - len(embedding)))
            else:
                embedding = embedding[:target_dim]
                
            # Normalize the embedding
            normalized_embedding = embedding / np.linalg.norm(embedding)
            return normalized_embedding
            
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
            
            if not text.strip():
                raise ValueError("No text content extracted from source")
                
            # Split text into chunks
            chunks = self.doc_processor.split_text(text)
            
            # Get embeddings for all chunks
            embeddings = []
            for chunk in chunks:
                chunk_embedding = self.get_embedding(chunk)
                embeddings.append(chunk_embedding)
            
            # Convert list of embeddings to numpy array
            embeddings_array = np.vstack(embeddings)
            
            # Add to vector store
            self.vector_store.add_texts(chunks, embeddings_array)
            
            return len(chunks)
            
        except Exception as e:
            raise Exception(f"Error processing source: {str(e)}")
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG approach."""
        try:
            if not query.strip():
                return "Please provide a valid question."
                
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get relevant context
            try:
                relevant_texts = self.vector_store.similarity_search(query_embedding)
                context = " ".join(relevant_texts)
            except ValueError:
                return "No content has been loaded yet. Please add a website or PDF first."
            
            # Construct prompt with context
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.3},
                safety_settings=self.safety_settings
            )
            
            return response.text if hasattr(response, 'text') else "I couldn't generate a response. Please try again."
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG-based Q&A Chatbot")
    initialize_session_state()
    
    # Sidebar for uploading documents
    with st.sidebar:
        st.header("📚 Upload Content")
        
        # Website URL input
        website_url = st.text_input("🌐 Enter website URL:")
        if st.button("Process Website", key="process_website") and website_url:
            with st.spinner("Processing website content..."):
                try:
                    num_chunks = st.session_state.chatbot.process_source(website_url, "website")
                    st.success(f"✅ Successfully processed website into {num_chunks} chunks!")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
        
        st.markdown("---")
        
        # PDF upload
        st.write("📄 Upload PDF Document")
        pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if pdf_file is not None:
            if st.button("Process PDF", key="process_pdf"):
                with st.spinner("Processing PDF..."):
                    try:
                        num_chunks = st.session_state.chatbot.process_source(pdf_file, "pdf")
                        st.success(f"✅ Successfully processed PDF into {num_chunks} chunks!")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the content..."):
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
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
