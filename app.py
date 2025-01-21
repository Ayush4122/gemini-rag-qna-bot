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
from urllib.parse import urlparse, urljoin
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# # Configure environment variables
# if 'GOOGLE_API_KEY' not in os.environ:
#     st.error("Please set your GOOGLE_API_KEY environment variable")
#     st.stop()

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self.max_pages = 5  # Default value
        
    def set_max_pages(self, max_pages: int):
        """Set maximum pages to crawl."""
        self.max_pages = max_pages
        
    def clean_text(self, text: str) -> str:
        """Remove special characters and extra whitespace."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract valid links from the page."""
        base_url = urlparse(url)
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(url, href)
            parsed_url = urlparse(absolute_url)
            
            if parsed_url.netloc == base_url.netloc and parsed_url.scheme in ['http', 'https']:
                links.append(absolute_url)
                
        return list(set(links))  # Remove duplicates
    
    def extract_from_website(self, url: str) -> str:
        """Extract text content from website with recursive crawling."""
        visited = set()
        to_visit = [url]
        combined_text = []
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        while to_visit and len(visited) < self.max_pages:
            current_url = to_visit.pop(0)
            
            if current_url not in visited:
                try:
                    response = requests.get(current_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                        element.decompose()
                    
                    # Extract text
                    text = soup.get_text(separator=' ', strip=True)
                    combined_text.append(text)
                    visited.add(current_url)
                    
                    # Get new links
                    if len(visited) < self.max_pages:
                        new_links = self.get_links(current_url, soup)
                        to_visit.extend([link for link in new_links if link not in visited])
                    
                    # Update progress
                    progress = len(visited) / self.max_pages
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {len(visited)} pages out of {self.max_pages}")
                    time.sleep(1)  # To prevent too rapid requests
                    
                except Exception as e:
                    st.warning(f"Error processing {current_url}: {str(e)}")
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not combined_text:
            raise Exception("No content could be extracted from the website")
        
        return "\n\n".join(combined_text)
    
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
    
    def process_source(self, source: Union[str, io.BytesIO], source_type: str, max_pages: int = 5):
        """Process either website URL or PDF file."""
        try:
            # Set max pages for website crawling
            self.doc_processor.set_max_pages(max_pages)
            
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
            
            # Generate response
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.3}
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
        st.write("🌐 Enter Website Details")
        website_url = st.text_input("Website URL:")
        max_pages = st.number_input("Maximum pages to crawl:", min_value=1, max_value=20, value=5)
        
        if st.button("Process Website", key="process_website") and website_url:
            with st.spinner(f"Processing website content (up to {max_pages} pages)..."):
                try:
                    num_chunks = st.session_state.chatbot.process_source(
                        website_url,  # Pass URL directly
                        "website",
                        max_pages  # Pass max_pages as separate parameter
                    )
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
