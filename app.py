import streamlit as st
import google.generativeai as genai 
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import PyPDF2
from typing import List, Dict, Union
import os
from urllib.parse import urlparse, urljoin
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from io import BytesIO 
import time

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=100,
            length_function=len,
        )
        self.max_pages = 5
        self.timeout = 10  # Timeout for requests
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch URL content asynchronously."""
        try:
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
                return ""
        except Exception as e:
            st.warning(f"Error fetching {url}: {str(e)}")
            return ""

    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract valid links from page, with improved filtering."""
        base_url = urlparse(url)
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
                
            absolute_url = urljoin(url, href)
            parsed_url = urlparse(absolute_url)
            
            # More stringent link filtering
            if (parsed_url.netloc == base_url.netloc and 
                parsed_url.scheme in ['http', 'https'] and
                not any(ext in parsed_url.path for ext in ['.pdf', '.jpg', '.png', '.gif'])):
                links.append(absolute_url)
                
        return list(set(links))

    async def extract_from_website(self, url: str) -> str:
        """Extract text content from website with async crawling."""
        visited = set()
        to_visit = [url]
        combined_text = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        async with aiohttp.ClientSession(headers=headers) as session:
            while to_visit and len(visited) < self.max_pages:
                # Process multiple URLs concurrently
                batch_size = min(5, len(to_visit))  # Process up to 5 URLs at once
                current_batch = to_visit[:batch_size]
                to_visit = to_visit[batch_size:]
                
                tasks = [self.fetch_url(session, url) for url in current_batch]
                responses = await asyncio.gather(*tasks)
                
                for current_url, html_content in zip(current_batch, responses):
                    if not html_content or current_url in visited:
                        continue
                        
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'header', 'footer', 'nav', 'meta', 'link']):
                        element.decompose()
                    
                    # Extract meaningful text content
                    paragraphs = []
                    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article']):
                        text = p.get_text(strip=True)
                        if len(text) > 50:  # Only keep substantial paragraphs
                            paragraphs.append(text)
                    
                    if paragraphs:
                        combined_text.append('\n'.join(paragraphs))
                        
                    visited.add(current_url)
                    
                    # Get new links if needed
                    if len(visited) < self.max_pages:
                        new_links = self.get_links(current_url, soup)
                        to_visit.extend([link for link in new_links if link not in visited])
                    
                    # Update progress
                    progress = len(visited) / self.max_pages
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {len(visited)} pages out of {self.max_pages}")
        
        progress_bar.empty()
        status_text.empty()
        
        if not combined_text:
            raise Exception("No content could be extracted from the website")
        
        return "\n\n".join(combined_text)

    def extract_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF with improved handling."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) > 100:  # Only keep substantial content
                        text_content.append(cleaned_text)
            
            return "\n\n".join(text_content)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with improved handling."""
        if not text.strip():
            raise ValueError("Empty text provided for splitting")
            
        # Remove any extremely short lines or noise
        lines = [line for line in text.split('\n') if len(line.strip()) > 50]
        cleaned_text = '\n'.join(lines)
        
        return self.text_splitter.split_text(cleaned_text)

class VectorStore:
    def __init__(self, embedding_dimension: int = 768):
        # Using IVFFlat index for faster similarity search
        self.dimension = embedding_dimension
        self.index = None
        self.texts = []
        
    def init_index(self, initial_vectors: np.ndarray):
        """Initialize FAISS index with actual vectors."""
        nlist = min(len(initial_vectors), 50)  # Number of clusters
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        self.index.train(initial_vectors)
        self.index.add(initial_vectors)
        
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        """Add texts and their embeddings to the vector store."""
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts and embeddings must match")
            
        if self.index is None:
            self.init_index(embeddings)
        else:
            self.index.add(embeddings)
            
        self.texts.extend(texts)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Search for most similar texts with improved handling."""
        if not self.texts:
            raise ValueError("No texts in the vector store")
            
        k = min(k, len(self.texts))
        self.index.nprobe = min(32, len(self.texts))  # Improve search accuracy
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Filter out low-quality matches
        valid_results = []
        for d, idx in zip(distances[0], indices[0]):
            if idx >= 0 and d < 10:  # Distance threshold
                valid_results.append(self.texts[idx])
                
        return valid_results if valid_results else [self.texts[indices[0][0]]]

class RAGChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = genai.GenerativeModel('models/text-embedding-004')
        self.vector_store = None  # Initialize later with actual data
        self.doc_processor = DocumentProcessor()
        
    async def process_source(self, source: Union[str, BytesIO], source_type: str, max_pages: int = 5) -> int:
        """Process source with better error handling and feedback."""
        try:
            self.doc_processor.set_max_pages(max_pages)
            
            # Extract text based on source type
            if source_type == "website":
                text = await self.doc_processor.extract_from_website(source)
            else:  # PDF
                text = self.doc_processor.extract_from_pdf(source)
            
            if not text.strip():
                raise ValueError("No meaningful text content extracted from source")
                
            # Split text into chunks
            chunks = self.doc_processor.split_text(text)
            
            if not chunks:
                raise ValueError("No valid text chunks generated")
                
            # Process embeddings in batches
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings = [self.get_embedding(chunk) for chunk in batch]
                all_embeddings.extend(embeddings)
                
            embeddings_array = np.vstack(all_embeddings)
            
            # Initialize or update vector store
            if self.vector_store is None:
                self.vector_store = VectorStore(embedding_dimension=768)
                
            self.vector_store.add_texts(chunks, embeddings_array)
            return len(chunks)
            
        except Exception as e:
            raise Exception(f"Error processing source: {str(e)}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings with improved error handling."""
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
            
            # Ensure consistent dimensionality
            target_dim = 768
            if len(embedding) < target_dim:
                embedding = np.pad(embedding, (0, target_dim - len(embedding)))
            else:
                embedding = embedding[:target_dim]
                
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def generate_response(self, query: str) -> str:
        """Generate response with improved context handling."""
        try:
            if not query.strip():
                return "Please provide a valid question."
                
            if not self.vector_store:
                return "No content has been loaded yet. Please add a website or PDF first."
                
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get relevant context
            relevant_texts = self.vector_store.similarity_search(query_embedding, k=3)
            
            if not relevant_texts:
                return "I couldn't find relevant information to answer your question."
                
            # Combine context with clear separation
            context = "\n---\n".join(relevant_texts)
            
            # Generate response with improved prompt
            prompt = f"""Based on the following context, please answer the question clearly and concisely.
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_output_tokens": 1024
                }
            )
            
            return response.text if hasattr(response, 'text') else "I couldn't generate a response. Please try again."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

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
                    # Use asyncio to run async code
                    num_chunks = asyncio.run(st.session_state.chatbot.process_source(
                        website_url,
                        "website",
                        max_pages
                    ))
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
                        num_chunks = asyncio.run(st.session_state.chatbot.process_source(pdf_file, "pdf"))
                        st.success(f"✅ Successfully processed PDF into {num_chunks} chunks!")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages with improved styling
    message_container = st.container()
    with message_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input with improved error handling and user feedback
    if prompt := st.chat_input("Ask a question about the content...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not st.session_state.chatbot.vector_store:
                        response = "Please add some content first by processing a website or uploading a PDF."
                    else:
                        response = st.session_state.chatbot.generate_response(prompt)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Add a clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
