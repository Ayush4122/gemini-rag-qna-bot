import streamlit as st
from typing import Union, List, Optional
from io import BytesIO
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import faiss
import numpy as np
import magic
import PyPDF2
import docx2txt

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class TextProcessor:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks of approximately equal size."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                if last_period != -1 and last_newline != -1:
                    end = max(last_period, last_newline) + 1
                elif last_period != -1:
                    end = last_period + 1
                elif last_newline != -1:
                    end = last_newline + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
        return chunks

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text."""
        text = ' '.join(text.split())
        text = text.replace('\x00', '')
        return text.strip()

class VectorStoreManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def add_texts(self, texts: List[str], embeddings: List[List[float]]):
        """Add texts and their embeddings to the vector store."""
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        
    def search(self, query_embedding: List[float], k: int = 3) -> List[str]:
        """Search for most similar texts given a query embedding."""
        query_array = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query_array, k)
        return [self.texts[i] for i in I[0]]

class DocumentProcessor:
    def __init__(self):
        self.max_pages = 5
        self.supported_types = ['pdf', 'txt', 'docx']

    def set_max_pages(self, max_pages: int) -> None:
        """Set the maximum number of pages to process."""
        self.max_pages = max_pages

    def detect_file_type(self, source: Union[str, BytesIO]) -> str:
        """Detect the type of file being processed."""
        if isinstance(source, str):
            mime = magic.from_file(source, mime=True)
        else:
            mime = magic.from_buffer(source.getvalue(), mime=True)
        
        if 'pdf' in mime:
            return 'pdf'
        elif 'text' in mime:
            return 'txt'
        elif 'docx' in mime or 'openxmlformats-officedocument' in mime:
            return 'docx'
        else:
            raise ValueError(f"Unsupported file type: {mime}")

    def process_pdf(self, source: Union[str, BytesIO]) -> List[str]:
        """Process PDF files."""
        if isinstance(source, str):
            pdf_file = open(source, 'rb')
        else:
            pdf_file = source

        pdf_reader = PyPDF2.PdfReader(pdf_file)
        texts = []
        
        for page in list(pdf_reader.pages)[:self.max_pages]:
            texts.append(page.extract_text())
            
        if isinstance(source, str):
            pdf_file.close()
            
        return texts

    def process_txt(self, source: Union[str, BytesIO]) -> List[str]:
        """Process text files."""
        if isinstance(source, str):
            with open(source, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            text = source.getvalue().decode('utf-8')
        return [text]

    def process_docx(self, source: Union[str, BytesIO]) -> List[str]:
        """Process DOCX files."""
        if isinstance(source, str):
            text = docx2txt.process(source)
        else:
            text = docx2txt.process(source)
        return [text]

    def process_document(self, source: Union[str, BytesIO]) -> List[str]:
        """Main method to process any supported document type."""
        file_type = self.detect_file_type(source)
        
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if file_type == 'pdf':
            return self.process_pdf(source)
        elif file_type == 'txt':
            return self.process_txt(source)
        elif file_type == 'docx':
            return self.process_docx(source)

class RAGChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = genai.GenerativeModel('models/embedding-001')
        self.text_processor = TextProcessor()
        self.vector_store = None
        self.doc_processor = DocumentProcessor()

    async def process_source(self, source: Union[str, BytesIO], source_type: str, max_pages: int = 5) -> int:
        """Process source with better error handling and feedback."""
        try:
            self.doc_processor.set_max_pages(max_pages)
            raw_texts = self.doc_processor.process_document(source)
            
            processed_texts = []
            for text in raw_texts:
                clean_text = self.text_processor.preprocess_text(text)
                chunks = self.text_processor.chunk_text(clean_text)
                processed_texts.extend(chunks)
            
            embeddings = []
            for text in processed_texts:
                embedding = await self.embedding_model.embed_content(
                    text=text,
                    task_type="retrieval_document",
                )
                embeddings.append(embedding)
            
            if self.vector_store is None:
                self.vector_store = VectorStoreManager(len(embeddings[0]))
            
            self.vector_store.add_texts(processed_texts, embeddings)
            
            return len(raw_texts)
            
        except Exception as e:
            raise Exception(f"Error processing source: {str(e)}")

    async def query(self, question: str, num_results: int = 3) -> str:
        """Query the RAG system with a question."""
        try:
            if self.vector_store is None:
                raise ValueError("No documents have been processed yet.")
                
            question_embedding = await self.embedding_model.embed_content(
                text=question,
                task_type="retrieval_query",
            )
            
            relevant_texts = self.vector_store.search(question_embedding, k=num_results)
            context = "\n\n".join(relevant_texts)
            
            prompt = f"""Context: {context}

Question: {question}

Based on the provided context, please answer the question thoroughly and accurately. Only use information from the context. If the context doesn't contain enough information to answer the question fully, please say so."""

            response = await self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            raise Exception(f"Error querying RAG system: {str(e)}")

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Streamlit interface
st.title("RAG Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or DOCX)", type=['pdf', 'txt', 'docx'])

def run_async(func):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(func)


async def init_chatbot():
    if st.session_state.chatbot is None:
        st.session_state.chatbot = RAGChatbot()

async def process_uploaded_file():
    if uploaded_file and not st.session_state.document_processed:
        try:
            bytes_data = BytesIO(uploaded_file.getvalue())
            
            num_pages = run_sync(st.session_state.chatbot.process_source(
                bytes_data, 
                uploaded_file.type,
                max_pages=5
            )
                                )
            
            st.session_state.document_processed = True
            st.success(f"Successfully processed {num_pages} pages from the document!")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.session_state.document_processed = False

# Chat interface
if st.session_state.document_processed:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get chatbot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = run_sync(st.session_state.chatbot.query(prompt))
                message_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                message_placeholder.error(f"Error generating response: {str(e)}")
