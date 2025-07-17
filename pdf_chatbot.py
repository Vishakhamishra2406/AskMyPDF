import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from utils import ensure_event_loop
import google.generativeai as genai

class PDFChatbot:
    def __init__(self, google_api_key):
        self.google_api_key = google_api_key
        self.vectorstore = None
        self.retriever = None

    def extract_text(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {e}")

    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        """Split text into manageable chunks for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def embed_and_store(self, texts):
        """Embed text chunks and store them in a FAISS vector store using Gemini embeddings."""
        try:
            ensure_event_loop()
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-exp-03-07",
                google_api_key=self.google_api_key
            )
            self.vectorstore = FAISS.from_texts(texts, embeddings)
            self.retriever = self.vectorstore.as_retriever()
        except Exception as e:
            raise RuntimeError(f"Embedding or FAISS storage failed: {e}")

    def ask(self, query):
        """Query the PDF content using Gemini API directly with retrieved context."""
        if not self.retriever:
            raise RuntimeError("No retriever found. Please upload and process a PDF first.")
        try:
            ensure_event_loop()
            # Retrieve relevant context from the vector store
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Prepare the prompt for Gemini
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            # Set up Gemini client
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")  # or another available model

            # Call Gemini API
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to get answer: {e}") 