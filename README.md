# AskMyPDF

# 📄 AskMyPDF - GenAI Powered PDF Chatbot

AskMyPDF is an intelligent chatbot that allows users to upload any PDF document and interact with its content through natural language queries. Whether it’s a research paper, textbook, or instruction manual — just upload and ask!

## 💡 Features

- Upload any PDF file and ask questions about it.
- Utilizes Google's Generative AI (Gemini) for intelligent responses.
- User-friendly Streamlit interface.

## 🚀 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Libraries Used**:  
  - `PyMuPDF` (`fitz`) – for PDF parsing  
  - `Langchain` – for text splitting and RetrievalQA chain  
  - `FAISS` – for efficient vector search  
  - `GoogleGenerativeAI` – for generating responses  
  - `dotenv` – for API key management  

## 📦 Setup Instructions

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Create a .env file in the root directory and add your Gemini API key:

ini
Copy
Edit
GOOGLE_API_KEY=your_api_key_here
Run the app:

bash
Copy
Edit
streamlit run app.py
📸 Screenshots
<img width="1907" height="902" alt="image" src="https://github.com/user-attachments/assets/c7cc94de-4b75-4ad5-a464-6f15c4c5caa1" />
