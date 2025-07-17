import streamlit as st
from pdf_chatbot import PDFChatbot
from utils import load_api_key, save_uploaded_file

# Page ka title aise dete hn
st.set_page_config(page_title="PDF Chatbot", layout="centered")

# Headline ki designing kri hai html,css use krke
st.markdown(
    "<h1 style='text-align: center; color: Red; font-family: Courier;'>AskMyPDF</h1>",
    unsafe_allow_html=True #mandatory to allow html varna text format me dena output me
)

# Change background and sidebar color : just for designing css use kra h
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #e6e6fa;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True #ye jruri hai... permission deta h ki html allow krne ki
)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
<div style='font-family: Arial; font-size: 15px; color: white;'>
<h4> How to Use AskMyPDF</h4>
<ol>
<li>📄 <b>Upload</b> a PDF file using the upload button.</li>
<li>💬 <b>Ask questions</b> about the content of the uploaded PDF.</li>
<li>✅ <b>Ensure access is set up</b>: Make sure the app has access to an AI service to answer your questions.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Session state for chatbot
if 'chatbot' not in st.session_state:    # Check karta hai ki "chatbot" naam ka variable session state me hai ya nahi.
    st.session_state['chatbot'] = None   # agr nhi hota h to blank bna lo

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"]) # PDF upload karne ka button

if uploaded_file:
    try:    # Agar neeche koi line fail ho jaaye, toh error show hoga.
        api_key = load_api_key() #.env se api key uthaega
        file_path = save_uploaded_file(uploaded_file)  #local folder me save krega uploaded file ko
        chatbot = PDFChatbot(api_key) #PDFChatbot class ka object banaya jata hai jisme API key diya gaya hai. Ye object sari processing karega (text extract, split, embed, answer).
        text = chatbot.extract_text(file_path) # PDF se text extract karta h
        if not text.strip(): #agr text format me usse nhi milega to error show krega
            st.error("No text could be extracted from the PDF. Please try another file.")
            st.stop()
        chunks = chatbot.split_text(text) #Long PDF text ko chhote chhote chunks me todta hai, jise embeddings me convert kar sakein.
        if not chunks:
            st.error("The PDF was extracted, but no text chunks were created. Please try another file.")
            st.stop() #Agar splitting ke baad bhi kuch nahi mila, toh error dikhayega.
        chatbot.embed_and_store(chunks)    #Ye function chunks ko vector embeddings me convert karta hai (semantic understanding ke liye).Phir wo data ko memory me store karta hai taaki tum query kar sako.
        st.session_state['chatbot'] = chatbot  #Processed chatbot object ko session state me store kar diya jaata hai.(So that jab user question puche, woh chatbot already ready ho.)
        st.success("PDF processed and ready for questions!") #success ka message... abhi tak sab sahi chl rha h
    except Exception as e:
        st.error(f"Error processing PDF: {e}") #Agar upar ke kisi step me koi error aaya, toh woh message ke saath show karega.

if st.session_state.get('chatbot'):    #Agar chatbot object bana hua hai tabhi user se sawal puche jaayenge.
    query = st.text_input("Ask a question about the PDF:")  #ready to give answers about pdf
    if query: #Agar user ne kuch likha hai (not empty), tabhi chatbot chalega.
        with st.spinner("Thinking..."): #ye btata h ki chat bot kaam kr rha h(soch rha h: for user)
            try:
                answer = st.session_state['chatbot'].ask(query)  #ask(query) function call hota hai chatbot ke upar.Ye query ka answer generate karta hai using vector search + AI.
                st.markdown(f"**Answer:** {answer}") #answer dikhata h
            except Exception as e:
                st.error(f"Error: {e}")  #agr koi dikkat aaegi to error show kr dega

