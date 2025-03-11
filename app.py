import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # ✅ Removed extra space in key name
genai.configure(api_key=api_key)

# ✅ Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

# ✅ Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# ✅ Function to generate FAISS vector store
def get_vector_stores(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # ✅ Ensure old index is deleted before creating a new one
    if os.path.exists("faiss_index"):
        import shutil
        shutil.rmtree("faiss_index")

    # ✅ Create FAISS index
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    print("✅ FAISS index created successfully!")

# ✅ Function to generate the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide the correct answer.
    If you don't know the answer, just say "I don't know", but do not provide the wrong answer.

    Context: \n{context}\n
    Question: \n{question}\n
    Answer:"""


    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template  # ✅ Fixed incorrect reference
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ✅ Function to process user input and query FAISS
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ✅ Check if FAISS index exists before loading
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("⚠️ FAISS index file not found! Please upload PDFs and process them first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.subheader("The Response is:")
    st.write(response["output_text"])

# ✅ Main function for Streamlit App
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📄")
    st.header("📖 Chat with PDFs using Gemini AI")

    # ✅ User Input for Question
    user_question = st.text_input("Ask a question from the PDF file")

    if user_question:  # ✅ Corrected condition
        user_input(user_question)

    with st.sidebar:
        st.title("📌 Menu")
        pdf_docs = st.file_uploader("📂 Upload your PDF files", accept_multiple_files=True)

        if st.button("📤 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text
                    text_chunks = get_text_chunks(raw_text)  # Split text into chunks
                    get_vector_stores(text_chunks)  # ✅ Save FAISS index
                    st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
