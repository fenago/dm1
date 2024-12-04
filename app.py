import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        # Read and decode uploaded file
        documents = [uploaded_file.read().decode()]
        # Split document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Store embeddings in a vector database
        db = Chroma.from_documents(texts, embeddings)
        # Create a retriever for the vector database
        retriever = db.as_retriever()
        # Use a RetrievalQA chain for question-answering
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Streamlit app configuration
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File uploader for text document
uploaded_file = st.file_uploader('Upload a text document', type='txt')

# Input for user's query
query_text = st.text_input('Enter your question:', placeholder='Type your question here...', disabled=not uploaded_file)

# Form for OpenAI API key input and submission
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key  # Remove API key for security

# Display the result
if len(result):
    st.info(result[-1])
