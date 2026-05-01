import streamlit as st
import time
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API key
#os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader=PyPDFDirectoryLoader("Research Papers")
        st.session_state.docs=st.session_state.loader.load()#Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Rag Chatbot with Groq")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Ready")

## Using Rag Chain as a Retriever
if user_prompt:
    retriever=st.session_state.vectors.as_retriever()
    
    rag_chain=(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    )
   
    start=time.process_time()
    response=rag_chain.invoke(user_prompt)

    st.write(response.content)
    st.write(f"Response time: {time.process_time()-start}")



    ## With a streamlit expander
    # it will show the retrieved docs

    with st.expander("Document similarity Search"):
        docs=retriever.get_relevant_documents(user_prompt)

        for i,doc in enumerate(response['content']):
            st.write(doc.page_content)
            st.write('-------------------')
            
