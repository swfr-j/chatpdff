import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.header("Chat with multiple pdfs")
    st.text_input("Ask a question about your documents: ")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your pdfs here and click on process",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Process"):
            # get pdf text
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            with st.spinner("Splitting text into chunks..."):
                chunks = get_text_chunks(raw_text)

            # Create vector store
            with st.spinner("Creating vector store..."):
                vector_store = get_vector_store(chunks)

            # create conversation chain
            conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
