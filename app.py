import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.openai import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template, css

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
    # llm = HuggingFaceHub(
    #     reon_id="google/flan-t5-xxl",
    #     model_kwargs={"temperature": 0.5, "max_length": 100},
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple pdfs")
    user_question = st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_user_input(user_question)

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
            st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
