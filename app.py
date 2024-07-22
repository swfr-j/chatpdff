import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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
            with st.spinner("Processing your documents..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

            with st.spinner("Splitting text into chunks..."):
                # get the text chunks
                chunks = get_text_chunks(raw_text)

            with st.spinner("Creating vector store..."):
                # Create vector store
                pass


if __name__ == "__main__":
    main()
