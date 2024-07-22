import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()


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
                pass

                # get the text chunks

                # Create vector store


if __name__ == "__main__":
    main()
