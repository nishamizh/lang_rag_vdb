import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader,PyPDFLoader)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name

        try:
            temp_file.flush()
            #loads the text file using TextLoader
            loader = TextLoader(temp_file_path)
            documents = loader.load()

            print(f"Loaded {len(documents)} document(s)")
            print(f"Content Preview: {documents[0].page_content[:100]}...")
            print(f"MetaData: {documents[0].metadata}")

            # Print the loaded documents:
            # for doc in documents:
            #    print("Document Contents:")
            #    print(doc)
            #    print(doc.page_content)
        
        finally:
            #cleans up the temporary file
            os.remove(temp_file_path)

def doc_structure():
    """
    """


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Laoded {len(documents)} document(s) from PDF")

    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    #load_text_file()
    pdf_loader("/Users/nisha/Documents/learn_ai_with_nisha/lang_rag_vdb/docs/langchain_demo.pdf")