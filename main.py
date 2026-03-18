

import os
import re
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:?!()'\"%\-\s]", "", text)
    return text.strip()


def build_subject_db(subject):
    pdf_folder = f"subjects/{subject}/pdfs"
    db_folder = f"vector_dbs/{subject}_db"

    if not os.path.exists(pdf_folder):
        print(f"❌ PDF folder not found → {pdf_folder}")
        return

    if not os.path.exists("vector_dbs"):
        os.makedirs("vector_dbs")

    print(f"\n Building Vector DB for → {subject.upper()}")
    time.sleep(1)

    docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠ No PDFs found!")
        return

    for f in pdf_files:
        print(f" Loading {f}")
        loader = PyPDFLoader(os.path.join(pdf_folder, f))
        pages = loader.load()

        for p in pages:
            p.page_content = clean_text(p.page_content)

        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_folder)

    print(f" Saved → {db_folder}")


if __name__ == "__main__":
    print("Available: aps | constitution")
    sub = input("Enter subject name: ").lower().strip()
    build_subject_db(sub)
