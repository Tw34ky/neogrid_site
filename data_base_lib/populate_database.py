import os
import shutil
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def populate_database(params, documents: list[str]):
    chunks = split_documents(documents)
    add_to_chroma(chunks)



def split_documents(documents: list[str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.create_documents(documents)
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Сохраненных фрагментов: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Добавление новых фрагментов: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("Новых фрагментов не обнаружено")


def calculate_chunk_ids(chunks):
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        chunk_id = f"{source}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        current_chunk_index += 1

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


