# import argparse
import pprint
import time
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from data_base_lib.get_embedding_function import get_embedding_function

import math
from langchain_core.globals import set_llm_cache
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

from langchain_core.caches import InMemoryCache

store = InMemoryByteStore()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос ссылаясь исключительно на следующий запрос:

{context}

Ответь на вопрос ссылаясь только на запрос приведенный ранее: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    start_time = time.time()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print("\n--- DB initiated in %s seconds ---" % (time.time() - start_time))
    # Search the DB.
    start_time = time.time()
    res_num = max(math.ceil(0.01 * len(db.get()['ids'])), 6)
    results = db.similarity_search_with_score(query_text, k=res_num)  # MAKE IT k * DB SIZE
    print("\n--- results found in %s seconds ---" % (time.time() - start_time))
    print(results)
    context_list, acc_score = [], 0
    sorted_data = sorted(results, reverse=True, key=lambda item: item[1])
    pprint.pprint(sorted_data)
    third_index = len(sorted_data) // 3
    sorted_data = sorted_data[:third_index]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in sorted_data])
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(len(context_text.split()))
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model="llama3.1")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\nИсточники: {sources}"
    print(formatted_response)
    return response_text, sources
