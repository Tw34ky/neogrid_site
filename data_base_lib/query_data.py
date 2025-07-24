# import argparse
import pprint

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from data_base_lib.get_embedding_function import get_embedding_function
import warnings

import math

# warnings.filterwarnings("ignore", category=DeprecationWarning)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос ссылаясь исключительно на следующий запрос:

{context}

---

Ответь на вопрос ссылаясь только на запрос приведенный ранее: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=math.ceil(0.01*len(db.get()['ids']))) # MAKE IT k * DB SIZE

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.1")

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\nИсточники: {sources}"
    print(formatted_response)
    pprint.pprint(sources)
    return response_text
