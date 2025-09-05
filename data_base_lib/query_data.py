import pprint, math, importlib
from typing import List

from funcs import global_vars
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from data_base_lib.get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос ссылаясь исключительно на следующие данные:

"{context}"

Ответь на вопрос ссылаясь только на данные приведенные ранее: "{question}."
"""


def create_hybrid_retriever(query: str, vector_retriever=None, bm25_retriever=None) -> List[Document]:
    # Retrieve documents and scores from retriever1
    docs1, scores1 = zip(*vector_retriever.similarity_search_with_score(query, k=math.ceil(0.05*len(vector_retriever.get()['ids']))))
    for doc, score in zip(docs1, scores1):
        doc.metadata["score"] = score

    # Retrieve documents and scores from bm25_retriever
    bm25_retriever.k = math.ceil(0.05*len(vector_retriever.get()))
    docs2, scores2 = zip(*bm25_retriever.invoke(query))
    # for doc, score in zip(docs2, scores2):
    #     doc.metadata["score"] = score

    # Combine results from both retrievers
    combined_docs = list(docs1) + list(docs2)

    # Filter results by score threshold
    filtered_docs = sorted([doc for doc in combined_docs], reverse=True) # if doc.metadata["score"] >= score_threshold]

    return filtered_docs


def query_rag(query_text: str):
    import psutil

    importlib.reload(global_vars)
    cpu_count_physical = psutil.cpu_count(logical=False)
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    chroma_retriever = db.as_retriever()

    import pickle

    def load_object(filename):
        """Loads an object from a file using pickle."""
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    # Load the saved retriever
    bm25_retriever = load_object('appdata/retriever.pkl')
    results = create_hybrid_retriever(query_text, bm25_retriever=bm25_retriever, vector_retriever=chroma_retriever)

    # results = db.similarity_search_with_score(query_text, k=math.ceil(0.05*len(db.get()['ids']))) # MAKE IT k * DB SIZE
    pprint.pprint(results)
    response_text = ''
    if global_vars.use_llm:
        context_text = "\n---\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        model = OllamaLLM(model="llama3.1", top_k=30, num_thread=cpu_count_physical - 1)
        response_text = model.invoke(prompt)

    sources = reversed([{'name': doc.metadata.get("id", None)[0:doc.metadata.get("id", None).rfind(':')], 'content': db.get(doc.metadata['id'])['documents'][0]} for doc, _score in results])
    formatted_response = f"{response_text}\nИсточники: {sources}"
    print(formatted_response)
    return response_text, sources


def _query_rag(query_text: str):
    import pickle

    def load_object(filename):
        """Loads an object from a file using pickle."""
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    # Load the saved retriever
    retriever = load_object('appdata/retriever.pkl')
    results = retriever.invoke(query_text)
    import psutil
    cpu_count_physical = psutil.cpu_count(logical=False)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.1", num_thread=cpu_count_physical - 1, top_k=30)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"{response_text}\nИсточники: {sources}"

    print(f"--- Запрос {query_text} ---\n--- Вывод БД {prompt}\n--- Ответ {formatted_response} ---")

    return response_text, sources


def query_rag_2(query_text: str):

    import pickle

    def load_object(filename):
        """Loads an object from a file using pickle."""
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    # Load the saved retriever
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    chroma_retriever = db.as_retriever()
    print(type(chroma_retriever))
    bm25_retriever = load_object('appdata/retriever.pkl')  # retriever 1
    print(type(bm25_retriever))
    ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retriever,
                                                       bm25_retriever],
                                           weights=[0.5, 0.5])

    ensemble_results = ensemble_retriever.invoke(query_text)
