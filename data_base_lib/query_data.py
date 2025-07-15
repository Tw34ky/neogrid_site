# import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос ссылаясь исключительно на следующий контекст:

{context}

---

Ответь на вопрос ссылаясь только на контекст приведенный ранее: {question}
"""
PROMPT_TEMPLATE_INJECT = '''
Объедини информацию из отрывка документа (document) и новую информацию от пользователя (info) так, чтобы отрывок можно было встроить обратно.
info:
---
{info}
---
document:
--- 
{document}
---
Выведи итоговый текст без пояснений, потерь информации. Вставь как можно больше информации из info.
'''



def main():
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    query_text = input()
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5) # make k depend on data

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.1")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\nИсточники: {sources}"
    print(formatted_response)
    return response_text


def query_rag_inject(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=1)  # make k depend on data
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_INJECT)
    # print(results[0][0].page_content)
    prompt = prompt_template.format(document=results[0][0].page_content, info=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.1")
    response_text = model.invoke(prompt)
    print(response_text)
    return response_text


if __name__ == "__main__":
    main()
