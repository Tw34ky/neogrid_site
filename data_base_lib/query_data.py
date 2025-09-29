import pprint, math, importlib
import time, torch
from typing import List

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# from langchain_ollama import OllamaLLM

from data_base_lib.get_embedding_function import get_embedding_function
from funcs import global_vars

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос ссылаясь исключительно на следующие данные:

"{context}"

Ответь на вопрос ссылаясь только на данные приведенные ранее: "{question}."
"""


# def create_hybrid_retriever(query: str, vector_retriever=None, bm25_retriever=None) -> List[Document]:
#     # Retrieve documents and scores from retriever1
#     docs1, scores1 = zip(*vector_retriever.similarity_search_with_score(query, k=math.ceil(0.05*len(vector_retriever.get()['ids']))))
#     for doc, score in zip(docs1, scores1):
#         doc.metadata["score"] = score
#
#     # Retrieve documents and scores from bm25_retriever
#     bm25_retriever.k = math.ceil(0.05*len(vector_retriever.get()))
#     docs2, scores2 = zip(*bm25_retriever.invoke(query))
#     # for doc, score in zip(docs2, scores2):
#     #     doc.metadata["score"] = score
#
#     # Combine results from both retrievers
#     combined_docs = list(docs1) + list(docs2)
#
#     # Filter results by score threshold
#     filtered_docs = sorted([doc for doc in combined_docs], reverse=True) # if doc.metadata["score"] >= score_threshold]
#
#     return filtered_docs


def query_rag(query_text: str):
    # import psutil
    # cpu_count_physical = psutil.cpu_count(logical=False) # Получим кол-во доступных ядер для эффективного запуска модели через langchain (не используется с huggingface pipeline'ом)
    importlib.reload(global_vars)

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # chroma_retriever = db.as_retriever()
    #
    # import pickle
    #
    # def load_object(filename):
    #     """Loads an object from a file using pickle."""
    #     with open(filename, 'rb') as inp:
    #         return pickle.load(inp)
    #
    # # Load the saved retriever
    # bm25_retriever = load_object('appdata/retriever.pkl')
    # results = create_hybrid_retriever(query_text, bm25_retriever=bm25_retriever, vector_retriever=chroma_retriever) # недоделанный код для использования двух баз данных одновременно

    """
    
    Воспользуемся similarity_search_with_score чтобы найти самые релевантные к запросу чанки из базы данных 
    
    """

    results = db.similarity_search_with_score(query_text, k=math.ceil(0.05*len(db.get()['ids']))) # MAKE IT k * DB SIZE
    pprint.pprint(results)

    """
    
    Воспользуемся моделью из семейства Qwen для очистки ненужных чанков, тем самым впоследствии оптимизируя работу LLM 
    
    """
    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                         query=query, doc=doc)
        return output


    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    current_time = time.time()

    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 1024
    print(f'--- Reranker intialized in {round(time.time() - current_time, 3)} seconds ---')

    task = 'Given a database search query, retrieve relevant passages that answer the query'

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    pairs = [format_instruction(task, query_text, doc) for doc in results]
    current_time = time.time()

    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)

    print(f'--- Documents reranked in {round(time.time() - current_time, 3)} seconds ---')
    print("scores: ", scores)
    avg_score = sum(scores) / len(scores)
    results_copy = []
    for doc, score in results:
        if score > avg_score:
            results_copy.append((doc, scores[results.index((doc, score))]))


    response_text = ''
    if global_vars.use_llm:
        context_text = "\n---\n".join([doc.page_content for doc, _score in results_copy])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        model = HuggingFacePipeline.from_model_id(
                model_id="Qwen/Qwen3-4B-Instruct-2507",
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 4096,
                    "top_k": 30,
                    "temperature": 0.6,
                },
            )
        response_text = model.invoke(prompt)

    sources = reversed([{'name': doc.metadata.get("id", None)[0:doc.metadata.get("id", None).rfind(':')], 'content': db.get(doc.metadata['id'])['documents'][0]} for doc, _score in results])
    formatted_response = f"{response_text}\nИсточники: {sources}"
    print(formatted_response)
    return response_text, sources


# def _query_rag(query_text: str):
#     """
#
#     Альтернативный метод получения данных из векторной БД (через BM25)
#
#     :param query_text:
#     :return:
#     """
#
#
#     import pickle
#
#     def load_object(filename):
#         """Loads an object from a file using pickle."""
#         with open(filename, 'rb') as inp:
#             return pickle.load(inp)
#
#     # Load the saved retriever
#     retriever = load_object('appdata/retriever.pkl')
#     results = retriever.invoke(query_text)
#     import psutil
#     cpu_count_physical = psutil.cpu_count(logical=False)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#
#     # model = OllamaLLM(model="llama3.1", num_thread=cpu_count_physical - 1, top_k=30)
#
#     response_text = model.invoke(prompt)
#
#     sources = [doc.metadata.get("id", None) for doc in results]
#     formatted_response = f"{response_text}\nИсточники: {sources}"
#
#     print(f"--- Запрос {query_text} ---\n--- Вывод БД {prompt}\n--- Ответ {formatted_response} ---")
#
#     return response_text, sources
#
#
# def query_rag_2(query_text: str):
#
#     import pickle
#
#     def load_object(filename):
#         """Loads an object from a file using pickle."""
#         with open(filename, 'rb') as inp:
#             return pickle.load(inp)
#
#     # Load the saved retriever
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
#     chroma_retriever = db.as_retriever()
#     print(type(chroma_retriever))
#     bm25_retriever = load_object('appdata/retriever.pkl')  # retriever 1
#     print(type(bm25_retriever))
#     ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retriever,
#                                                        bm25_retriever],
#                                            weights=[0.5, 0.5])
#
#     ensemble_results = ensemble_retriever.invoke(query_text)
