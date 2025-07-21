from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function


db = Chroma(
    persist_directory=r'C:\Users\Тимофей\Documents\neogrid_site\neogrid_site\chroma',
    embedding_function=get_embedding_function()
)


print(len(db.get()['ids']))