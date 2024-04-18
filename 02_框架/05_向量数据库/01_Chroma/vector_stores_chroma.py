# pip install chromadb
# pip install langchain
# pip install langchain-community

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. 读取文件并分词
raw_documents = TextLoader("C:\\Users\\Administrator\\Desktop\\三国演义.txt",encoding='utf-8').load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

embeddings = OllamaEmbeddings(model='nomic-embed-text') # 文本嵌入模型
db = Chroma.from_documents(documents, embeddings)

query = "身长七尺，细眼长髯是谁"
docs = db.similarity_search(query)
print(docs[0].page_content)


embedding_vector = OllamaEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)