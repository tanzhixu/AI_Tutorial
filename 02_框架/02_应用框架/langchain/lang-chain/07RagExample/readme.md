### 基于
以下用一个完整的例子来完成一个具备功能集的LLM应用，该应用基于langchain库。通过加载本地pdf文档的形式，实现提供给用户基于该文档的问答能力

#### 1. 安装ollama及langchain
```
pip install -y langchain chromadb pymupdf
```

#### 2. 下载文档
```
PDF_NAME = 'C:\\Users\\Administrator\\Desktop\\OpsAny.pdf'
```

#### 3. 加载文档
```
from langchain.document_loaders import PyMuPDFLoader
docs = PyMuPDFLoader(PDF_NAME).load()
```

#### 4. 拆分文档并存储文本嵌入的向量数据
```
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model='nomic-embed-text')  # 文本嵌入模型
vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name="serverless_guide")
```

#### 5. 基于ollama创建QA链
```
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
llm = Ollama(model="qwen:7b")
chain = load_qa_chain(llm, chain_type="stuff")
```

#### 6. 基于提问，进行相似性查询
```
query = "opsany如何用docker部署?"
similar_docs = vectorstore.similarity_search(query)
```

#### 7. 基于相似性查询结果，进行回答
```
chain.run(input_documents=similar_docs, question=query)
```