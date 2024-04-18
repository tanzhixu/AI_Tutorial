
### 数据加载
使用文档加载器从源加载数据作为文档。文档是一段文本和相关的元数据。例如，有文档加载器用于加载简单的. txt文件、加载任何网页的文本内容，甚至用于加载YouTube视频的抄本。

文档加载器提供了一种"加载"方法，用于将数据从配置的源加载为文档。它们可以选择实现“延迟加载”以及将数据延迟加载到内存中

#### 加载txt文档
```
from langchain_community.document_loaders import TextLoader
raw_documents = TextLoader("C:\\Users\\Administrator\\Desktop\\三国演义.txt",encoding='utf-8').load()
print(raw_documents)
```
输出
```
[Document(page_content='第一回\u3000宴桃园豪杰三结义\u3000斩黄巾英雄首立功\n\n话说天下大势，分久必合，合久必分。周末七国分争，并入于秦；及秦灭之后，楚汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。推其致乱之由，始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，大将军窦武、太傅陈蕃共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，机事不密，反为所害：中涓自此愈横。\n\n建宁二年四月望日，帝御温德殿。方升座，殿角狂风骤起，只见一条大青蛇从梁上飞将下来，蟠于椅上。帝惊倒，左右急救入宫，百官俱奔避。须臾，蛇不见了。忽然大雷大雨，加以冰雹，落到半夜方止，坏却房屋无数。建宁四年二月，洛阳地震；又海水泛溢，沿海居民尽被大浪卷入海中。光和元年，雌鸡化雄。六月朔，黑气十余丈，飞入温德殿中。秋七月，有虹见于玉堂，五原山岸尽皆崩裂。种种不祥，非止一端。帝下诏问群臣以灾异之由。议郎蔡邕上疏，以为霓堕鸡化，乃妇寺干政之所致，言颇切直。帝览奏叹息，因起更衣。曹节在后窃视，悉宣告左右，遂以他事陷邕于罪，放归田里。后张让、赵忠、封谞、段珪、曹节、侯览、蹇硕、程旷、夏恽、郭胜十人朋比为奸，号为“十常侍”。帝尊信张让，呼为阿父。朝政日非，以致天下人心思乱，盗贼蜂起。', metadata={'source': 'C:\\Users\\Administrator\\Desktop\\三国演义.txt'})]
```

### 数据连接的几个步骤

1. 加载数据：从不同的数据源加载文档
2. 数据转换：拆分文档，将文档转换为问答格式，去除冗余文档，等等
3. 嵌入模型：将非结构化文本转换为浮点数数组表现形式，也称为向量
4. 向量存储：存储和搜索嵌入数据（向量）
5. 检索：提供数据查询的通用接口

### 演示
下面以一个完成的例子（web加载）来演示数据如何加载数据、数据如何转换、嵌入模型、向量存储、检索等

#### 加载网页数据
##### 导入模型
```
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
```
##### 加载网页数据
```
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
```
##### 数据转换
```
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
```
##### 嵌入模型
```
embeddings = OllamaEmbeddings(model='nomic-embed-text')
```
##### 向量存储
```
vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()
```
##### 检索
```
retriever.get_relevant_documents("how to upload a dataset")[0]
```

#### 完整例子
本例子通过加载web数据的方式，展示数据如何加载数据、数据如何转换、嵌入模型、向量存储、检索等
```
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

retriever.get_relevant_documents("how to upload a dataset")[0]
```
输出
```
Document(page_content="issues, please reach out to us at support@langchain.dev.My team deals with sensitive data that cannot be logged. How can I ensure that only my team can access it?â€‹If you are interested in a private deployment of LangSmith or if you need to self-host, please reach out to us at sales@langchain.dev. Self-hosting LangSmith requires an annual enterprise license that also comes with support and formalized access to the LangChain team.Was this page helpful?NextUser GuideIntroductionInstall LangSmithCreate an API keySetup your environmentLog your first traceCreate your first evaluationNext StepsAdditional ResourcesFAQHow do I migrate projects between organizations?Why aren't my runs aren't showing up in my project?My team deals with sensitive data that cannot be logged. How can I ensure that only my team can access it?CommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright Â© 2024 LangChain, Inc.", metadata={'description': 'Introduction', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'Getting started with LangSmith | ğŸ¦œï¸�ğŸ›\xa0ï¸� LangSmith'})
```