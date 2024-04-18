import os
current_directory = os.getcwd()

'''
    按字符拆分
'''
from langchain_text_splitters import CharacterTextSplitter
with open(os.path.join(current_directory, 'txt_spliter.txt'), "r", encoding="utf8") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
# print(texts)
# [Document(page_content='加载文档后，您通常希望对其进行转换以更好地适应您的应用程序。最简单的例子是，您可能希望将长文档拆分为适合模型上下文窗口的较小块。LangChain有许多内置
# 的文档转换器，可以轻松拆分、组合、过滤和以其他方式操作文档。\n\n当您想处理长文本时，有必要将该文本拆分为块。尽管这听起来很简单，但这里有很多潜在的复杂性。理想情况下，您希 
# 望将语义相关的文本片段放在一起。“语义相关”的含义可能取决于文本的类型。本笔记本展示了几种方法来做到这一点。\n\n在高级别上，文本拆分器的工作方式如下：\n\n1. 将文本拆分为语义
# 上有意义的小块（通常是句子）。\n2. 开始将这些小块组合成一个更大的块，直到您达到一定的大小（由某些函数测量）。\n3. 一旦你达到了这个大小，让这个块成为自己的文本块，然后开始 
# 创建一个有一些重叠的新文本块（以保持块之间的上下文）。\n\n这意味着有两个不同的轴，您可以沿着它们自定义文本拆分器：\n\n- 文本是如何拆分的\n- 如何测量块大小')]

'''
    按代码拆分
'''
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
language_list = [e.value for e in Language]
# print(language_list)
# ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol', 'c', 
# 'lua', 'perl']
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
# print(python_docs)
# [Document(page_content='def hello_world():\n    print("Hello, World!")'), Document(page_content='# Call the function\nhello_world()')]

'''
    按Markdown拆分
'''
markdown_text = """
### LangChain
LangChain is an open-source project that makes it easy to build applications with LLMs.
#### Quick Install

```bash
# Hopefully this code block isn't split
pip install langchain
```

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text])
md_docs
# print(md_docs)
# [Document(page_content='### LangChain'), Document(page_content='LangChain is an open-source project that makes it easy to'), Document(page_content='build applications with LLMs.'), Document(page_content='#### Quick Install\n\n```bash'), Document(page_content="# Hopefully this code block isn't split"), Document(page_content='pip install langchain'), Document(page_content='```'), Document(page_content='As an open-source project in a rapidly developing field, we'), Document(page_content='are extremely open to contributions.')]