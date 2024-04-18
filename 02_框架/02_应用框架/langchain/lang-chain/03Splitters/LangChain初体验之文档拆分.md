### 文档拆分器
加载文档后，您通常希望对其进行转换以更好地适应您的应用程序。最简单的例子是，您可能希望将长文档拆分为适合模型上下文窗口的较小块。LangChain有许多内置的文档转换器，可以轻松拆分、组合、过滤和以其他方式操作文档。

当您想处理长文本时，有必要将该文本拆分为块。尽管这听起来很简单，但这里有很多潜在的复杂性。理想情况下，您希望将语义相关的文本片段放在一起。“语义相关”的含义可能取决于文本的类型。本笔记本展示了几种方法来做到这一点。

在高级别上，文本拆分器的工作方式如下：

1. 将文本拆分为语义上有意义的小块（通常是句子）。
2. 开始将这些小块组合成一个更大的块，直到您达到一定的大小（由某些函数测量）。
3. 一旦你达到了这个大小，让这个块成为自己的文本块，然后开始创建一个有一些重叠的新文本块（以保持块之间的上下文）。

这意味着有两个不同的轴，您可以沿着它们自定义文本拆分器：

- 文本是如何拆分的
- 如何测量块大小

#### 按字符拆分
这是最简单的方法。这基于字符（默认为“”）进行拆分，并通过字符数测量块长度。

1. 文本如何拆分：按单个字符。
2. 如何测量块大小：通过字符数。

```
from langchain_text_splitters import CharacterTextSplitter

with open("text_spliter.txt", "r") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
```

#### 按代码拆分
CodeTextSplitter允许使用支持的多种语言拆分代码。导入枚举语言并指定语言
```
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
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
python_docs
```

#### 按Marddown拆分
```
markdown_text = """
### LangChain
LangChain is an open-source project that makes it easy to build applications with LLMs.
#### Quick Install

# Hopefully this code block isn't split
pip install langchain

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text])
md_docs
```

> https://gitee.com/AIOldTan/lang-chain.git