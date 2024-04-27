### 介绍
LangChain是一个用于开发由大型语言模型（LLM）提供支持的应用程序的框架。

LangChain简化了LLM应用程序生命周期的每个阶段：

- 开发：使用LangChain的开源构建块和组件构建您的应用程序。使用第三方集成和模板开始运行。
- 产品化：使用LangSmith检查、监控和评估您的链，让您可以放心地持续优化和部署。
- 部署：使用LangServe将任何链变成API。

具体来说，该框架由以下开源库组成：

- langchain-core：基础抽象和LangChain表达式语言。
- langchain-Community：第三方集成。合作伙伴包（例如langchain-openai、langchain-不人道等）：一些集成已进一步拆分为它们自己的轻量级包，这些包仅依赖于langchain-core。
- langchain：构成应用程序认知架构的链、代理和检索策略。
- langgraph：通过将步骤建模为图中的边和节点，使用LLM构建健壮且有状态的多参与者应用程序。
- langserve：部署为REST API。

### 安装
#### PIP
```
pip install langchain
```
#### Conda
```
conda install langchain -c conda-forge
```

### using Ollama
Ollama允许您在本地运行开源大型语言模型，例如Llama 2，需提前安装

#### 示例
使用的是llama2-chinese大模型
```
from langchain_community.llms import Ollama
llm = Ollama(model="llama2-chinese")
```

一旦您安装并初始化了您选择的LLM，我们可以尝试使用它
```
llm.invoke("如何学习使用langchain?")
```
输出
```
1. 了解 langchain 的基本概念和用法，了解它是一个从字符串中提取关键词的自然语言处理工具。\n2. 使用 langchain 查看示例代码，了解如何在 Python 中使用 Langchain。\n3. 查看 Langchain 官方文档，了解所有关键词和其他功能。\n4. 通过对 langchain 进行练习，学会如何使用它来处理自然语言数据集。\n5. 为了更好地理解 Langchain，可以使用几种方法，例如：\n\t- 查看示例代码。\n\t- 使用 langchain 进行搜索。\n\t- 通过对 langchain 进行练习来学会如何使用 Langchain。\n6. 在 Langchain 官方文档上可以看到示例代码，这将帮助你了解如何使用它。\n7. 通过对 langchain 进行练习来学会如何使用 Langchain。\n8. 使用 langchain 查看示例代码，这将帮助你更好地理解 Langchain。\n9. 可以通过对langchain 进行练习来学会如何使用 Langchain。\n10. 在 Langchain 官方文档上，查看示例代码，这将帮助你更好地理解 Langchain。\n'
```

同时，我们也可以使用提示词模板来进行指导响应，提示模板将原始用户输入转换为对LLM的更好输入
```
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
```

我们现在可以将这些组合成一个简单的LLM链：
```
chain = prompt | llm 
```

一旦我们有了链，我们就可以使用它了
```
chain.invoke({"input": "how can langsmith help with testing?"})
```
输出：
```
1. 阅读LangChain的官方文档，了解它的功能和用法。\n2. 试着使用LangChain来创建一个简单的项目，看看如果需要支持多语言、多模块等特性。\n3. 查看LangChain的社区资源，比如Github上的issue和pull request，了解用户对其中的问题和改进建议。\n4. 与相关人员共享您的想法和经验，并获取他们的反馈和建议。\n5. 参加LangChain社区活动，与其他用户交流，了解他们的经验和培养。\n6. 根据您的需求和使用情况进行调整和优化。\n\n以上是参考的答案，希望能帮到您。'
```

添加一个简单的输出解析器来将聊天消息转换为字符串。
```
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
```

将其添加到之前的链中：
```
chain = prompt | llm | output_parser
```

我们现在可以调用它并问同样的问题。答案现在将是一个字符串（而不是ChatMessage）
```
chain.invoke({"input": "how can langsmith help with testing?"})
```

输出
```
1. 学习langchain的基本概念和使用方法，可以从官方文档、社区网站或者相关课程中获取。\n2. 了解语言处理任务分类，并选择最适合自己的翻译任务类型。\n3. 使用langchain编写一个简单的翻译任务，并调用相关的api进行处理和分析。\n4. 根据语言模型的性能和准确度进行选择和调整，以便为不同语言提供更好的翻译结果。\n5. 使用langchain的相关技术和工具来实现高效、可靠的翻译任务，如文本处理、机器学习等。\n6. 对于更复杂或特定的翻译任务，可以使用langchain的分割翻译功能进行逐个文本处理和生成，从而提高翻译效率。\n7. 掌握langchain的开发环境和编程语言，并根据自身需求或问题进行调试、改进等工作。\n\n总之，使用langchain更好的方法是通过学习相关技能和知识，加上实际应用和调试来提升翻译效果。同时，建立一个包含多种语言处理任务类型的翻译任务库也可以提高langchain的使用效率。
```