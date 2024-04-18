### 提示词（Prompts）
语言模型的提示是用户提供的一组指令或输入，用于指导模型的响应，帮助它理解上下文并生成相关且连贯的基于语言的输出，例如回答问题、完成句子或参与对话。

#### 提示词模板（PromptTemplate）
使用PromptTemplate为字符串提示符创建模板。
默认情况下，PromptTemplate使用Python的str. format语法进行模板化
```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")
```
该模板支持任意数量的变量，包括没有变量：
```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
prompt_template.format()
```

#### 聊天提示词模板（ChatPromptTemplate）
聊天模型/的提示是聊天消息列表。

每个聊天消息都与内容和一个称为角色的附加参数相关联。例如，在OpenAI Chat Completions API中，聊天消息可以与AI助手、人类或系统角色相关联。

创建一个聊天提示模板，如下所示
```
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
print(messages)
[SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name?')]
```

看一个完全的例子

```
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)
human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)
chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()
```
输出如下
```
[HumanMessage(content='What is the best way to learn programming?'),
 AIMessage(content='1. Choose a programming language: Decide on a programming language that you want to learn.\n\n2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.\n\n3. Practice, practice, practice: The best way to learn programming is through hands-on experience'),
 HumanMessage(content='Summarize our conversation so far in 10 words.')]
```

消息提示模板类型的完整列表包括：

- AIMessagePromptTemplate， 对于AI助手消息；
- SystemMessagePromptTemplate， 对于系统消息；
- HumanMessagePromptTemplate， 对于用户消息；
- ChatMessagePromptTemplate， 对于具有任意角色的消息；

### 样本选择器
在LLM应用开发中，可能需要从大量样本数据中，选择部分数据包含在提示词中。样本选择器（Example Selector）正是满足该需求的组件，它也通常与少样本提示词配合使用。LangChain 提供了样本选择器的基础接口类 BaseExampleSelector，每个选择器类必须实现的函数为 select_examples。LangChain 实现了若干基于不用应用场景或算法的选择器：

- LengthBasedExampleSelector(根据在一定长度内可以容纳的数量选择示例)
- MaxMarginalRelevanceExampleSelector(使用输入和示例之间的最大边际相关性来决定选择哪些示例)
- NGramOverlapExampleSelector(使用输入和示例之间的ngram重叠来决定选择哪些示例)
- SemanticSimilarityExampleSelector(使用输入和示例之间的语义相似性来决定选择哪些示例)

如果您有大量示例，您可能需要选择要包含在提示中的示例。示例选择器是负责执行此操作的类。
基本接口定义如下：
```
class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        
    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
```
它需要定义的唯一方法是一个select_examples方法。它接受输入变量，然后返回一个示例列表。如何选择这些示例取决于每个特定的实现。

### 自定义样本选择器
```
examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivaderci"},
    {"input": "soccer", "output": "calcio"},
]

from langchain_core.example_selectors.base import BaseExampleSelector

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # This assumes knowledge that part of the input will be a 'text' key
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        # Initialize variables to store the best match and its length difference
        best_match = None
        smallest_diff = float("inf")

        # Iterate through each example
        for example in self.examples:
            # Calculate the length difference with the first word of the example
            current_diff = abs(len(example["input"]) - new_word_length)

            # Update the best match if the current one is closer in length
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]
example_selector = CustomExampleSelector(examples)
example_selector.select_examples({"input": "okay"})
```
输出：
```
[{'input': 'bye', 'output': 'arrivaderci'}]
```

添加示例example，并重新进行选择
```
example_selector.add_example({"input": "hand", "output": "mano"})
example_selector.select_examples({"input": "okay"})
```

输出：
```
[{'input': 'hand', 'output': 'mano'}]
```

#### 使用提示词
```
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italain:",
    input_variables=["input"],
)

print(prompt.format(input="word"))
```

输出如下
```
Translate the following words from English to Italain:

Input: hand -> Output: mano

Input: word -> Output:
```