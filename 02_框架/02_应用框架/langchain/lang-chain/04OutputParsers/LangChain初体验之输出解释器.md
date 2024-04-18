### 输出解释器
输出解析器负责获取LLM的输出并将其转换为更合适的格式。当您使用LLM生成任何形式的结构化数据时，这非常有用。

除了拥有大量不同类型的输出解析器之外，LangChain OutputParser的一个显着优势是它们中的许多都支持流
#### 解释器类型
- List parser
- Datetime parser
- Enum parser
- JSON parser
- Pydantic parser
- Retry parser
- Structured output parser
- XML parser
- YAML parser
##### List parser
当您想要返回逗号分隔项的列表时，可以使用此输出解析器

```
from langchain_community.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

llm = Ollama(model="qwen:7b")
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser
chain.invoke({"subject": "输出5种颜色"})
```

输出
```
['Sure',
 "here's a list of 5 colors:\n\n```\nred",
 'blue',
 'green',
 'yellow',
 'orange\n``` \n\nEach color is separated by a comma for clarity in the list.']
```

##### Json parser
此输出解析器允许用户指定任意JSON模式并查询符合该模式的输出的LLM。

```
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import Ollama

llm = Ollama(model="qwen:7b")

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

joke_query = ""给我讲个笑话""

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

chain.invoke({"query": joke_query}) 
```

输出
```
{'setup': '你有没有遇到过完全理解不了的数学题？',
 'punchline': '嗯，比如最近的一道‘两对父子’的家庭成员关系题...我至今还是没想明白呢！哈哈哈！'}
```

##### YAML parser
此输出解析器允许用户指定任意模式并查询符合该模式的输出的LLM，使用YAML格式化他们的响应
```
from langchain.prompts import PromptTemplate
from langchain.output_parsers import YamlOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import Ollama

llm = Ollama(model="qwen:7b")

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

joke_query = "给我讲个笑话"

parser = YamlOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

chain.invoke({"query": joke_query}) 
```

输出
```
Joke(setup='为什么熊猫总是抱着竹子？', punchline='因为它们不会‘剥皮’啊！')
```