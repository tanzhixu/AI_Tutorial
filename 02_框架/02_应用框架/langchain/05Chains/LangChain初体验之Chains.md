### 简介
LangChain 中的链由多个组件组成，每个组件都实现了 Chain 接口。这些组件可以包括 LLM、工具、数据处理器等。

#### 导入Ollama
```
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="qwen:7b")
prompt = PromptTemplate(
    input_variables=["color"],
    template="What is the hex code of color {color}?",
)
```
#### 创建链

```
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

#### 调用链
```
print(chain.invoke("红色")) 
```

#### 输出
```
{'color': '红色', 'text': 'The hex code for color red (中国文化中的红色，通常指的是暖色调的深红) is typically represented as #FF0000. This is a standard web color code, but it refers to pure red without any saturation or shade variations.\n'}
```
