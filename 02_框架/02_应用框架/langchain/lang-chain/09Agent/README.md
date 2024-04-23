### Agent介绍
LangChain Agent是一个基于LangChain的对话代理，它能够根据用户输入的问题，生成相应的回答。Agent使用LangChain的文档理解模块和语言模型模块来完成对话任务。

### 关键组件
- LangChain文档理解模块：用于提取问题和回答中的关键信息。
- LangChain语言模型模块：用于生成回答。
- LangChain对话代理：将文档理解和语言模型模块结合起来，生成回答。
- LangChain工具：用于提供问题和回答中的关键信息。
- LangChainLLM：用于生成回答。
- LangChainAgent：用于生成回答。
- LangChainAgentType：用于指定Agent的类型。
- LangChainAgentExecutor：用于执行Agent的任务。
- LangChainAgentOutputParser：用于解析Agent的输出。
- LangChainAgentTool：用于提供Agent的工具。


### 安装
```
pip install langchain-agent
```

### 使用    
```
from langchain.agents import load_tools 
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))

# 初始化代理
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 对话
agent.run("What is the capital of France?")
```

### 参考 
https://python.langchain.com/en/latest/modules/agents.html 
https://python.langchain.com/en/latest/modules/agents.html#zero-shot-react-description
https://python.langchain.com/en/latest/modules/agents.html#agent-types
