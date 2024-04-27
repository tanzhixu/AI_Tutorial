# 在前两篇文章里，我们已经介绍了如何在本地运行Ollama以及如何通过提供外部数据库的方式微调模型的答案。本篇文章将继续探索如何使用“函数调用(function-calling)”功能以扩展模型能力，使其在“智能”的道路上越走越远。

# function-calling介绍
# 根据OpenAI官方文档，function-calling是使得大型语言模型具备可以连接到外部工具的能力。简而言之，开发者事先给模型提供了若干工具（函数），在模型理解用户的问题后，自行判断是否需要调用工具以获得更多上下文信息，帮助模型更好的决策。

# 举个例子：在上一篇文章我们是利用Document loaders将事先准备好的文本作为上下文提供给模型，而使用function-calling以后，我们只要提供一个“搜索函数”作为工具，模型即可自己通过搜索引擎进行搜索然后得出答案。

# 得益于最新的模型训练，现在的模型既能够检测何时应调用函数（取决于输入），还能够以比以前的模型更贴近函数签名的方式响应 JSON。

# 用途
# function-calling有什么用？官网给出3个例子：

# 创建通过调用外部 API 回答问题的助手
# 将自然语言转换为 API 调用
# 从文本中提取结构化数据
# 下面我们就一步一步来理解一下function-calling。
import inspect
import json
import requests
from typing import get_type_hints


def generate_full_completion(model: str, prompt: str) -> dict[str, str]:
    params = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(params),
        timeout=60,
    )
    return json.loads(response.text)


# 首先定义4个function，这4个function有不同的使用场景。
def get_gas_prices(city: str) -> float:
    """Get gas prices for specified cities."""
    print(f'Get gas prices for {city}.')
def github(project: str) -> str:
    '''Get the infomations such as author from github with the project name.'''
    print(f'Access the github for {project}.')
def get_weather(city: str) -> str:
    """Get the current weather given a city."""
    print(f'Getting weather for {city}.')
def get_directions(start: str, destination: str) -> float:
    """Get directions from Google Directions API.
    start: start address as a string including zipcode (if any)
    destination: end address as a string including zipcode (if any)"""
    print(f'Get directions for {start} {destination}.')


def get_type_name(t):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def function_to_json(func):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {"type": "object", "properties": {}},
        "returns": type_hints.get("return", "void").__name__,
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def main():
    # 定义Prompts
    functions_prompt = f"""
You have access to the following tools:
{function_to_json(get_weather)}
{function_to_json(get_gas_prices)}
{function_to_json(get_directions)}
{function_to_json(github)}

You must follow these instructions:
Always select one or more of the above tools based on the user query
If a tool is found, you must respond in the JSON format matching the following schema:
{{
   "tools": {{
        "tool": "<name of the selected tool>",
        "tool_input": <parameters for the selected tool, matching the tool's JSON schema
   }}
}}
If there are multiple tools required, make sure a list of tools are returned in a JSON array.
If there is no tool that match the user request, you must respond empty JSON {{}}.

User Query:
    """

    GPT_MODEL = "mistral:7b-instruct-v0.2-q8_0"

    prompts = [
        "What's the weather like in Beijing?",
        "What is the distance from Shanghai to Hangzhou and how much do I need to fill up the gas in advance to drive from Shanghai to Hangzhou?",
        "Who's the author of the 'snake-game' on github?",
        "What is the exchange rate between US dollar and Japanese yen?",
    ]

    for prompt in prompts:
        print(f"{prompt}")
        question = functions_prompt + prompt
        response = generate_full_completion(GPT_MODEL, question)
        try:
            data = json.loads(response.get("response", response))
            # print(data)
            for tool_data in data["tools"]:
                execute_fuc(tool_data)
        except Exception:
            print('No tools found.')
        print(f"Total duration: {int(response.get('total_duration')) / 1e9} seconds")


def execute_fuc(tool_data):
    func_name = tool_data["tool"]
    func_input = tool_data["tool_input"]

    # 获取全局命名空间中的函数对象
    func = globals().get(func_name)

    if func is not None and callable(func):
        # 如果找到了函数并且是可调用的，调用它
        func(**func_input)
    else:
        print(f"Unknown function: {func_name}")


if __name__ == "__main__":
    main()
