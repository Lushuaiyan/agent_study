import json

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)



# 创建工具
def get_weather(city:str):
    return f"{city}今天的天气是晴天"
# 这里为了简化没有真实的实现

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取目标城市的当天天气",
            "parameters": {
                "type": "object",
                "properties":{
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 定义用户信息
messages = [
    {"role": "user", "content": "北京今天天气如何"}
]


response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # 让模型自己判断是否调用工具, none是不调用, required是必须调用
)

# 获取模型的返回结果
assistant_message = response.choices[0].message
messages.append(assistant_message)

# 定义函数映射表
function_map = {
    "get_weather": get_weather,
}


# 检查模型是否需要调用工具
if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # 从函数映射表中获取函数并调用
        func = function_map.get(function_name)
        if func:
            result = func(**arguments)
        else:
            result = f"未知工具: {function_name}"

        # 返回工具结果
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            }
        )

    # 将工具结果发给模型, 获取最终结果
    final_reaponse = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages
    )
    messages.append(final_reaponse.choices[0].message)


# 检查最后的结果
print(messages)