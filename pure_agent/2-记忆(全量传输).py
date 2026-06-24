import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 创建客户端
client= OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

messages = [
    {"role": "system", "content": "你是一个很有用的助手, 小帅"}
]

while True:
    query = input("请输入您的用户提示词或输入'/exit'退出\n")
    if query=="/exit":
        break
    messages.append({"role": "user", "content": query})
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages
        )
    print(resp.choices[0].message.content)
    messages.append({"role": "assistant", "content": resp.choices[0].message.content})