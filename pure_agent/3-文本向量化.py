import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 创建客户端
embedding_client=OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
# 定义信息的模型
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"<Message(role={self.role}, content={self.content})>"

# 定义转化信息的模型
class TransMessage:
    def __init__(self, role, content, transmessage):
        self.role = role
        self.content = content
        self.transmessage = transmessage

    def __repr__(self):
        return f"<Message(role={self.role}, content={self.content}, transmessage={self.transmessage})>"


# 定义转化的函数
# 这里使用ollama本地下载的模型处理
def transform(message: Message):
    transform = embedding_client.embeddings.create(
        model="bge-m3:567m",    # 该模型会将文本转化成1024维向量
        input=message.content
    )
    # 返回的是一个响应对象, 真实的向量数据在其中的.data[0].embedding中
    return TransMessage(role=message.role, content=message.content, transmessage=transform.data[0].embedding)


m = Message(role="user", content="你好")

transform(m)