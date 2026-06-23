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

# 发送请求并获取响应
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "你是一个很好的助手, 小帅"},
        {"role": "user", "content": "你叫什么名字?"}
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)
"""
reasoning_effort和extra_body(额外参数, 其中的thinking)是deepseek-v4高级模型支持的
thinking控制是否开启(enabled: 开启, disabled: 关闭)思考模式(默认开启), 在思考模式下, tempurature会失效, 生成的响应会有thinking_content(思考的内容)和content(最终响应)
reasoning_effort是控制思考的强度, 默认是high, 可选max(更强)
"""

# 响应解析
print(response.choices[0].message.content)