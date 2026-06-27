import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
import requests
from typing import List

# 自定义嵌入函数
# EmbeddingFunction的作用是内置许多方法, 可以调用或重写
class OllamaBGEEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str = "bge-m3:567m", api_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self._api_url = f"{api_base_url}/api/embeddings"
    
    # 最重要的方法, 用来处理调用模型的过程
    # Dodumenets表示这是一个文本列表, 处理所有需要转化的文本, Embeddings是向量列表, 和文本一一对应
    def __call__(self, texts: Documents)-> Embeddings:
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "prompt": f"passage: {text}"
            }
            # 向指定url发送请求
            try:
                resp = requests.post(self._api_url, json=payload, timeout=60)
                resp.raise_for_status() #抛出错误
                # 从响应中提取向量
                embedding = resp.json().get("embedding")
                if embedding is None:
                    raise ValueError(f"Ollama API 未找到'embedding'字段: {resp.text}")
                embeddings.append(embedding)

            except Exception as e:
                print(f"出现以下错误:{e}")
                raise e # 中断向量转化, 防止将不完整的向量存储
        return embeddings
    # 重写查询转化向量的方法, bge模型对查询和文档转化时推荐使用不同的前缀
    def embed_query(self, input):
        return self.__call__([f"query: {text}" for text in input])
    # 修改默认计算相似度的方法, bge推荐使用cosine, 原本默认的是l2
    # 处理计算方法的位置不是这里, 要在创建集合时的metadata中定义
    def default_space(self)->Space:
        return "cosine"
    @staticmethod
    def name() -> str:
        return "ollama_bge_m3"

    def get_config(self) -> dict:
        return {"model": self.model_name, "api_base_url": self.api_base_url}
    
# 创建chromadb客户端和集合
# 存储数据到磁盘
client = chromadb.PersistentClient(path="./my_chroma_db")

# 创建自定义嵌入函数实例
embedding_fn = OllamaBGEEmbedding()

# 创建一个集合(类似数据库中的表)
# 指定embedding_function后在添加和查询时就会自动调用它
try:
    collection = client.create_collection(
        name="my_knowledge_base",   #集合名称
        embedding_function=embedding_fn,    # 指定文本嵌入的函数
        metadata={"hnsw:space": embedding_fn.default_space()},  # 在元数据中设置计算相似度的方法, 这是这个集合自己的元数据(不是添加在其中的文档中的)
        get_or_create=True  # 当集合存在时, 直接返回存在的集合
    )
    print("成功创建集合 my_knowledge_base")
except Exception as e:
    raise e

# 将文档添加到集合中
documents = [
    "Ollama 是一个用于本地运行大型语言模型的框架。",
    "ChromaDB 是一个轻量级的向量数据库，专为 AI 应用设计。",
    "BGE-M3 是一个强大的文本嵌入模型，由 BAAI 开发。",
    "Python 是数据科学和机器学习领域最流行的编程语言之一。",
]
# 创建每个文档的唯一id
# 这里的id使用最简单的数字索引
ids = [f"doc_{i}" for i in range(len(documents))]

# 添加到集合中
# 会自动调用指定的embedding_function来生成向量
collection.add(
    ids=ids,
    documents=documents
    # 还可以添加一些metadata, 这是每个文本的元数据
)


# 查询数据
# 需要查询的文本
query_text = "如何在本地运行大语言模型?"

# 执行相似性搜索
results = collection.query(
    query_texts=[query_text],   # 查询文本列表
    n_results=3,    # 返回最相似的3个结果
)
"""
results返回的结果是:
results = {
    "ids": List[List[str]],          # 文档ID，外层列表对应查询批次，内层列表是返回的文档
    "distances": List[List[float]],  # 相似度距离（越小越相似），结构同上
    "documents": List[List[str]],    # 文档的原文内容，结构同上
    "metadatas": List[List[dict]],   # 文档的元数据（你代码里没加，此处为空列表）
    "embeddings": List[List[list]]   # 默认不返回，除非你在 query 中指定 include=["embeddings"]
}

这里返回的结果之所以是列表的列表, 是因为查询文本可以有多条, 因此外层列表包含的是不同查询批次
每次查询会有多个结果, 所以内层列表返回的是找到的相似文档的条数
"""


print(f"查询: {query_text}")
print("最相似的文档是:")
for i in range(len(results['ids'][0])):
    doc_id = results['ids'][0][i]
    distance = results["distances"][0][i] # 距离, 使用cosine结果在[0,2], 越小越相似
    document = results["documents"][0][i]
    print(f"{i+1}.ID: {doc_id}, 距离: {distance:.4f}")
    print(f"内容: {document}")