from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained("infgrad/puff-base-v1")
model = AutoModel.from_pretrained("infgrad/puff-base-v1")


# 将文本转换为向量嵌入
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


# 搜索最相似的文档
def search_similar_papers(
    query_text, model, tokenizer, client, collection_name="test_collection", top_k=1
):
    # 将查询文本转换为嵌入向量
    query_embedding = get_embedding(query_text, model, tokenizer)

    # 搜索最相似的向量
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,  # 返回最相似的 top_k 个结果
    )

    return search_result


# 初始化 Qdrant 客户端，连接到本地 Qdrant 实例
client = QdrantClient(host="localhost", port=6333)

# 示例执行查询
if __name__ == "__main__":
    query_text = "Attention is all you need"  # 你可以修改这个查询文本

    # 执行搜索，获取前 5 个最相似的结果
    results = search_similar_papers(query_text, model, tokenizer, client, top_k=1)

    # 输出检索结果
    for result in results:
        print(
            f"ID: {result.id}, Title: {result.payload['title']}, Score: {result.score}"
        )
