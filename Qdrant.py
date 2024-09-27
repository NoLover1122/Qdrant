from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize
import os
import uuid  # 用于生成唯一 ID

# 加载模型和分词器（从 Hugging Face 模型库直接加载）
tokenizer = AutoTokenizer.from_pretrained("infgrad/puff-base-v1")
model = AutoModel.from_pretrained("infgrad/puff-base-v1").eval()

# 定义线性层，将模型的输出转换为 768 维
vector_dim = 768  # 确认向量的维度是 768
vector_linear = torch.nn.Linear(
    in_features=model.config.hidden_size, out_features=vector_dim
)


# 提取 PDF 文件中的文本
def extract_text_from_pdf(pdf_path):
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# 将文本转换为向量嵌入
def generate_embedding(text, model, tokenizer, vector_linear):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        # 获取模型的最后一层隐藏状态输出
        last_hidden_state = model(**inputs)[0]

        # 使用 attention_mask 掩盖填充的部分
        attention_mask = inputs["attention_mask"]
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        # 对最后一层隐藏状态进行池化（按行求和，除以有效词数）
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # 通过线性变换将嵌入向量转换为指定维度（768 维）
        vectors = vector_linear(vectors)

        # 对嵌入向量进行归一化处理 (使用 PyTorch Tensor 的归一化)
        vectors = normalize(vectors, p=2, dim=1)  # 在 Tensor 上进行归一化

        # 将结果转换为 NumPy 数组
        vectors = vectors.cpu().numpy()

    return vectors


# 提取 PDF 文件中的文本并生成嵌入
pdf_path = "Attention Is All You Need.pdf"  # 你自己的 PDF 文件路径
paper_text = extract_text_from_pdf(pdf_path)
embedding = generate_embedding(paper_text, model, tokenizer, vector_linear)

# 确保嵌入向量的形状是二维数组
if embedding.shape != (1, vector_dim):
    embedding = embedding.reshape(1, -1)  # 调整形状为 (1, 768)

# 连接到本地 Qdrant 实例
client = QdrantClient(host="localhost", port=6333)

# 创建一个新的集合来存储向量，固定向量维度为 768
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(
        size=vector_dim, distance=Distance.COSINE  # 使用 768 维向量  # 使用余弦距离
    ),
)

# 使用 upsert 方法将文档向量存储到 Qdrant
client.upsert(
    collection_name="test_collection",  # 集合的名称
    points=[
        {
            "id": str(uuid.uuid4()),  # 为文档生成唯一 ID
            "vector": embedding[0],  # 存储生成的嵌入向量
            "payload": {"title": "Sample Research Paper"},  # 文档的元数据
        }
    ],
)

print("向量已成功存储到 Qdrant")
