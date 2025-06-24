from pymilvus import model
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch    
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, connections

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 OpenAI 嵌入函数
embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            # model_name='nvidia/NV-Embed-v2', 
            # model_name='dunzhang/stella_en_1.5B_v5',
            # model_name='all-mpnet-base-v2',
            # model_name='intfloat/multilingual-e5-large-instruct',
            # model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
            model_name='BAAI/bge-m3',
            # model_name='jinaai/jina-embeddings-v3',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
# embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

# 文件路径
file_path = "backend/data/finance.csv"
# db_path = "backend/db/snomed_bge_m3.db"
milvus_host = "127.0.0.1"
milvus_port = 19530
collection_name = "fin_term"

# 连接到 Milvus
client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")

# 加载数据
logging.info("Loading data from CSV")
df = pd.read_csv(file_path, dtype=str, low_memory=False).fillna("NA")

# 获取向量维度（使用一个样本文档）
sample_doc = "Sample Text"
sample_embedding = embedding_function([sample_doc])[0]
vector_dim = len(sample_embedding)

# 构造Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim), # BGE-m3 最重要
    FieldSchema(name="concept_name", dtype=DataType.VARCHAR, max_length=300),
    FieldSchema(name="domain_id", dtype=DataType.VARCHAR, max_length=100),
]
schema = CollectionSchema(fields, "Finance Concepts", enable_dynamic_field=True)

# 如果集合不存在，创建集合
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        # dimension=vector_dim
    )
    logging.info(f"Created new collection: {collection_name}")

# # 在创建集合后添加索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",  # 指定要为哪个字段创建索引，这里是向量字段
    index_type="AUTOINDEX",  # 使用自动索引类型，Milvus会根据数据特性选择最佳索引
    metric_type="COSINE"  # 使用余弦相似度作为向量相似度度量方式
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params
)

# 批量处理
batch_size = 1024

for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    # 准备文档
    docs = []
    for _, row in batch_df.iterrows():
        docs.append(row['concept_name'])

    # 生成嵌入
    try:
        embeddings = embedding_function(docs)
        logging.info(f"Generated embeddings for batch {start_idx // batch_size + 1}")
    except Exception as e:
        logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
        continue

    # 准备数据
    data = [
        {
            # "id": idx + start_idx,
            "vector": embeddings[idx],
            "concept_name": str(row['concept_name']),
            "domain_id": str(row['domain_id'])
        } for idx, (_, row) in enumerate(batch_df.iterrows())
    ]

    try:
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        logging.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
    except Exception as e:
        logging.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

logging.info("Insert process completed.")


# 示例查询
query = "A Priori Probability"
query_embeddings = embedding_function([query])
client.load_collection(
    collection_name=collection_name
)

# 搜索余弦相似度最高的
search_result = client.search(
    collection_name=collection_name,
    data=[query_embeddings[0].tolist()],
    limit=5,
    output_fields=["concept_name", "domain_id"])
logging.info(f"Search result for '{query}': {search_result}")

