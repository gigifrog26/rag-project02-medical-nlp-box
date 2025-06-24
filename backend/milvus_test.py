from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, connections
from pymilvus import model

milvus_host = "127.0.0.1"
milvus_port = 19530
# collection_name = "concepts_only_name"
collection_name = "fin_term"

# 连接到 Milvus
client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")

# res = client.describe_collection(
#     collection_name= collection_name
# )
# res = client.list_indexes(collection_name=collection_name)
# print(res)

client.load_collection(
    collection_name=collection_name
)

# res = client.get_load_state(
#     collection_name=collection_name
# )
# print(res)

# client.drop_collection(collection_name=collection_name)


# index_params = client.prepare_index_params()
# index_params.add_index(
#     field_name="vector",  # 指定要为哪个字段创建索引，这里是向量字段
#     index_type="AUTOINDEX",  # 使用自动索引类型，Milvus会根据数据特性选择最佳索引
#     metric_type="COSINE"  # 使用余弦相似度作为向量相似度度量方式
#       # 索引参数：nlist表示聚类中心的数量，值越大检索精度越高但速度越慢
# )

# client.create_index(
#     collection_name=collection_name,
#     index_params=index_params
# )


embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cpu',
            trust_remote_code=True
        )

# query = "SOB"
# query_embeddings = embedding_function([query])


# # 搜索余弦相似度最高的
# search_result = client.search(
#     collection_name=collection_name,
#     data=[query_embeddings[0].tolist()],
#     limit=5,
#     output_fields=["concept_name", 
#                 #    "synonyms", 
#                    "concept_class_id", 
#                    ]
# )
# print(f"Search result for 'Somatic hallucination': {search_result}")

# # 查询所有匹配的实体
# query_result = client.query(
#     collection_name=collection_name,
#     filter="concept_name == 'Dyspnea'",
#     output_fields=["concept_name", 
#                 #    "synonyms", 
#                    "concept_class_id", 
#                    ],
#     limit=5
# )
# print(f"Query result for concept_name == 'Dyspnea': {query_result}")


# 示例查询
# query = "A Priori Probability"
# query_embeddings = embedding_function([query])

# # 搜索余弦相似度最高的
# search_result = client.search(
#     collection_name=collection_name,
#     data=[query_embeddings[0].tolist()],
#     limit=5,
#     output_fields=["concept_name", "domain_id"])
# print(f"Search result for '{query}': {search_result}")


# query_result = client.query(
#     collection_name=collection_name,
#     filter="concept_name == 'A Priori Probability'",
#     output_fields=["concept_name", "domain_id"],
#     limit=5
# )
# print(f"Query result for concept_name == '{query}': {query_result}")
