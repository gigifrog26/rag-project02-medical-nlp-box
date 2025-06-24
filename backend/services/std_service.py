from pymilvus import MilvusClient
from dotenv import load_dotenv
from utils.embedding_factory import EmbeddingFactory
from utils.embedding_config import EmbeddingProvider, EmbeddingConfig
import os
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class StdService:

    def __init__(self, 
                 provider="huggingface",
                 model="BAAI/bge-m3",
                 collection_name="fin_term"):
        # 根据 provider 字符串匹配正确的枚举值
        provider_mapping = {
            'openai': EmbeddingProvider.OPENAI,
            'bedrock': EmbeddingProvider.BEDROCK,
            'huggingface': EmbeddingProvider.HUGGINGFACE
        }
        
        # 创建 embedding 函数
        embedding_provider = provider_mapping.get(provider.lower())
        if embedding_provider is None:
            raise ValueError(f"Unsupported provider: {provider}")
            
        config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=model
        )
        self.embedding_func = EmbeddingFactory.create_embedding_function(config)
        
        # 连接 Milvus
        milvus_host = "127.0.0.1"
        milvus_port = 19530
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
        self.collection_name = collection_name
        self.client.load_collection(self.collection_name)

    def search_similar_terms(self, query: str, limit: int = 5) -> List[Dict]:
        """
        搜索与查询文本相似的金融术语
        
        Args:
            query: 查询文本
            limit: 返回结果的最大数量
            
        Returns:
            包含相似术语信息的列表，每个术语包含：
            - concept_name: 概念名称
            - domain_id: 领域ID
            - distance: 相似度距离
        """
        # 获取查询的向量表示
        query_embedding = self.embedding_func.embed_query(query)
        
        # 设置搜索参数
        search_params = {
            "collection_name": self.collection_name,
            "data": [query_embedding],
            "limit": limit,
            "output_fields": [
                "concept_id", "concept_name", "domain_id"
            ],
            # "filter": "domain_id == 'Condition'"
        }
        
        # 搜索相似项
        search_result = self.client.search(**search_params)

        results = []
        for hit in search_result[0]:
            results.append({
                "concept_name": hit['entity'].get('concept_name'),
                "domain_id": hit['entity'].get('domain_id'),
                "distance": float(hit['distance'])
            })
        return results

    def __del__(self):
        """清理资源，释放集合"""
        if hasattr(self, 'client') and hasattr(self, 'collection_name'):
            self.client.release_collection(self.collection_name)