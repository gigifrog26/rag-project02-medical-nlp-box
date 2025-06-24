import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import ast
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceNERService:
   
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template("""
                You are a data analyst, and your job is reading sentences and figure out 
                finance related entities (for example: money, profit, stock) and 
                ONLY output the Python list, NO other words, NO formatting, NO code block.
                Please analysis below sentence:
                {question}""")
        
        self.llm = ChatOpenAI(
            model="deepseek-reasoner",  # DeepSeek API 支持的模型名称
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,        # 控制输出的随机性(0-1之间,越大越随机)
            max_tokens=2048,        # 最大输出长度
            top_p=0.95,            # 控制输出的多样性(0-1之间)
            presence_penalty=0.0,   # 重复惩罚系数(-2.0到2.0之间)
            frequency_penalty=0.0,  # 频率惩罚系数(-2.0到2.0之间)
            api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
        )
  
    def process(self, text):
        answer = self.llm.invoke(self.prompt.format(question=text))
        result_list = []
        if answer.content != "":
            result_list = ast.literal_eval(answer.content)
        return result_list
        