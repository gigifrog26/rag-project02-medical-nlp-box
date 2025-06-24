from transformers import pipeline
import torch
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import ast

# 加载环境变量
load_dotenv("C:\czdata\rag-project02-medical-nlp-box\backend\my.env")

text = "By maintaining a strong accounting cushion, businesses can better withstand financial volatility, while strategically managing after-tax contributions to optimize overall profitability"

prompt = ChatPromptTemplate.from_template("""
                You are a data analyst, and your job is reading sentences and figure out 
                finance related entities (for example: money, profit, stock) and 
                ONLY output the Python list, NO other words, NO formatting, NO code block.
                Please analysis below sentence:
                {question}""")

llm = ChatOpenAI(
    model="deepseek-reasoner",  # DeepSeek API 支持的模型名称
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,        # 控制输出的随机性(0-1之间,越大越随机)
    max_tokens=2048,        # 最大输出长度
    top_p=0.95,            # 控制输出的多样性(0-1之间)
    presence_penalty=0.0,   # 重复惩罚系数(-2.0到2.0之间)
    frequency_penalty=0.0,  # 频率惩罚系数(-2.0到2.0之间)
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)
answer = llm.invoke(prompt.format(question=text))
print(answer)

result_list = ast.literal_eval(answer.content)
print(len(result_list))
for item in result_list:
    print(item)





# pipe = pipeline("token-classification", 
#     model="shrimp1106/bert-finetuned-finance_ner", 
#     device=-1)
# result = pipe(text)
# print(result)       
# # 确保结果是实体列表
# if isinstance(result, dict):
#     result = result.get('entities', [])
#     print("Found!!!")
#     print(result)
# else:
#     print("No!!!")
#     print(result)


    #     # 合并相关实体（如生物结构和症状）
    #     combined_result = self._combine_entities(result, text, options)
        
    #     # 移除重叠实体
    #     non_overlapping_result = self._remove_overlapping_entities(combined_result)
        
    #     # 根据术语类型过滤实体
    #     filtered_result = self._filter_entities(non_overlapping_result, term_types)
        
    #     return {
    #         "text": text,
    #         "entities": filtered_result
    #     }

    # def _combine_entities(self, result, text, options):
    #     """
    #     合并相关的实体，如生物结构和症状
    #     """
    #     combined_result = []
    #     i = 0
    #     while i < len(result):
    #         entity = result[i]
    #         entity['score'] = float(entity['score'])

    #         if options['combineBioStructure'] and entity['entity_group'] in ['SIGN_SYMPTOM', 'DISEASE_DISORDER']:
    #             # 检查并合并生物结构
    #             combined_entity = self._try_combine_with_bio_structure(result, i, text)
    #             if combined_entity:
    #                 combined_result.append(combined_entity)
    #                 i += 1
    #                 continue
    #         combined_result.append(entity)
    #         i += 1
    #     return combined_result

    # def _try_combine_with_bio_structure(self, result, i, text):
    #     """
    #     尝试将当前实体与生物结构实体合并
    #     """
    #     # 检查前一个实体
    #     if i > 0 and result[i-1]['entity_group'] == 'BIOLOGICAL_STRUCTURE':
    #         return self._create_combined_entity(result[i-1], result[i], text)
    #     # 检查后一个实体
    #     elif i < len(result) - 1 and result[i+1]['entity_group'] == 'BIOLOGICAL_STRUCTURE':
    #         return self._create_combined_entity(result[i], result[i+1], text)
    #     return None

    # def _create_combined_entity(self, entity1, entity2, text):
    #     """
    #     创建合并后的实体
    #     """
    #     start = min(entity1['start'], entity2['start'])
    #     end = max(entity1['end'], entity2['end'])
    #     word = text[start:end]
    #     return {
    #         'entity_group': 'COMBINED_BIO_SYMPTOM',
    #         'word': word,
    #         'start': start,
    #         'end': end,
    #         'score': (entity1['score'] + entity2['score']) / 2,
    #         'original_entities': [entity1, entity2]
    #     }

    # def _remove_overlapping_entities(self, entities):
    #     """
    #     移除重叠的实体，保留得分最高的实体
    #     """
    #     # 按开始位置、结束位置（降序）和得分（降序）排序
    #     sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end'], -x['score']))
    #     non_overlapping = []
    #     last_end = -1

    #     i = 0
    #     while i < len(sorted_entities):
    #         current = sorted_entities[i]
            
    #         # 如果当前实体与之前的实体不重叠，直接添加
    #         if current['start'] >= last_end:
    #             non_overlapping.append(current)
    #             last_end = current['end']
    #             i += 1
    #         else:
    #             # 处理重叠实体
    #             same_span = [current]
    #             j = i + 1
    #             while j < len(sorted_entities) and sorted_entities[j]['start'] == current['start'] and sorted_entities[j]['end'] == current['end']:
    #                 same_span.append(sorted_entities[j])
    #                 j += 1
                
    #             # 选择得分最高的实体
    #             best_entity = max(same_span, key=lambda x: x['score'])
    #             if best_entity['end'] > last_end:
    #                 non_overlapping.append(best_entity)
    #                 last_end = best_entity['end']
                
    #             i = j

    #     return non_overlapping

    # def _filter_entities(self, entities, term_types):
    #     """
    #     根据术语类型过滤实体
    #     """
    #     filtered_result = []
    #     for entity in entities:
    #         if term_types.get('allMedicalTerms', False):
    #             filtered_result.append(entity)
    #         elif (term_types.get('symptom', False) and entity['entity_group'] in ['SIGN_SYMPTOM', 'COMBINED_BIO_SYMPTOM']) or \
    #              (term_types.get('disease', False) and entity['entity_group'] == 'DISEASE_DISORDER') or \
    #              (term_types.get('therapeuticProcedure', False) and entity['entity_group'] == 'THERAPEUTIC_PROCEDURE'):
    #             filtered_result.append(entity)
    #     return filtered_result
