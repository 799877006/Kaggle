from openai import OpenAI
import pandas as pd
from datasets import load_dataset
import time
import logging
import os

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/chinese.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger()

# 配置火山引擎客户端
client = OpenAI(
    api_key = "6566d89b-1398-4347-bca8-c4a0dd80b0c6",
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

ds = load_dataset("mandarjoshi/trivia_qa", "rc")

def get_last_processed_index(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return len(df)
    return 0

def process_and_save_dataset(dataset, output_file, subset_name):
    # 检查是否存在已处理的数据
    start_index = get_last_processed_index(output_file)
    logger.info(f"从索引 {start_index} 开始处理 {subset_name} 数据集")
    
    # 如果文件不存在，创建带有表头的空文件
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['question', 'answer']).to_csv(output_file, index=False)
    
    for i in range(start_index, len(dataset)):
        question = dataset['question'][i]
        answer = dataset['answer'][i]['normalized_value']
        
        try:
            # 使用火山引擎API翻译问题
            question_response = client.chat.completions.create(
                model="ep-20250104210627-75nhw",
                messages=[
                    {"role": "system", "content": "你是一个专业的翻译助手，你的回答不能出现除了人名，地名，专有名词，术语以外的任何英语单词。"},
                    {"role": "user", "content": f"这有一组英语问答，用地道的汉语只将其中的英文问题改写翻译成中文问题。\n问题：{question}\n答案：{answer}"}
                ]
            )
            chinese_question = question_response.choices[0].message.content
            
            # 使用火山引擎API翻译答案
            answer_response = client.chat.completions.create(
                model="ep-20250104210627-75nhw",
                messages=[
                    {"role": "system", "content": "你是一个专业的翻译助手，你的回答不能出现除了人名，地名，专有名词，术语以外的任何英语单词。"},
                    {"role": "user", "content": f"这有一组英语问答，用地道的汉语只将其中的英文回答改写翻译成中文回答。\n问题：{question}\n答案：{answer}"}
                ]
            )
            chinese_answer = answer_response.choices[0].message.content
            
            # 将单条数据追加到CSV文件
            pd.DataFrame({
                'question': [chinese_question],
                'answer': [chinese_answer]
            }).to_csv(output_file, mode='a', header=False, index=False)
            
            logger.info(f"{subset_name} - 问题：{chinese_question}")
            logger.info(f"{subset_name} - 答案：{chinese_answer}")
            
        except Exception as e:
            logger.error(f"处理第 {i} 条数据时出错: {str(e)}")
            # 记录空值，保持数据条数一致
            pd.DataFrame({
                'question': [''],
                'answer': ['']
            }).to_csv(output_file, mode='a', header=False, index=False)
            continue
            
        if i % 100 == 0:
            logger.info(f"已处理 {subset_name} 数据集的 {i} 条数据")
    
    # 返回处理完的数据总数
    return get_last_processed_index(output_file)

# 处理各个数据集
train_size = process_and_save_dataset(ds["train"], 'dataset/chinese_qa_train.csv', "训练")
validation_size = process_and_save_dataset(ds["validation"], 'dataset/chinese_qa_validation.csv', "验证")
test_size = process_and_save_dataset(ds["test"], 'dataset/chinese_qa_test.csv', "测试")

# 显示各数据集的样本数
logger.info(f"训练集大小：{train_size}行")
logger.info(f"验证集大小：{validation_size}行")
logger.info(f"测试集大小：{test_size}行")