import pandas as pd
from datasets import load_dataset
import logging
import os
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import sys
import queue
from logging.handlers import QueueHandler, QueueListener

# 创建日志队列
log_queue = queue.Queue(-1)  # 无限队列大小

# 配置日志处理器
file_handler = logging.FileHandler('peft/logs/chinese.log', mode='a', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

# 设置格式
formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 配置根日志记录器
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(QueueHandler(log_queue))

# 创建并启动队列监听器
listener = QueueListener(log_queue, file_handler, console_handler)
listener.start()

# 设置 httpx 日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)

# 在程序结束时停止监听器
import atexit
atexit.register(listener.stop)

logger = logging.getLogger()

# 配置多个模型ID和对应的描述
MODEL_INFO = {
    "ep-20250107150303-5shkl": "Doubao-pro-32k",
    "ep-20250107150339-slv5z": "Doubao-pro-128k",
    "ep-20250107150339-p2fwr": "Doubao-lite-32k",
    "ep-20250107150339-8pvn6": "Doubao-lite-128k",
    "ep-20250107150421-9nglh": "Doubao-lite-128k",
    "ep-20250107150421-xgkrn": "Doubao-pro-128k",
    "ep-20250107150421-7dgnz": "Doubao-lite-32k",
    "ep-20250107150435-v4c8f": "Doubao-pro-4k",
    "ep-20250107150435-p5krn": "Doubao-lite-4k",
}

# 线程本地存储，为每个线程保存一个独立的OpenAI客户端
thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key="6566d89b-1398-4347-bca8-c4a0dd80b0c6",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
    return thread_local.client

class DataProcessor:
    def __init__(self, output_file, subset_name):
        self.output_file = output_file
        self.subset_name = subset_name
        self.lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0
        self.last_progress_time = time.time()
        
    def save_result(self, chinese_question, chinese_answer):
        with self.lock:
            pd.DataFrame({
                'question': [chinese_question],
                'answer': [chinese_answer]
            }).to_csv(self.output_file, mode='a', header=False, index=False)
            self.processed_count += 1
            
            # 每30秒显示一次进度
            current_time = time.time()
            if current_time - self.last_progress_time > 30:
                logger.info(f"{self.subset_name} - 已处理: {self.processed_count}, 错误: {self.error_count}")
                self.last_progress_time = current_time

    def process_item(self, args):
        i, question, answer, model_id = args
        model_desc = MODEL_INFO[model_id]
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                client = get_client()
                
                # 翻译问题
                question_response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "你是一个专业的翻译助手，你的回答不能出现除了人名，地名，专有名词，术语以外的任何英语单词。"},
                        {"role": "user", "content": f"这有一组英语问答，用地道的汉语只将其中的英文问题改写翻译成中文问题。请直接写出翻译后的中文问题，不要出现问题冒号的格式。\n问题：{question}\n答案：{answer}"}
                    ]
                )
                chinese_question = question_response.choices[0].message.content
                
                # 翻译答案
                answer_response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "你是一个专业的翻译助手，你的回答不能出现除了人名，地名，专有名词，术语以外的任何英语单词。"},
                        {"role": "user", "content": f"这有一组英语问答，用地道的汉语只将其中的英文回答改写翻译成中文回答。请直接写出翻译后的中文回答，不要出现答案冒号的格式。\n问题：{question}\n答案：{answer}"}
                    ]
                )
                chinese_answer = answer_response.choices[0].message.content
                
                self.save_result(chinese_question, chinese_answer)
                logger.info(f"{self.subset_name} [{model_desc}] - 问题：{chinese_question}")
                logger.info(f"{self.subset_name} [{model_desc}] - 答案：{chinese_answer}")
                
                return True
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    with self.lock:
                        self.error_count += 1
                    logger.error(f"处理第 {i} 条数据失败 [模型: {model_desc}]: {str(e)}")
                    self.save_result('', '')
                    return False
                time.sleep(1)  # 重试前等待1秒

def process_and_save_dataset(dataset, output_file, subset_name):
    start_index = get_last_processed_index(output_file)
    total_items = len(dataset) - start_index
    logger.info(f"从索引 {start_index} 开始处理 {subset_name} 数据集，总计需要处理 {total_items} 条数据")
    
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['question', 'answer']).to_csv(output_file, index=False)
    
    processor = DataProcessor(output_file, subset_name)
    model_ids = list(MODEL_INFO.keys())
    
    # 将数据集分批处理，每批处理的数量减少
    batch_size = 9  # 减小批次大小
    for batch_start in range(start_index, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        logger.info(f"开始处理 {subset_name} 数据集第 {batch_start}-{batch_end} 条数据")
        
        tasks = [
            (i, dataset['question'][i], dataset['answer'][i]['normalized_value'], model_ids[i % len(model_ids)])
            for i in range(batch_start, batch_end)
        ]
        
        # 计算最优线程数
        optimal_workers = 16
        logger.info(f"使用线程数：{optimal_workers}")
        
        # 使用线程池处理任务
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [executor.submit(processor.process_item, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    logger.error(f"任务执行失败: {str(e)}")
        
        logger.info(f"完成批次处理 {batch_start}-{batch_end}, 已处理: {processor.processed_count}/{total_items}")
    
    logger.info(f"{subset_name}数据集处理完成 - 总计: {processor.processed_count}, 错误: {processor.error_count}")
    return get_last_processed_index(output_file)

def get_last_processed_index(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return len(df)
    return 0

if __name__ == '__main__':
    # 处理各个数据集
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    logger.info("数据集加载完成，开始处理...")
    
    # 显示数据集大小信息
    logger.info(f"训练集大小: {len(ds['train'])} 条")
    logger.info(f"验证集大小: {len(ds['validation'])} 条")
    logger.info(f"测试集大小: {len(ds['test'])} 条")
    
    train_size = process_and_save_dataset(ds["train"], 'dataset/chinese_qa_train.csv', "训练")
    validation_size = process_and_save_dataset(ds["validation"], 'dataset/chinese_qa_validation.csv', "验证")
    test_size = process_and_save_dataset(ds["test"], 'dataset/chinese_qa_test.csv', "测试")

    logger.info(f"全部处理完成:")
    logger.info(f"训练集处理完成：{train_size}行")
    logger.info(f"验证集处理完成：{validation_size}行")
    logger.info(f"测试集处理完成：{test_size}行")