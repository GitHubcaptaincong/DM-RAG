from colorama import Fore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from FlagEmbedding import FlagReranker
import os, math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, \
    ContextRecall, ContextRelevance, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
import hashlib
import json

from LLMConfig import MODEL_NAME, BASE_URL
from ProcessFileTool import get_file_type, extract_text_from_pdf, process_audio_file, process_text_file
from utils.ColoredPrintHandler import ColoredPrintHandler


def init_vector_store():
    """初始化嵌入模型"""
    embeddings = _get_dash_scope_embeddings()
    vector_store = Chroma(persist_directory="data",
                          embedding_function=embeddings,
                          collection_name="lc_chroma_demo")
    collection = vector_store.get()
    if len(collection["ids"]) == 0:
        # 替换原来的单个PDF处理为目录处理
        directory_path = "./data"  # 指定要处理的目录路径
        text_list, processed_file_list = process_directory(directory_path)
        # 文档切割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, add_start_index=True)
        text_documents = text_splitter.split_documents([Document(page_content=text) for text in text_list])

        # 存入向量数据库
        vector_store = Chroma.from_documents(
            documents=text_documents,
            embedding=embeddings,
            persist_directory="data",
            collection_name="lc_chroma_demo")
    return vector_store


prompt_template = """基于以下已知信息，简洁和专业的回答用户的问题，不需要在答案中添加编造成分。
    已知内容：{context}
    问题：{question}"""


def get_prompt(question, vector_store, k):
    simial_text = []
    search_list = vector_store.similarity_search(question, k=k)
    for i in search_list:
        simial_text.append(i.page_content)
    after_rank_search_list = after_ranker(question, simial_text, k)
    return PromptTemplate.from_template(prompt_template).invoke(
        {"context": after_rank_search_list, "question": question}).text


def after_ranker(question, search_list, k):
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    score_text = []
    for search in search_list:
        score_text.append([reranker.compute_score([question, search]), search])

    sorted_data = sorted(score_text, key=lambda x: x[0], reverse=True)
    min_k = max(min(len(sorted_data), 3), math.floor(k / 3))
    return list(map(lambda x: x[1], sorted_data[:min_k]))


def talk_with_llm(prompt, model=MODEL_NAME):
    model = ChatOpenAI(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=BASE_URL,
        callbacks=[ColoredPrintHandler(Fore.GREEN)]
    )
    return model.invoke(prompt).content


def process_directory(directory_path: str) -> tuple[list[str], list[str]]:
    """同步处理目录下的所有文件，带缓存机制"""
    cache_dir = ".file_processing_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    all_text = []
    processed_files = []

    # 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 收集所有任务
        future_to_path = {}

        # 遍历目录下的所有文件和文件夹
        for path in Path(directory_path).rglob('*'):
            if path.is_file():
                file_path = str(path)
                file_type = get_file_type(file_path)
                
                if file_type in ["pdf", "audio", "text"]:
                    # 检查缓存
                    cache_file = get_cache_filename(file_path, cache_dir)
                    
                    if os.path.exists(cache_file):
                        # 如果缓存存在且文件未修改，使用缓存
                        file_mtime = os.path.getmtime(file_path)
                        try:
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                            
                            if cache_data.get('mtime') == file_mtime:
                                all_text.append(cache_data.get('text', ''))
                                processed_files.append(file_path)
                                continue  # 跳过处理这个文件
                        except Exception as e:
                            print(f"读取缓存失败，将重新处理文件 {file_path}: {e}")
                    
                    # 提交任务到线程池
                    future = executor.submit(process_single_file_with_cache, 
                                            file_path, file_type, cache_dir)
                    future_to_path[future] = file_path

        # 等待所有任务完成并处理结果
        for future in future_to_path:
            file_path = future_to_path[future]
            try:
                text = future.result()
                if text:  # 如果提取到了文本
                    processed_files.append(file_path)
                    all_text.append(f"\n--- Content from {file_path} ---\n{text}\n")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    return all_text, processed_files


def get_cache_filename(file_path, cache_dir):
    """生成缓存文件名"""
    # 使用文件路径的哈希作为缓存文件名
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    return os.path.join(cache_dir, f"{file_hash}.json")


def process_single_file_with_cache(file_path, file_type, cache_dir):
    """处理单个文件并缓存结果"""
    try:
        # 处理文件
        text = process_single_file(file_path, file_type)
        
        # 保存到缓存
        if text:
            cache_file = get_cache_filename(file_path, cache_dir)
            file_mtime = os.path.getmtime(file_path)
            
            cache_data = {
                'text': text,
                'mtime': file_mtime,
                'type': file_type
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
        
        return text
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}")


def process_single_file(file_path: str, file_type: str) -> str:
    """同步处理单个文件"""
    try:
        if file_type == "pdf":
            return extract_text_from_pdf(file_path)
        elif file_type == "audio":
            return process_audio_file(file_path)
        elif file_type == "text":
            return process_text_file(file_path)
        return ""
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}")


def evaluate_rag_system(vector_store, test_questions, custom_model_config, ground_truth_answers=None):
    """使用RAGAS评估RAG系统的性能"""
    # 准备评估数据集
    eval_dataset = prepare_evaluation_dataset(vector_store, test_questions, ground_truth_answers)
    
    # 配置评估模型
    evaluate_llm, embeddings = configure_evaluation_models(custom_model_config)
    
    # 执行评估
    result = run_evaluation(eval_dataset, evaluate_llm, embeddings)
    
    return result


def prepare_evaluation_dataset(vector_store, test_questions, ground_truth_answers=None):
    """准备RAGAS评估数据集"""
    from tqdm import tqdm
    
    contexts = []
    questions = []
    answers = []

    # 如果没有提供标准答案，则使用当前模型生成答案
    if ground_truth_answers is None:
        ground_truth_answers = [""] * len(test_questions)

    # 添加进度条
    print("准备评估数据集...")
    for i, (question, ground_truth) in enumerate(tqdm(list(zip(test_questions, ground_truth_answers)), 
                                               desc="处理测试问题")):
        # 获取上下文
        prompt = get_prompt(question, vector_store, 25)
        retrieved_context = prompt.split("已知内容：")[1].split("问题：")[0].strip()

        # 生成答案
        answer = talk_with_llm(prompt)
        
        # 添加到评估数据集
        contexts.append(retrieved_context)
        questions.append(question)
        answers.append(answer)
        
        # 打印进度
        print(f"\n问题 {i+1}/{len(test_questions)}:")
        print(f"问题: {question}")
        print(f"答案: {answer[:100]}..." if len(answer) > 100 else f"答案: {answer}")

    # 准备RAGAS评估数据
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": [[ctx] for ctx in contexts],
    }

    # 如果有标准答案，添加到数据集
    if all(ground_truth_answers):
        dataset_dict["ground_truth"] = ground_truth_answers

    # 创建Hugging Face Dataset对象
    return Dataset.from_dict(dataset_dict)


def configure_evaluation_models(custom_model_config):
    """配置评估所需的模型"""
    # 配置LLM
    evaluate_llm = LangchainLLMWrapper(ChatOpenAI(
        model=custom_model_config.get("model_name"),
        api_key=custom_model_config.get("api_key"),
        base_url=custom_model_config.get("base_url"),
    ))
    
    # 配置嵌入模型
    embeddings = LangchainEmbeddingsWrapper(_get_dash_scope_embeddings())
    
    return evaluate_llm, embeddings


def run_evaluation(eval_dataset, evaluate_llm, embeddings):
    """运行RAGAS评估"""
    print("\n开始评估...")
    
    # 定义要使用的评估指标
    metrics = [
        ContextPrecision(), 
        ContextRecall(),
        Faithfulness(), 
        AnswerRelevancy(),
        ContextRelevance(), 
        AnswerCorrectness()
    ]
    
    # 执行评估
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluate_llm,
            embeddings=embeddings
        )
        return result
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return f"评估失败: {str(e)}"


def _get_dash_scope_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )


def main():
    """主函数，用于控制程序的执行流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG系统测试和评估工具')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'ragas', 'both'],
                      help='运行模式: test(常规测试), ragas(RAGAS评估), both(两者都运行)')
    parser.add_argument('--query', type=str, default="王小冉的性格外貌是怎么样的",
                      help='测试查询（仅在test模式下使用）')
    parser.add_argument('--save', action='store_true', help='是否保存评估结果到文件')
    
    args = parser.parse_args()
    
    vector_store = init_vector_store()
    
    if args.mode in ['test', 'both']:
        print("\n=== 常规RAG测试 ===")
        query = args.query
        prompt = get_prompt(query, vector_store, 25)
        print(prompt)
        talk_with_llm(prompt)
    
    if args.mode in ['ragas', 'both']:
        run_ragas_evaluation(vector_store, args.save)

def run_ragas_evaluation(vector_store, save_results=False):
    """运行RAGAS评估"""
    print("\n=== RAG系统评估 ===")
    
    # RAGAS评估测试用例
    test_questions = [
        "王小冉的性格外貌是怎么样的",
        "数据中包含哪些主要人物",
        "故事发生在什么地点"
    ]

    ground_truth_answers = [
        "王小冉的眼神忧郁,唯唯诺诺,是个个子不高的女人。",
        "主要人物有：袁本，陈烁，姚波，王小冉和刘伯钊。",
        "发生在T市郊外160多公里外的小村。"
    ]

    # 自定义评估模型配置
    custom_model_config = {
        "base_url": BASE_URL,
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "model_name": MODEL_NAME
    }

    print("开始评估中，这可能需要几分钟时间...")
    result = evaluate_rag_system(
        vector_store,
        test_questions,
        custom_model_config,
        ground_truth_answers=ground_truth_answers
    )

    print("\n评估结果:")
    print(result)
    
    if save_results:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"ragas_results_{timestamp}.txt"
        
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"RAGAS评估结果 - {datetime.now()}\n")
            f.write(str(result))
            
        print(f"\n评估结果已保存到: {result_path}")

    visualize_evaluation_results(result)


def visualize_evaluation_results(result):
    """可视化RAGAS评估结果"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # 尝试将结果转换为DataFrame
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
        elif isinstance(result, dict):
            df = pd.DataFrame([result])
        else:
            # 尝试从对象中提取评估指标
            metrics = {}
            for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 
                         'context_precision', 'context_relevance', 'answer_correctness']:
                if hasattr(result, metric):
                    metrics[metric] = getattr(result, metric)
            df = pd.DataFrame([metrics])
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        ax = df.T.plot(kind='bar', ylim=(0, 1))
        plt.title('RAGAS 评估结果')
        plt.ylabel('分数')
        plt.xlabel('评估指标')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('ragas_results.png')
        print("\n评估结果图表已保存到 ragas_results.png")
        
        # 在支持显示的环境中显示图表
        plt.show()
    
    except Exception as e:
        print(f"可视化结果时出错: {e}")
        print("请确保已安装matplotlib和pandas库")


def check_environment():
    """检查环境变量和依赖项"""
    required_env_vars = ["DASHSCOPE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"错误: 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请设置这些环境变量后再运行程序")
        return False
    
    # 检查目录
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录 {data_dir} 不存在，将创建该目录")
        os.makedirs(data_dir, exist_ok=True)
    
    return True


if __name__ == "__main__":
    if check_environment():
        main()
    else:
        import sys
        sys.exit(1)
