from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from FlagEmbedding import FlagReranker
from langchain_community.retrievers import BM25Retriever
import os, math, fitz, PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import speech_recognition as sr
from aip import AipSpeech

from config import MODEL_NAME, BASE_URL
from processFileTool import get_file_type, extract_text_from_pdf, process_audio_file, process_text_file

prompt_template = """基于以下已知信息，简洁和专业的回答用户的问题，不需要在答案中添加编造成分。
    已知内容：{context}
    问题：{question}"""


def getPrompt(question, vector_store, k):
    simialText = []
    search_list = vector_store.similarity_search(question, k=k)
    for i in search_list:
        simialText.append(i.page_content)
    after_rank_search_list = afterRanker(question, simialText, k)
    return PromptTemplate.from_template(prompt_template).invoke(
        {"context": after_rank_search_list, "question": question}).text


def afterRanker(question, search_list, k):
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    score_text = []
    for search in search_list:
        score_text.append([reranker.compute_score([question, search]), search])
    sorted_data = sorted(score_text, key=lambda x: x[0], reverse=True)
    minK = max(min(len(sorted_data), 3), math.floor(k / 3))
    text = []
    for i in sorted_data[:minK]:
        text.append(i[1])
    return text


def talkWithLLM(prompt):
    model = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=BASE_URL,
    )
    return model.invoke(prompt).content


def process_directory(directory_path: str) -> tuple[list[str], list[str]]:
    """同步处理目录下的所有文件"""
    all_text = []
    processed_files = []

    # 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 收集所有任务
        future_to_path = {}

        # 遍历目录下的所有文件和文件夹
        for path in Path(directory_path).rglob('*'):
            if path.is_file():
                file_type = get_file_type(str(path))
                if file_type in ["pdf", "audio", "text"]:
                    # 提交任务到线程池
                    future = executor.submit(process_single_file, str(path), file_type)
                    future_to_path[future] = str(path)

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


def main():
    # 初始化嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    vector_store = Chroma(persist_directory="data",
                          embedding_function=embeddings,
                          collection_name="lc_chroma_demo")
    collection = vector_store.get();
    if len(collection["ids"]) == 0:
        # 替换原来的单个PDF处理为目录处理
        directory_path = "./data"  # 指定要处理的目录路径
        text_list, processed_file_list = process_directory(directory_path)
        # 文档切割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, add_start_index=True)
        text_documents = text_splitter.split_documents([Document(page_content=text) for text in text_list])

        # 存入向量数据库
        vector_store = Chroma.from_documents(
            documents=text_documents,
            embedding=embeddings,
            persist_directory="data",
            collection_name="lc_chroma_demo")

    query = "王小冉的性格外貌是怎么样的"
    prompt = getPrompt(query, vector_store, 25)
    print(prompt)

    ans = talkWithLLM(prompt)
    print(print)


if __name__ == "__main__":
    main()