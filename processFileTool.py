from FlagEmbedding import FlagReranker
from langchain_community.retrievers import BM25Retriever
import os, io, PyPDF2, fitz, base64, mimetypes
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import speech_recognition as sr
from pydub import AudioSegment
from aip import AipSpeech

from config import BAIDU_APP_ID, BAIDU_API_KEY, SECRET_KEY, DASHSCOPE_API_KEY


def get_file_type(file_path: str) -> str:
    """判断文件类型"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # 检查文件扩展名
        ext = file_path.lower().split('.')[-1]
        if ext in ['mp3', 'm4a', 'wav']:
            return "audio"
        return "unknown"

    if mime_type.startswith('audio/') or mime_type in ['audio/mp3', 'audio/x-m4a']:
        return "audio"
    elif mime_type.startswith('text/'):
        return "text"
    elif mime_type == 'application/pdf':
        return "pdf"
    else:
        return "unknown"


def extract_images_as_base64_from_pdf(pdf_path):
    """将PDF文件转化为图片列表"""
    pdf_document = fitz.open(pdf_path)
    images_base64 = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 读取图片并转换为 PIL.Image
            image = Image.open(io.BytesIO(image_bytes))

            # 转换为 Base64
            buffered = io.BytesIO()
            image.save(buffered, format=image_ext.upper())
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_base64.append(base64_image)

    pdf_document.close()
    return images_base64


def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF文件中提取文本，支持扫描版和文本版PDF"""
    text = ""

    # 首先尝试直接提取文本
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():  # 如果能提取到文本
                    text += page_text + "\n"
    except Exception as e:
        print(f"直接提取文本失败: {str(e)}")

    # 如果直接提取失败或没有文本，使用LLM识别
    if not text.strip():
        try:
            # 将PDF转换为图片
            images = extract_images_as_base64_from_pdf(pdf_path)
            text = getTextFromImagesWithLLM(images)
        except Exception as e:
            print(f"LLM识别图片失败: {str(e)}")
    return text


def process_audio_file(file_path: str) -> str:
    """处理音频文件，将语音转换为文本"""
    # 初始化百度语音识别客户端
    client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, SECRET_KEY)

    # 转换音频格式为wav（如果需要）
    audio_path = file_path
    if not file_path.lower().endswith('.wav'):
        try:
            audio = AudioSegment.from_file(file_path)
            temp_wav = file_path + '.temp.wav'
            audio.export(temp_wav, format='wav')
            audio_path = temp_wav
        except Exception as e:
            print(f"音频转换失败: {str(e)}")
            return ""

    try:
        # 读取音频文件
        with open(audio_path, 'rb') as fp:
            audio_data = fp.read()

        # 调用百度API进行语音识别
        result = client.asr(audio_data, 'wav', 16000, {
            'dev_pid': 1537,  # 普通话(支持简单的英文识别)
        })

        # 清理临时文件
        if audio_path != file_path:
            os.remove(audio_path)

        # 返回识别结果
        if result['err_no'] == 0:
            return ' '.join(result['result'])
        else:
            print(f"语音识别失败，错误码：{result['err_no']}")
            return ""

    except Exception as e:
        print(f"音频识别失败 {file_path}: {str(e)}")
        # 确保清理临时文件
        if audio_path != file_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return ""


def process_text_file(file_path: str) -> str:
    """处理文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error processing text file {file_path}: {str(e)}")
        return ""


def getTextFromImagesWithLLM(images):
    """视觉大模型解析图片信息获得图片文本"""
    text = ""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    for i in images:
        images_messages = []
        con = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{i}"},
        }
        images_messages.append(con)
        images_messages.append({"type": "text", "text": "请详细的说明图片上的内容，不要进行总结和概述"});
        response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[
                {
                    "role": "user",
                    "content": images_messages,
                }
            ],
            stream=True
        )
        for msg in response:
            delta = msg.choices[0].delta
            if delta.content:
                text_delta = delta.content
                # print(text_delta, end='', flush=True)
                text = text + text_delta
    return text
