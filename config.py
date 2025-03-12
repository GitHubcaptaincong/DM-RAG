import os

# 对话大模型配置
MODEL_NAME = "YOUR_MODEL_NAME"
BASE_URL = "LLM_BASE_URL"

# 百度语音识别配置
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID')
BAIDU_API_KEY = os.getenv('BAIDU_API_KEY')
SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')

# qianwen视觉大模型KEY
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")