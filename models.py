# models.py
# 可用模型列表，以及获得访问模型的客户端
#     实际使用时可以根据自己的实际情况调整

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

from config import BATCH_SIZE,DASHSCOPE_API_VL_MODEL

load_dotenv()

# 阿里的通义千问大模型（主要使用）
#    官网: https://bailian.console.aliyun.com/#/home
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = os.getenv("DASHSCOPE_API_URL")
DASHSCOPE_API_MODEL = os.getenv("DASHSCOPE_API_MODEL")
DASHSCOPE_API_EMBEDDING_MODEL = os.getenv("DASHSCOPE_API_EMBEDDING_MODEL")

# DeepSeek
#   官网：https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_MODEL = os.getenv("DEEPSEEK_API_MODEL")
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


import inspect
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings




# 【不要用这个，只是方便后面return】通过LangChain获得指定平台和模型的客户端 (默认是：阿里通义千问)
def get_lc_model_client(api_key=DASHSCOPE_API_KEY,
                        base_url=DASHSCOPE_API_URL,
                        model=DASHSCOPE_API_MODEL,
                        temperature=0.7, verbose=False, debug=False):
    """
        通过LangChain获得指定平台和模型的客户端，设定的默认平台和模型为阿里百炼qwen
        也可以通过传入api_key，base_url，model三个参数来覆盖默认值
        verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：{api_key}")
    return ChatOpenAI(api_key=api_key,
                      base_url=base_url,
                      model=model,
                      temperature=temperature,
                      extra_body={"enable_thinking": False})



# 【正常使用】通过LangChain使用阿里大模型ChatTongyi
qwen = ChatTongyi(
    model='qwen-plus', # 注意这个方法只支持：qwen-plus,qwen-turbo,qwen-max,qwen-max-longcontext,qwen-vl-plus / qwen-vl-max,qwen-audio-turbo
    temperature=0.2,
    max_tokens=2048,
    stream=False,
    enable_search=False, # 禁用搜索功能
)


# 【正常使用】通过LangChain使用阿里大模型
def get_ali_model_client(model=DASHSCOPE_API_MODEL,
                         temperature=0.7, verbose=False, debug=False):
    """通过LangChain使用阿里大模型"""
    return get_lc_model_client(api_key=DASHSCOPE_API_KEY,
                               base_url=DASHSCOPE_API_URL,
                               model=model,
                               temperature=temperature,
                               verbose=verbose,
                               debug=debug)



# 【正常使用】通过LangChain使用 DeepSeek大模型
def get_ds_model_client(model=DEEPSEEK_API_MODEL,
                        temperature=0.7, verbose=False, debug=False):
    """通过LangChain使用DeepSeek大模型"""
    return get_lc_model_client(api_key=DEEPSEEK_API_KEY,
                               base_url=DEEPSEEK_API_URL,
                               model=model,
                               temperature=temperature,
                               verbose=verbose,
                               debug=debug)


# [向量化]该平台限制单次请求最多 20 条文本，需要按 BATCH_SIZE 分批调用
class BatchedDashScopeEmbeddings(DashScopeEmbeddings):
    """按小批次调用 DashScope 向量模型，避免超出平台单次批量上限"""

    def embed_documents(self, texts):
        batch_size = min(BATCH_SIZE, 20)  # 平台限制：单次最多 20 条
        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(super().embed_documents(texts[i : i + batch_size]))
        return embeddings


# [向量化]通过LangChain获得一个阿里通义千问向量化模型的实例
def get_ali_embeddings(model=DASHSCOPE_API_EMBEDDING_MODEL):
    """通过LangChain获得一个阿里通义千问向量化模型的实例"""
    return BatchedDashScopeEmbeddings(
        model=model,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

# [视觉模型]通过LangChain获得一个阿里通义千问视觉模型的实例
def get_ali_vl_model_client(model=DASHSCOPE_API_VL_MODEL,  # 使用视觉模型的环境变量
                          temperature=0.7, verbose=False, debug=False):
    """通过LangChain获得一个阿里通义千问视觉模型的实例"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_URL,
        model=model,
        temperature=temperature,
        verbose=verbose

    )