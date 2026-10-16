import json
import time
from pprint import pprint

from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
import re

from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from unicodedata import category

from models import qwen

load_dotenv()

# set_debug(True)
# 重试机制（复用）
def call_qwen_with_retry(prompt, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):

        try:
            response = qwen.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"模型调用失败:{e},第{attempt}次重试,延迟{retry_delay}秒")
            time.sleep(retry_delay)
    return "模型暂不可用，请稍后重试"

# print(call_qwen_with_retry("你好"))

# 提取id函数
def extract_order_id(text):
    """提取订单id，优先正则，失败用AI提取，输出字典格式"""
    prompt = f"""
    你是一个电商订单id提取专家，你能从给定文本{text}中提取ORDid
    订单ID通常以ORD开头，10位数字
    请严格按JSON格式返回结果：{{"order_id":"提取结果"}}
    如果找不到订单ID，返回"Not Found"
    """
    try:
        match = re.search(r'ORD\d{10}', text)
        return {"order_id": match.group(0) if match else "Not Found" }
    except:
        # 备选方案，用AI提取订单id
        result = call_qwen_with_retry(prompt)
        return json.loads(result.strip())

# print(extract_order_id("你好，订单号为ORD1234567890"))

# 情感分类函数
def analyze_sentiment(text):
    """模型按prompt分类情感，分等级，提取关键情感句"""
    prompt = f"""
    你是一个情感分析专家
    你需要对文本{text}进行情感分析,
    要求：
    1，判断情感类型：积极，中性，消极
    2，评估置信度：(0,1)
    3，提取3个关键短语
    
    请严格按照JSON格式返回结果：
    {{
        "sentiment":"情感类型",
        "confidence":"置信度",
        "key_phrases":["短语1","短语2","短语3"],
    }}
    """
    try:
        result = call_qwen_with_retry(prompt)
        return json.loads(result.strip())
    except Exception as e:
        print(f"情感分析失败: {e}")
        return {"sentiment": "Not Found", "confidence": 0.7, "key_phrases": []}

# print(analyze_sentiment("""
#     老天爷，点了好多次还推荐给同事了好多次，结果今天吃完之后没多久上吐下泻，已经上了好多次厕所，反胃一整天。。。
# 可乐要冰也不给冰，点了牛肉酱也不给。现在就这么弄呗，一个外卖40分钟。
# """
# ))

# 问题分类函数
def classify_issue(text):
    """按prompt要求分类问题，找到最主要2类问题，返回JSON格式，无法分类就返回其他"""
    prompt = f"""
    你是一个电商问题分类专家，你要对文本{text}提到的问题分类，找到最主要2类问题
    问题分类：
    - 物流问题：配送延迟、物流损坏等
    - 产品质量：商品瑕疵、功能故障等
    - 客户服务：客服态度、响应速度等
    - 支付问题：扣款异常、退款延迟等
    - 退货退款：退货流程、退款金额等
    - 其他：无法归类的反馈
    
    要求：
    1，找到最主要的2个分类
    2，按照相关性排序
    
    请严格按照JSON格式返回结果：
    {{
        "categories":["分类1","分类2"],
    }}
    """

    try:
        result = call_qwen_with_retry(prompt)
        return json.loads(result.strip())

    except Exception as e:
        print(f"问题分类失败: {e}")
        return {"categories": ["其他"]} # 不写的话没输出可能导致崩溃

# print(classify_issue("""
# 质量不是一般的差，没穿几次下摆的线全部脱落，千千万万别买，价格死贵质量死差
# 质量太差了，洗了三次就起毛。polo衫起毛？？？？从来没有过的体验。
# """
# ))

# 评估紧急程度函数
def assess_urgency(text):
    """评估紧急程度"""
    prompt = f"""
    你是一个紧急程度评估专家，你需要评估文本{text}的紧急程度
    要求：
    1，判断紧急程度：高、中等、低
    2，评估响应时间：小时
    3，提取评估原因
    
    请严格按照JSON格式返回结果：
    {{
        "urgency":"紧急程度",
        "sla_hours":"响应时间",
        "reason":["原因1","原因2"],
    }}
    """

    try:
        result = call_qwen_with_retry(prompt)
        result = json.loads(result.strip())
        result["sla_hours"] = int(result["sla_hours"]) # 强转响应时间为数字类型
        return result

    except Exception as e:
        print(f"评估紧急程度失败: {e}")
        return {
            "urgency": "高",
            "sla_hours": 24,
            "reason": "评估失败",
        }
# print(assess_urgency("""
# 质量不是一般的差，没穿几次下摆的线全部脱落，千千万万别买，价格死贵质量死差
# 质量太差了，洗了三次就起毛。polo衫起毛？？？？从来没有过的体验。
# """
# ))

# 生成回复函数
def generate_reply(data:dict)->dict:
    """把所用函数的分析结果拼接成prompt给AI，生成回复"""
    """需要拼接的内容：原文本，订单id，问题分类，紧急程度，响应时间，情感分析，关键情感句"""
    prompt_template = """
    你是一名资深电商客服专家，根据以下内容生成回复：
    
    ### 客服反馈原文
     {feedback}
    
    ### 分析结果
    - 订单ID:{order_id}
    - 问题分类：{categories}
    - 紧急程度：{urgency}
    - 响应时间：{sla_hours}小时内回复
    - 情感分析：{sentiment}
    - 关键情感句：{key_phrases}
    
    ### 要求：
    1，根据客户反馈情感调整语气：
     - 积极反馈：语气积极，表达对客户的感谢和认可
     - 消极反馈：诚恳道歉，明确解决方案
    2，返回内容包括订单ID、问题分类 
    3，明确处理时间和后续处理步骤
    4，语气自然，要符合客户心理预期，口语化，字数100-150字
    5，结尾询问是否有其他问题
    6，不能承诺给优惠券
    
    直接输出回答，不要添加其他无关内容
    
    """

    try:
        prompt = prompt_template.format(
            feedback=data["original_feedback"],
            order_id=data["order_id"],
            categories=data["categories"],
            urgency=data["urgency"],
            sla_hours=data["sla_hours"],
            sentiment=data["sentiment"],
            key_phrases=data["key_phrases"],
        ) # 把具体数据填进prompt
        result = call_qwen_with_retry(prompt)
        return {
            "final_response": result,
            "assigned_team": data["categories"][0] if data["categories"] else "General",
            "result":data
        } # 这里返回的格式要和html一致
    except Exception as e:
        print(f"生成回复失败: {e}")
        return {
            "final_response":"感谢您的反馈，我们将尽快回答您的问题。",
            "assign_team":"General"
        }

# 构建LCEL链

extract_chain = RunnableParallel(
    order_id=RunnableLambda(extract_order_id),
    original_feedback=lambda x:x
)

analysis_chain = RunnableParallel(
    # 情感分析
    sentiment=RunnableLambda(analyze_sentiment),
    # 紧急程度
    urgency=RunnableLambda(assess_urgency),
    # 问题分类
    category=RunnableLambda(classify_issue),
)

processing_chain=(
    extract_chain
    |
    RunnablePassthrough.assign(
        analysis=lambda x:analysis_chain.invoke(x["original_feedback"])
    )
    |
    {
        "original_feedback":lambda x:x["original_feedback"],
        "order_id":lambda x:x["order_id"]["order_id"],
        "categories":lambda x:x['analysis']["category"]["categories"],
        "urgency":lambda x:x['analysis']["urgency"]["urgency"],
        "sla_hours":lambda x:x['analysis']["urgency"]["sla_hours"],
        "sentiment":lambda x:x['analysis']["sentiment"]["sentiment"],
        "key_phrases":lambda x:x['analysis']["sentiment"]["key_phrases"],
    }
    |
    RunnableLambda(generate_reply)

)

query="""
订单ORD2024071501的物流太慢了，承诺三天实际七天才到！紧急处理
"""
# print(extract_chain.invoke(query))
"""
{
'order_id': {'order_id': 'ORD2024071501'}, 
'original_feedback': '\n        "订单ORD2024071501的物流太慢了，承诺三天实际七天才到！紧急处理！",\n\n'
}

"""

# pprint((
#          extract_chain
#             |
#          RunnablePassthrough.assign(
#         analysis=lambda x:analysis_chain.invoke(x["original_feedback"]))
#      ).invoke(query))
"""
{'order_id': {'order_id': 'ORD2024071501'}, 'original_feedback': '\n        "订单ORD2024071501的物流太慢了，承诺三天实际七天才到！紧急处理！",\n\n'}
{'analysis': {'category': {'categories': ['物流问题', '客户服务']},
              'sentiment': {'confidence': 0.98,
                            'key_phrases': ['物流太慢了', '承诺三天实际七天才到', '紧急处理'],
                            'sentiment': '消极'},
              'urgency': {'reason': ["客户明确使用'紧急处理！'字样，表明情绪急迫且诉求强烈",
                                     '物流履约严重超时（承诺3天 vs 实际7天，超期133%，影响客户体验和信任',
                                     '订单号明确可追溯，问题具象、可立即介入核查物流异常原因'],
                          'sla_hours': 2,
                          'urgency': '高'}},
 'order_id': {'order_id': 'ORD2024071501'},
 'original_feedback': '\n        "订单ORD2024071501的物流太慢了，承诺三天实际七天才到！紧急处理！",\n\n'}
"""

# print(processing_chain .invoke(query))


# 部署为API服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
app = FastAPI(title="电商客服系统")


class FeedbackRequest(BaseModel):
    content: str
    user_id: str = "anonymous"

@app.get("/")
async def read_index():
    return FileResponse("index.html")

# async  awaite一起使用，前端js是类似的用法，并行获取数据
@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    try:
        start = time.time()
        result = processing_chain.invoke(request.content) # 调用LCEL链
        elapsed = time.time() - start

        return {
            "success": True,
            "processing_time": f"{elapsed:.2f}s",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

