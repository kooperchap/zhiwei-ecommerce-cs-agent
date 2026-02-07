import logging
import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    ORDER_QUERY = "order_query"
    ORDER_CANCEL = "order_cancel"
    REFUND_REQUEST = "refund_request"
    LOGISTICS_QUERY = "logistics_query"
    PRODUCT_CONSULT = "product_consult"
    PRODUCT_COMPARE = "product_compare"
    STOCK_QUERY = "stock_query"
    FAQ = "faq"
    CHITCHAT = "chitchat"
    RAG = "rag"
    IMAGE_QUERY = "image_query"


FAQ_ANSWERS = {
    "你好": "您好，我是电商智能客服，请问有什么可以帮您",
    "你是谁": "我是电商智能客服助手，很高兴为您服务",
    "再见": "再见，祝您购物愉快",
    "谢谢": "不客气，还有问题随时问我",
}


CHITCHAT_PATTERNS = [
    "哈喽", "嗨", "hi", "hello", "在吗", "在不在", 
    "晚安", "早安", "嘿", "喂"
]


RAG_KEYWORDS = [
    "满多少免运费", "免运费", "包邮", "运费规则", "运费",
    "会员权益", "PLUS", "plus", "会员",
    "花呗分期", "白条分期", "分期付款", "分期",
    "七天无理由", "无理由退货", "退换货规则", "退货规则",
    "夜间配送", "配送费", "配送时间", "配送规则",
    "增值税专票", "专用发票", "电子发票", "发票",
    "退款到账", "多久到账", "到账时间", "到账周期",
    "售后返修", "返修单", "返修审核", "审核时效", "多久审核",
    "第三方卖家", "第三方售后", "卖家售后",
    "保修政策", "保修期", "保修范围",
    "续航", "抗风", "图传", "遥控器", "避障", "像素",
    "工作日", "储蓄卡", "信用卡"
]


KEYWORD_RULES = [
    (["到哪了", "物流", "快递", "发货没", "配送到"], "logistics_query"),
    (["取消订单", "不想要了取消"], "order_cancel"),
    (["我要退款", "申请退款", "退货退款", "买错", "想退"], "refund_request"),
    (["查订单", "订单状态", "订单详情", "我的订单", "查一下订单"], "order_query"),
    (["对比", "比较", "哪个好", "和.*比"], "product_compare"),
    (["有货吗", "库存", "还有吗", "缺货", "有没有货"], "stock_query"),
    (["图片", "这张图", "看看图"], "image_query"),
]


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    faq_answer: str = ""
    entities: Optional[dict] = None


class IntentRecognizer:
    def __init__(self):
        pass

    def recognize(self, query: str, context: List[str] = None, has_image: bool = False) -> IntentResult:
        query_lower = query.lower().strip()
        
        for pattern in CHITCHAT_PATTERNS:
            if pattern in query_lower or query_lower == pattern:
                return IntentResult(intent=IntentType.CHITCHAT, confidence=0.95)
        
        for faq_q, faq_a in FAQ_ANSWERS.items():
            if faq_q in query or query == faq_q:
                return IntentResult(intent=IntentType.FAQ, confidence=0.95, faq_answer=faq_a)
        
        for kw in RAG_KEYWORDS:
            if re.search(kw, query, re.IGNORECASE):
                return IntentResult(intent=IntentType.RAG, confidence=0.92, entities={})
        
        for keywords, intent_str in KEYWORD_RULES:
            for kw in keywords:
                if re.search(kw, query):
                    try:
                        return IntentResult(intent=IntentType(intent_str), confidence=0.9, entities={})
                    except ValueError:
                        pass
        
        if has_image:
            return IntentResult(intent=IntentType.IMAGE_QUERY, confidence=0.8, entities={})
        
        return IntentResult(intent=IntentType.RAG, confidence=0.7, entities={})


_recognizer: Optional[IntentRecognizer] = None


def get_recognizer() -> IntentRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = IntentRecognizer()
    return _recognizer
