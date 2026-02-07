import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

MOCK_DB = {
    "orders": {
        "OD20240115001": {
            "status": "shipped",
            "items": ["iPhone 15 Pro"],
            "total": 8999,
            "tracking": "SF123456"
        },
        "OD20240116002": {
            "status": "pending",
            "items": ["Xiaomi 14"],
            "total": 4599,
            "tracking": ""
        }
    },
    "logistics": {
        "OD20240115001": {
            "carrier": "顺丰",
            "tracking_no": "SF123456",
            "status": "运输中",
            "trace": "已到达北京分拨中心"
        }
    },
    "stock": {
        "P001": {
            "name": "iPhone 15 Pro",
            "qty": 500,
            "price": 8999,
            "specs": "6.1英寸 A17Pro 256GB"
        },
        "P002": {
            "name": "Xiaomi 14",
            "qty": 0,
            "price": 4599,
            "specs": "6.36英寸 骁龙8Gen3 256GB"
        },
        "P003": {
            "name": "HUAWEI Mate 60",
            "qty": 200,
            "price": 6999,
            "specs": "6.69英寸 麒麟9000S 256GB"
        }
    }
}

STATUS_MAP = {"shipped": "已发货", "pending": "待发货", "delivered": "已签收"}


class SkillExecutor:
    """技能执行器，支持直接调用和Agent调用两种模式"""
    
    @staticmethod
    def query_order(order_id: str) -> str:
        order = MOCK_DB["orders"].get(order_id.upper())
        if not order:
            return f"未找到订单 {order_id}"
        status_cn = STATUS_MAP.get(order["status"], order["status"])
        result = f"订单 {order_id}: 状态[{status_cn}], 商品{order['items']}, 金额{order['total']}元"
        if order.get("tracking"):
            result += f", 快递单号 {order['tracking']}"
        return result

    @staticmethod
    def cancel_order(order_id: str) -> str:
        order = MOCK_DB["orders"].get(order_id.upper())
        if not order:
            return f"订单 {order_id} 不存在"
        if order["status"] == "shipped":
            return f"订单 {order_id} 已发货，无法直接取消，建议您拒收或申请售后"
        return f"订单 {order_id} 取消成功，退款将于1-3个工作日原路退回"

    @staticmethod
    def apply_refund(order_id: str, reason: str = "用户申请") -> str:
        order = MOCK_DB["orders"].get(order_id.upper())
        if not order:
            return f"订单 {order_id} 不存在"
        if order["status"] == "shipped":
            return f"订单 {order_id} 已发货，您可以选择拒收后申请退款，或收货后申请退货退款"
        return f"订单 {order_id} 退款申请已提交，原因：{reason}，审核通过后将退款"

    @staticmethod
    def query_logistics(order_id: str) -> str:
        order = MOCK_DB["orders"].get(order_id.upper())
        if not order:
            return f"订单 {order_id} 不存在"
        if order["status"] != "shipped":
            return f"订单 {order_id} 尚未发货，暂无物流信息"
        info = MOCK_DB["logistics"].get(order_id.upper())
        if info:
            return f"订单 {order_id} 已发货，{info['carrier']} {info['tracking_no']}: {info['status']} - {info['trace']}"
        return f"订单 {order_id} 已发货，单号 {order.get('tracking', '暂无')} 暂无轨迹更新"

    @staticmethod
    def check_stock(product_ids: str) -> str:
        pids = [p.strip().upper() for p in product_ids.split(",") if p.strip()]
        results = []
        for pid in pids:
            found = False
            for k, v in MOCK_DB["stock"].items():
                if pid == k or pid in v["name"].upper():
                    status = "有货" if v["qty"] > 0 else "缺货"
                    results.append(f"{v['name']}: 价格{v['price']}元, 库存{v['qty']}件({status})")
                    found = True
                    break
            if not found:
                results.append(f"商品 {pid} 未找到")
        return "\n".join(results) if results else "未找到相关商品"

    @staticmethod
    def compare_products(product_ids: str) -> str:
        pids = [p.strip().upper() for p in product_ids.split(",") if p.strip()]
        infos = []
        for pid in pids:
            for k, v in MOCK_DB["stock"].items():
                if pid == k or pid in v["name"].upper():
                    status = "有货" if v["qty"] > 0 else "缺货"
                    infos.append(f"- {v['name']}: {v['price']}元, 规格: {v['specs']}, {status}")
                    break
        if not infos:
            return "未找到相关商品信息"
        return "商品对比:\n" + "\n".join(infos)

    @staticmethod
    def search_knowledge(query: str) -> str:
        from rag import rag_query
        try:
            res = rag_query(query)
            return res.get("answer", "抱歉，未找到相关信息")
        except Exception as e:
            logger.error(f"知识库检索异常: {e}")
            return "系统繁忙，请稍后重试"


class FastPathRouter:
    """快速路径路由，对明确意图跳过LLM直接执行"""
    
    INTENT_SKILL_MAP = {
        "order_query": ("query_order", "order_id"),
        "order_cancel": ("cancel_order", "order_id"),
        "refund_request": ("apply_refund", "order_id"),
        "logistics_query": ("query_logistics", "order_id"),
        "stock_query": ("check_stock", "product_ids"),
        "product_compare": ("compare_products", "product_ids"),
    }
    
    @classmethod
    def can_fast_path(cls, intent: str, entities: Dict) -> bool:
        if intent not in cls.INTENT_SKILL_MAP:
            return False
        _, required_entity = cls.INTENT_SKILL_MAP[intent]
        if required_entity == "order_id":
            return "order_id" in entities
        if required_entity == "product_ids":
            return "product_id" in entities or "product_ids" in entities
        return False
    
    @classmethod
    def execute(cls, intent: str, entities: Dict) -> Optional[str]:
        if not cls.can_fast_path(intent, entities):
            return None
        
        skill_name, entity_key = cls.INTENT_SKILL_MAP[intent]
        executor = SkillExecutor()
        
        if entity_key == "order_id":
            param = entities.get("order_id", "")
        else:
            pids = entities.get("product_ids", [])
            if not pids and entities.get("product_id"):
                pids = [entities["product_id"]]
            param = ",".join(pids)
        
        skill_func = getattr(executor, skill_name, None)
        if skill_func:
            return skill_func(param)
        return None


def get_all_tools() -> List[StructuredTool]:
    executor = SkillExecutor()
    tools = [
        StructuredTool.from_function(
            func=executor.query_order,
            name="query_order",
            description="查询订单状态、金额和商品详情。输入: order_id"
        ),
        StructuredTool.from_function(
            func=executor.cancel_order,
            name="cancel_order",
            description="取消订单。输入: order_id"
        ),
        StructuredTool.from_function(
            func=executor.apply_refund,
            name="apply_refund",
            description="申请退款。输入: order_id, reason(可选)"
        ),
        StructuredTool.from_function(
            func=executor.query_logistics,
            name="query_logistics",
            description="查询物流轨迹。输入: order_id"
        ),
        StructuredTool.from_function(
            func=executor.check_stock,
            name="check_stock",
            description="查询商品库存价格。输入: product_ids(逗号分隔)"
        ),
        StructuredTool.from_function(
            func=executor.compare_products,
            name="compare_products",
            description="对比商品参数。输入: product_ids(逗号分隔)"
        ),
        StructuredTool.from_function(
            func=executor.search_knowledge,
            name="search_knowledge",
            description="检索知识库，回答运费、售后、政策、商品参数等问题。输入: query"
        ),
    ]
    return tools

