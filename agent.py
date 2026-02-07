import logging
from dataclasses import dataclass
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage
from config import settings
from skills import get_all_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是电商智能客服助手。

**核心原则**：
1. **槽位填充**：在调用工具前，必须检查所有必要参数（如订单号、商品ID）是否齐全。
   - 如果用户未提供必要参数，**请不要捏造**，必须反问用户获取信息。
   - 例如：用户问“查下我的订单”，你应回答“请提供您的订单号”。
2. **多轮对话**：请结合历史聊天内容理解用户意图（如“它”、“这个”指代的内容）。
3. **工具使用**：
   - 订单/库存/物流问题：必须调用对应工具。
   - 政策/售后问题：调用 search_knowledge。
4. **诚实性**：工具查不到数据时诚实告知，不要编造。

回答要求：简洁直接，使用中文。"""

INTENT_MAP = {
    "query_order": "order_query",
    "cancel_order": "order_cancel",
    "apply_refund": "refund_request",
    "query_logistics": "logistics_query",
    "check_stock": "stock_query",
    "compare_products": "product_compare",
    "search_knowledge": "rag"
}

@dataclass
class AgentResult:
    answer: str
    intent: str = "agent"

class CustomerServiceAgent:
    def __init__(self):
        self._llm = None
        self._tools = None
        self._tool_map = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(
                model=settings.GEMINI_MODEL,
                api_key=settings.GEMINI_API_KEY,
                base_url=f"{settings.GEMINI_BASE_URL}/v1",
                temperature=0.0,
                max_tokens=512,
                timeout=30
            )
        return self._llm

    @property
    def tools(self):
        if not self._tools:
            self._tools = get_all_tools()
            self._tool_map = {t.name: t for t in self._tools}
        return self._tools

    def _get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """
        获取并格式化历史消息，限制最近 10 条以节省 Token
        """
        try:
            from memory import get_memory
            # 获取最近 10 条记录
            records = get_memory().get_history(session_id, limit=10)
            messages = []
            for role, content in records:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            return messages
        except Exception as e:
            logger.warning(f"获取历史记录失败: {e}")
            return []

    def run(self, query: str, session_id: str = "default", context: str = "") -> AgentResult:
        try:
            # 1. 构造 Prompt 结构
            # SystemPrompt -> History (多轮对话核心) -> Current Context (RAG) -> User Query
            
            # 基础系统指令
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            
            # 注入历史对话 (关键修改)
            history_msgs = self._get_history_messages(session_id)
            messages.extend(history_msgs)
            
            # 注入当前轮的用户输入 (带 RAG 上下文)
            input_content = f"{context}\n用户问题: {query}" if context else query
            messages.append(HumanMessage(content=input_content))
            
            # 绑定工具
            llm_with_tools = self.llm.bind_tools(self.tools)
            
            final_answer = ""
            intent = "agent"
            
            # ReAct 循环
            for _ in range(5):
                resp = llm_with_tools.invoke(messages)
                messages.append(resp)
                
                if not resp.tool_calls:
                    final_answer = resp.content
                    break
                
                for tc in resp.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    
                    if intent == "agent":
                        intent = INTENT_MAP.get(tool_name, "agent")
                    
                    tool = self._tool_map.get(tool_name)
                    if tool:
                        logger.info(f"Agent调用工具: {tool_name} 参数: {tool_args}")
                        try:
                            result = tool.invoke(tool_args)
                        except Exception as e:
                            result = f"工具执行错误: {str(e)}"
                    else:
                        result = f"工具 {tool_name} 不存在"
                    
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            return AgentResult(answer=self._clean(final_answer), intent=intent)
            
        except Exception as e:
            logger.error(f"Agent执行错误: {e}")
            return AgentResult(answer="系统繁忙，请重试")

    def _clean(self, text: str) -> str:
        if not text: return "抱歉，无法回答"
        return text.replace("```json", "").replace("```", "").replace("**", "").strip()

_agent: Optional[CustomerServiceAgent] = None

def get_agent() -> CustomerServiceAgent:
    global _agent
    if not _agent:
        _agent = CustomerServiceAgent()
    return _agent
