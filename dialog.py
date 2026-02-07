import logging
import hashlib
from typing import Dict, Optional, Any
from rag import rag_query
from agent import get_agent

logger = logging.getLogger(__name__)

class DialogManager:
    def __init__(self):
        self.agent = get_agent()

    def process(self, query: str, session_id: str = "default", tenant_id: str = "default", image_data: bytes = None, skip_cache: bool = False, agent: bool = True) -> Dict[str, Any]:
        """
        核心对话处理流程
        """
        # 1. 优先处理图片 (多模态)
        if image_data:
            from llms import get_llm
            try:
                desc = get_llm().call_with_image("请提取图中所有关键文字信息，并描述商品特征。", image_data)
                query = f"{query}\n[图片内容]: {desc}"
            except Exception as e:
                logger.error(f"图片处理失败: {e}")

        # 2. Agent 模式 (包含意图识别、多轮对话、工具调用)
        if agent:
            # RAG 检索作为 Agent 的背景知识 (Context)
            # 注意：这里我们让 Agent 自己决定是否需要依赖 RAG 信息，或者直接把 RAG 结果喂给它
            rag_result = rag_query(query, tenant_id, skip_cache)
            context_str = ""
            if rag_result and rag_result.get("score", 0) > 0.3:
                # 将 RAG 检索到的文档片段整理给 Agent
                docs = [c["content"] for c in rag_result.get("context", [])]
                if docs:
                    context_str = "参考资料:\n" + "\n".join(docs)

            # 调用 Agent (传入 session_id 以支持多轮历史)
            res = self.agent.run(query, session_id=session_id, context=context_str)
            
            return {
                "answer": res.answer,
                "intent": res.intent,
                "context": rag_result.get("context", []), # 保留 RAG 上下文供前端展示
                "score": rag_result.get("score", 0),
                "type": "agent_tool" if res.intent != "agent" else "agent_chat"
            }

        # 3. 纯 RAG 模式 (降级兜底)
        return rag_query(query, tenant_id, skip_cache)

_manager: Optional[DialogManager] = None

def get_manager() -> DialogManager:
    global _manager
    if not _manager:
        _manager = DialogManager()
    return _manager
