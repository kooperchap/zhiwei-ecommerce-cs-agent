import logging
import json
import re
import hashlib
import requests
import os
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)

RESPONSE_CACHE: Dict[str, str] = {}
CACHE_MAX_SIZE = 1000

class LLMWrapper:
    def __init__(self):
        self._llm_instance = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm_instance is None:
            self._llm_instance = ChatOpenAI(
                model=settings.GEMINI_MODEL,
                api_key=settings.GEMINI_API_KEY,
                base_url=f"{settings.GEMINI_BASE_URL}/v1",
                temperature=0.1,
                max_tokens=512,
                timeout=20,
                max_retries=2
            )
        return self._llm_instance

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        return hashlib.md5(f"{prompt}:{temperature}".encode()).hexdigest()

    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        cache_key = self._get_cache_key(prompt, temperature)
        if cache_key in RESPONSE_CACHE:
            return RESPONSE_CACHE[cache_key]
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
                RESPONSE_CACHE.clear()
            RESPONSE_CACHE[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return "服务暂时不可用"

    def call_with_image(self, prompt: str, image_data: bytes) -> str:
        import base64
        try:
            b64 = base64.b64encode(image_data).decode()
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ])
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return "图片处理失败"

class DashScopeEmbedder:
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self._cache: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        payload = {
            "model": "text-embedding-v2",
            "input": {"texts": texts},
            "parameters": {"text_type": "document"}
        }
        try:
            resp = self.session.post(self.api_url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "output" in data and "embeddings" in data["output"]:
                    return [x["embedding"] for x in data["output"]["embeddings"]]
        except Exception as e:
            logger.error(f"Embedding失败: {e}")
        return [[0.0]*1536 for _ in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        if text in self._cache: return self._cache[text]
        payload = {
            "model": "text-embedding-v2",
            "input": {"texts": [text]},
            "parameters": {"text_type": "query"}
        }
        try:
            resp = self.session.post(self.api_url, json=payload, timeout=5)
            if resp.status_code == 200:
                emb = resp.json()["output"]["embeddings"][0]["embedding"]
                if len(self._cache) < 2000: self._cache[text] = emb
                return emb
        except Exception:
            pass
        return [0.0] * 1536

class LocalReranker:
    """
    本地重排序模型封装 (BAAI/bge-reranker-base)
    不需要API Key，完全本地推理
    """
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
            # 指定国内镜像下载，确保网络连通
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("正在加载本地重排序模型 BAAI/bge-reranker-base ...")
            # 首次运行会自动下载模型权重 (~1GB)
            self.model = CrossEncoder("BAAI/bge-reranker-base", device="cpu", automodel_args={"torch_dtype": "auto"})
            logger.info("本地重排序模型加载完成")
        except Exception as e:
            logger.error(f"加载本地重排序模型失败: {e}")
            self.model = None

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        if not self.model or not docs:
            return docs[:top_k]
        
        # 构造 (Query, Document) 对
        pairs = [[query, d.page_content] for d in docs]
        
        try:
            # 推理打分
            scores = self.model.predict(pairs)
            
            # 将分数与文档组合并排序
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 更新 metadata 并返回 Top K
            reranked_docs = []
            for doc, score in doc_scores[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"本地重排序推理异常: {e}")
            return docs[:top_k]

_llm_wrapper: Optional[LLMWrapper] = None
_embedder: Optional[DashScopeEmbedder] = None
_reranker: Optional[LocalReranker] = None

def get_llm() -> LLMWrapper:
    global _llm_wrapper
    if _llm_wrapper is None: _llm_wrapper = LLMWrapper()
    return _llm_wrapper

def get_embedder() -> Optional[DashScopeEmbedder]:
    global _embedder
    if _embedder is None:
        if settings.DASHSCOPE_API_KEY: _embedder = DashScopeEmbedder()
    return _embedder

def get_reranker() -> Optional[LocalReranker]:
    global _reranker
    if _reranker is None:
        _reranker = LocalReranker()
    return _reranker
