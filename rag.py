import logging
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from llms import get_llm, get_reranker
from config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8)

FAQ_PATTERNS = [
    (r"(京东)?自营.*满.*运费|自营.*免运费", "京东自营商品满59元免基础运费，不满59元收6元。PLUS会员无限免邮。"),
    (r"第三方.*满.*运费|第三方.*包邮", "第三方商家满99元免运费，不足收6元。"),
    (r"退款.*到账|多久.*到账", "微信1小时内，储蓄卡1-7工作日，信用卡1-15工作日。"),
    (r"七天无理由|无理由退货", "商品支持七天无理由退货（需商品完好）。"),
    (r"续航.*时间|飞行.*时间", "Mini 4 Pro续航：智能电池34分钟，长续航电池45分钟。")
]

def get_redis_cache():
    try:
        import redis
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=1.0
        )
    except Exception:
        return None

def cache_key(query: str) -> str:
    return "rag_v6:" + hashlib.md5(query.encode()).hexdigest()

def _vector_search(query: str, tenant_id: str, k: int) -> List[Document]:
    from vectorstore import search as vec_search
    try:
        return vec_search(query, k=k, tenant_id=tenant_id)
    except Exception:
        return []

def _es_search(query: str, tenant_id: str, k: int) -> List[Document]:
    from es_client import search as es_search
    try:
        return es_search(query, k=k, tenant_id=tenant_id)
    except Exception:
        return []

def reciprocal_rank_fusion(results: Dict[str, List[Document]], k_param: int = 60) -> List[Document]:
    """
    RRF 粗排算法
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    weights = {"vector": 1.0, "es": 1.5}
    
    for source, docs in results.items():
        weight = weights.get(source, 1.0)
        for rank, doc in enumerate(docs):
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in doc_map:
                doc_map[h] = doc
            scores[h] = scores.get(h, 0.0) + weight * (1.0 / (k_param + rank + 1))
    
    sorted_hashes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[h] for h in sorted_hashes]

def retrieve_pipeline(query: str, tenant_id: str) -> List[Document]:
    # 1. 并行多路召回 (Recall K=15)
    future_vec = _executor.submit(_vector_search, query, tenant_id, 15)
    future_es = _executor.submit(_es_search, query, tenant_id, 15)
    
    vec_docs = future_vec.result(timeout=2) or []
    es_docs = future_es.result(timeout=2) or []
        
    if not vec_docs and not es_docs:
        return []
        
    # 2. 粗排 (RRF) -> Top 10
    coarse_docs = reciprocal_rank_fusion({"vector": vec_docs, "es": es_docs})[:10]
    
    # 3. 精排 (Local BGE Rerank) -> Top 3
    # 这里的 reranker 是本地加载的 CrossEncoder
    reranker = get_reranker()
    if reranker:
        final_docs = reranker.rerank(query, coarse_docs, top_k=3)
    else:
        final_docs = coarse_docs[:3]
    
    return final_docs

def generate_answer(query: str, docs: List[Document], llm) -> Tuple[str, float]:
    if not docs:
        return "抱歉，未找到相关信息。", 0.0

    ctx_str = "\n".join([f"{i+1}. {d.page_content[:300]}" for i, d in enumerate(docs)])
    
    prompt = f"""基于以下资料回答问题。
资料:
{ctx_str}
问题: {query}
要求: 简洁准确，基于资料。"""

    try:
        answer = llm.call(prompt, temperature=0.1, max_tokens=200)
        return answer, 0.95
    except Exception:
        return "系统繁忙", 0.0

def rag_query(query: str, tenant_id: str = "default", skip_cache: bool = False) -> Dict:
    ck = cache_key(query)
    redis_cache = get_redis_cache()

    if not skip_cache and redis_cache:
        cached = redis_cache.get(ck)
        if cached:
            return json.loads(cached)

    # 1. FAQ 极速匹配
    for pattern, ans in FAQ_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return {"answer": ans, "context": [], "score": 1.0}

    # 2. RAG Pipeline
    llm = get_llm()
    final_docs = retrieve_pipeline(query, tenant_id)
    
    answer, score = generate_answer(query, final_docs, llm)
    
    result = {
        "answer": answer,
        "context": [{"content": d.page_content[:50]} for d in final_docs],
        "score": score
    }

    if not skip_cache and redis_cache and score > 0.6:
        redis_cache.setex(ck, 600, json.dumps(result))
        
    return result
