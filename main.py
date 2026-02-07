import logging
import os
import base64
import json
import time
import asyncio
import hashlib
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="电商智能客服")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


def get_redis_cache():
    try:
        import redis
        from config import settings
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    except Exception:
        return None


@app.get("/")
async def index():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"api": "/docs"}


@app.get("/admin")
async def admin():
    if os.path.exists("frontend/admin.html"):
        return FileResponse("frontend/admin.html")
    return {"error": "Admin page not found"}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    q = data.get("question", "")
    sid = data.get("session_id", "default")

    async def generate():
        from dialog import get_manager
        from memory import get_memory
        
        get_memory().add_message(sid, "user", q)
        
        t0 = time.time()
        r = get_manager().process(q, sid, "default", None, agent=True)
        latency = time.time() - t0
        
        ans = r.get("answer", "抱歉无法回答")
        intent = r.get("intent", "")
        path_type = r.get("type", "")
        
        yield f"data: {json.dumps({'type': 'start', 'intent': intent, 'path': path_type, 'latency': round(latency, 2)}, ensure_ascii=False)}\n\n"
        
        chunk_size = 20
        for i in range(0, len(ans), chunk_size):
            chunk = ans[i:i+chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        
        yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        
        get_memory().add_message(sid, "assistant", ans)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync(request: Request):
    try:
        data = await request.json()
        q = data.get("question", "")
        sid = data.get("session_id", "default")
        img_b64 = data.get("image_data")
        img = base64.b64decode(img_b64) if img_b64 else None
        
        cache = get_redis_cache()
        cache_key = f"chat:{hashlib.md5(q.encode()).hexdigest()}"
        
        if cache and not img:
            try:
                cached = cache.get(cache_key)
                if cached:
                    return JSONResponse(content=json.loads(cached))
            except Exception:
                pass
        
        from dialog import get_manager
        from memory import get_memory
        
        get_memory().add_message(sid, "user", q)
        
        t0 = time.time()
        r = get_manager().process(q, sid, "default", img, agent=True)
        latency = time.time() - t0
        
        ans = r.get("answer", "抱歉无法回答")
        get_memory().add_message(sid, "assistant", ans)
        
        result = {
            "answer": ans,
            "intent": r.get("intent", ""),
            "type": r.get("type", ""),
            "latency": round(latency, 2)
        }
        
        if cache and not img:
            try:
                cache.setex(cache_key, 300, json.dumps(result, ensure_ascii=False))
            except Exception:
                pass
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error("chat错误: %s", e)
        return JSONResponse(content={"answer": "服务异常", "intent": "error"})


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), category: str = Form("general")):
    from ocr_processor import process_document
    from vectorstore import add_documents
    from es_client import bulk_index
    from langchain_core.documents import Document
    
    total = 0
    for f in files:
        try:
            content = await f.read()
            r = process_document(content, f.filename, category)
            if r["status"] == "success" and r["docs"]:
                docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in r["docs"]]
                add_documents(docs)
                bulk_index([{"content": d["content"], **d["metadata"]} for d in r["docs"]])
                total += len(docs)
        except Exception as e:
            logger.error("文件处理失败 %s: %s", f.filename, e)
    
    return {"status": "success", "total_chunks": total}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/eval/report")
async def eval_report():
    from evaluation import get_evaluator
    return get_evaluator().get_report()


@app.post("/eval/run")
async def eval_run():
    """
    运行评测 (Web API 版) - 已同步更新为 LLM-as-a-Judge
    """
    from dialog import get_manager
    from evaluation import get_evaluator
    
    try:
        with open("test_data/eval_cases.json", "r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception:
        return {"error": "评测数据加载失败"}
    
    mgr = get_manager()
    ev = get_evaluator()
    ev.reset()
    
    results = []
    
    logger.info("Web端触发评测 共%d条", len(cases))
    
    for c in cases:
        q = c["query"]
        exp = c.get("expected_intent")
        keywords = c.get("keywords", [])
        
        if not exp:
            continue
        
        t0 = time.time()
        cid = c.get("id", 0)
        # 强制跳过缓存，确保评测真实性
        r = mgr.process(q, f"eval_{cid}", agent=True, skip_cache=True)
        lat = time.time() - t0
        
        ev.record_latency(lat)
        
        act = r.get("intent", "")
        ans = r.get("answer", "")
        path_type = r.get("type", "")
        
        # 1. 意图评测
        ok = ev.eval_intent(act, exp)
        
        # 2. 检索召回评测 (保留关键词匹配作为参考)
        recall = ev.eval_retrieval(ans, keywords) if keywords else 1.0
        
        # 3. 生成质量评测 (LLM Judge) - 修复点：调用新方法
        ev.eval_generation_llm(q, ans)
        
        # 为了前端展示，获取刚刚打的分数
        current_score = ev.m.gen_scores[-1] if ev.m.gen_scores else 0.0
        
        results.append({
            "id": cid,
            "query": q,
            "expected": exp,
            "actual": act,
            "correct": ok,
            "latency": round(lat, 2),
            "recall": round(recall, 2),
            "score": current_score,
            "path": path_type
        })
    
    report = ev.get_report()
    return {"results": results, "report": report}


@app.get("/tools")
async def tools():
    from skills import get_all_tools
    tool_list = get_all_tools()
    return {"tools": [{"name": t.name, "description": t.description} for t in tool_list]}
