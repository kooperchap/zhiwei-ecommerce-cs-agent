import json
import logging
import sys
import time
from dialog import get_manager
from evaluation import get_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    try:
        with open("test_data/eval_cases.json", "r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception as e:
        logger.error(f"加载评测数据失败: {e}")
        return

    mgr = get_manager()
    ev = get_evaluator()
    ev.reset()

    total = len(cases)
    logger.info(f"开始评测 {total} 条数据 (使用 LLM-as-a-Judge)...")
    
    passed = 0
    
    for c in cases:
        q = c["query"]
        exp_intent = c.get("expected_intent")
        keywords = c.get("keywords", [])
        
        if not exp_intent: continue

        t0 = time.time()
        # 强制走实时 RAG
        r = mgr.process(q, f"eval_{c.get('id')}", "default", skip_cache=True, agent=True)
        lat = time.time() - t0
        
        act_intent = r.get("intent", "unknown")
        ans = r.get("answer", "")
        
        # 1. 记录延迟
        ev.record_latency(lat)
        
        # 2. 评测意图
        intent_ok = ev.eval_intent(act_intent, exp_intent)
        
        # 3. 评测检索召回 (基于关键词)
        ev.eval_retrieval(ans, keywords)
        
        # 4. LLM 裁判打分 (不再使用正则匹配分数)
        ev.eval_generation_llm(q, ans)
        
        if intent_ok: passed += 1
        
        status = "PASS" if intent_ok else "FAIL"
        logger.info(f"[{status}] Q:{q[:15]}.. | I:{act_intent} | T:{lat:.2f}s")

    rpt = ev.get_report()
    
    print("\n" + "="*50)
    print(" 智能客服评测报告 (LLM Judge)")
    print("="*50)
    print(f"样本总数: {rpt['sample_count']}")
    print(f"意图准确: {rpt['intent_accuracy']*100:.1f}% ({passed}/{total})")
    print(f"检索召回: {rpt['retrieval_recall']*100:.1f}%")
    print(f"生成质量: {rpt['generation_quality']['avg_relevance']:.1f} (LLM打分 0-100)")
    print(f"平均延迟: {rpt['avg_latency']}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
