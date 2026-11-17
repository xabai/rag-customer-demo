"""
知识库质量评估 —— 轻量版（不依赖 RAGAS，自己实现 4 个核心指标）

指标：
1. Faithfulness（忠实度）—— 回答与检索到的文档是否一致
2. Answer Relevance（相关性）—— 回答是否对题
3. Context Precision（上下文精度）—— 检索到的内容多少是有用的
4. Latency（延迟）—— 端到端耗时

用法：
    from core.ragas_eval import evaluate_sample
    result = evaluate_sample(question, answer, retrieved_contexts)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass
class EvalResult:
    faithfulness: float        # 0-1
    answer_relevance: float    # 0-1
    context_precision: float   # 0-1
    latency_ms: float
    details: dict


def _tokenize(text: str) -> set[str]:
    """简单分词：按非字母数字切分 + 小写化"""
    return set(re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower()))


def faithfulness(answer: str, contexts: list[str]) -> float:
    """回答中出现的关键信息点在上下文里的覆盖率"""
    if not answer or not contexts:
        return 0.0
    ans_tokens = _tokenize(answer)
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens |= _tokenize(c)
    if not ans_tokens:
        return 0.0
    hit = ans_tokens & ctx_tokens
    return len(hit) / len(ans_tokens)


def answer_relevance(question: str, answer: str) -> float:
    """回答关键词与问题关键词的 Jaccard 相似度"""
    if not question or not answer:
        return 0.0
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    union = q_tokens | a_tokens
    inter = q_tokens & a_tokens
    return len(inter) / len(union) if union else 0.0


def context_precision(question: str, contexts: list[str]) -> float:
    """上下文内容中与问题相关的比例（简单版：每个 context 和 question 的 Jaccard 均值）"""
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    scores = []
    for c in contexts:
        c_tokens = _tokenize(c)
        if not q_tokens or not c_tokens:
            scores.append(0.0)
            continue
        inter = q_tokens & c_tokens
        union = q_tokens | c_tokens
        scores.append(len(inter) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def evaluate_sample(question: str, answer: str, contexts: list[str],
                    latency_ms: float = 0.0) -> EvalResult:
    """评估单个 Q&A 样本"""
    t0 = time.time()
    f = faithfulness(answer, contexts)
    a = answer_relevance(question, answer)
    c = context_precision(question, contexts)
    extra_ms = int((time.time() - t0) * 1000)
    return EvalResult(
        faithfulness=round(f, 3),
        answer_relevance=round(a, 3),
        context_precision=round(c, 3),
        latency_ms=round(latency_ms, 2),
        details={
            "question_len": len(question),
            "answer_len": len(answer),
            "context_count": len(contexts),
            "extra_eval_ms": extra_ms,
        },
    )


def evaluate_batch(samples: list[dict]) -> dict:
    """评估一批样本，返回聚合指标

    samples 格式: [{"question":..., "answer":..., "contexts":[...], "latency_ms":...}]
    """
    results = [evaluate_sample(**s) for s in samples]
    if not results:
        return {"count": 0}
    avg = lambda k: round(sum(getattr(r, k) for r in results) / len(results), 3)
    return {
        "count": len(results),
        "avg_faithfulness": avg("faithfulness"),
        "avg_relevance": avg("answer_relevance"),
        "avg_context_precision": avg("context_precision"),
        "avg_latency_ms": avg("latency_ms"),
        "results": [r.__dict__ for r in results],
    }
