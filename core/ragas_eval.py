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
                    latency_ms: float = 0.0, use_llm: bool = True) -> EvalResult:
    """评估单个 Q&A 样本

    use_llm=True 时，会尝试用本地 Ollama 做 LLM-as-Judge（更准确）。
    如果 Ollama 不可用，自动 fallback 到词面版（零依赖但粗糙）。
    """
    # 先尝试 LLM judge
    if use_llm:
        llm_result = _evaluate_with_llm(question, answer, contexts, latency_ms)
        if llm_result is not None:
            return llm_result

    # Fallback: 词面版
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
            "mode": "lexical_fallback",
            "question_len": len(question),
            "answer_len": len(answer),
            "context_count": len(contexts),
            "extra_eval_ms": extra_ms,
        },
    )


# ──────────────────────────────────────────────────────
# LLM-as-Judge 实现
# ──────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """你是 RAG 系统的评估专家。请根据问题、检索到的文档片段和 AI 回答，从三个维度打分（每项 0-5，再归一到 0-1）。
严格按 JSON 输出，不要额外文字：
{{"faithfulness": <0-5>, "relevance": <0-5>, "precision": <0-5>, "reason": "<一句话理由>"}}

【问题】
{question}

【检索到的文档片段】
{contexts_text}

【AI 回答】
{answer}

打分标准：
- faithfulness (忠实度): 回答中的每个关键断言是否都能从文档片段推出（5=完全不瞎说, 0=完全瞎编）
- relevance (相关性): 回答是否针对问题（5=完美切题, 0=答非所问）
- precision (精度): 检索到的文档片段里多少是真正有用的（5=每段都相关, 0=全是噪声）
"""


def _evaluate_with_llm(question: str, answer: str, contexts: list[str],
                       latency_ms: float) -> EvalResult | None:
    """尝试用 Ollama 做 LLM judge。不可用时返回 None 让调用方 fallback。"""
    try:
        from core import ollama_client
    except Exception:
        return None

    t0 = time.time()
    try:
        contexts_text = "\n\n---\n\n".join(contexts[:5])  # 最多喂 5 段，省 token
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question[:500],
            contexts_text=contexts_text[:2000],
            answer=answer[:1000],
        )
        raw = ollama_client.chat_once(
            [{"role": "system", "content": "你是严格的 RAG 评估专家，只输出 JSON。"},
             {"role": "user", "content": prompt}],
            timeout=45,
            fallback="",  # 失败返回空串而不是抛异常
        )
        if not raw or not raw.strip():
            return None

        # 从 LLM 输出里提取 JSON
        import json as _json
        import re as _re
        match = _re.search(r"\{[^{}]+\}", raw, _re.DOTALL)
        if not match:
            return None
        scores = _json.loads(match.group(0))

        def _norm(k):
            v = float(scores.get(k, 0))
            return max(0.0, min(1.0, v / 5.0))

        extra_ms = int((time.time() - t0) * 1000)
        return EvalResult(
            faithfulness=round(_norm("faithfulness"), 3),
            answer_relevance=round(_norm("relevance"), 3),
            context_precision=round(_norm("precision"), 3),
            latency_ms=round(latency_ms, 2),
            details={
                "mode": "llm_judge",
                "question_len": len(question),
                "answer_len": len(answer),
                "context_count": len(contexts),
                "judge_ms": extra_ms,
                "reason": scores.get("reason", ""),
            },
        )
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning(f"LLM judge 失败，回退词面版: {_e}")
        return None


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
