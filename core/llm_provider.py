# -*- coding: utf-8 -*-
"""统一 LLM Provider 入口：支持 Ollama + OpenAI 兼容 + 百度文心 + 阿里通义

设计原则：
- 业务层（ollama_client、app.py）只调本模块的 chat_once / chat_stream，
  不关心底层是 Ollama 还是云端 API
- embedding 不在这里做（向量模型保持本地 Ollama 最稳，本地 bge-m3 比云端快一个数量级）
- 统一返回：纯文本（chat_once）/ 逐 token 文本流（chat_stream）
- Provider 抽象：每种 Provider 实现 _chat_once 和 _chat_stream 两个私有方法

Provider 类型：
  ollama           — 本地 Ollama（默认）
  openai_compat    — OpenAI Chat Completions 协议（覆盖 90% 云模型：
                     OpenAI、DeepSeek、MiniMax、Kimi、智谱 GLM、月之暗面、
                     阿里云百炼、硅基流动 Together、Groq 等）
  baidu_ernie      — 百度文心一言（需要 access_token 获取）
  aliyun_dashscope — 阿里通义 DashScope
"""
import json
import logging
import time

import requests

import config

_log = logging.getLogger("rag.llm")


# ============================================================
# Provider 注册表
# ============================================================

PROVIDER_REGISTRY = {}


def _register(name):
    def _wrap(cls):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return _wrap


def get_provider(name):
    """按名称获取 Provider 实例；自动注入 settings.json 里的 provider 字段"""
    from core import settings
    cfg = settings.get_llm_config()
    provider_name = name or cfg.get("llm_provider") or "ollama"
    cls = PROVIDER_REGISTRY.get(provider_name)
    if not cls:
        _log.warning("Provider %s 未注册，fallback 到 ollama", provider_name)
        cls = PROVIDER_REGISTRY["ollama"]
    return cls(cfg)


def available_providers():
    """返回可用 provider 列表（前端设置页下拉用）"""
    return [
        {"key": "ollama", "label": "Ollama（本地开源模型）", "desc": "本地部署，隐私友好，零费用"},
        {"key": "openai_compat", "label": "OpenAI 兼容（DeepSeek/MiniMax/Kimi/智谱...）",
         "desc": "90% 云模型都走这个协议，一个 API Key 通吃"},
        {"key": "baidu_ernie", "label": "百度文心一言 ERNIE", "desc": "百度智能云账号 + API Key/Secret"},
        {"key": "aliyun_dashscope", "label": "阿里通义 DashScope", "desc": "阿里云 DashScope 百炼平台"},
    ]


# ============================================================
# 统一 Provider 基类
# ============================================================

class BaseProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def chat_once(self, messages, model=None, timeout=60, images=None):
        raise NotImplementedError

    def chat_stream(self, messages, model=None, images=None):
        raise NotImplementedError

    def embed_texts(self, texts, model=None):
        """绝大多数云 provider 不支持 embedding，只有 Ollama 做。
        这里默认 raise NotImplementedError，调用方会 fallback 到 Ollama embedding。"""
        raise NotImplementedError

    @staticmethod
    def _inject_images(messages, images):
        if not images:
            return messages
        msgs = [m.copy() for m in messages]
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                msgs[i]["images"] = images
                break
        return msgs


# ============================================================
# Ollama Provider（本地默认）
# ============================================================

def _ollama_options(keep):
    """统一生成参数：抑制小模型重复循环 + 限制输出长度"""
    return {
        "keep_alive": keep,
        "temperature": getattr(config, "CHAT_TEMPERATURE", 0.7),
        "repeat_penalty": getattr(config, "CHAT_REPEAT_PENALTY", 1.3),
        "repeat_last_n": getattr(config, "CHAT_REPEAT_LAST_N", 256),
        "num_predict": getattr(config, "CHAT_NUM_PREDICT", 1024),
    }


@_register("ollama")
class OllamaProvider(BaseProvider):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.base = cfg.get("ollama_base") or config.OLLAMA_BASE

    def chat_once(self, messages, model=None, timeout=60, images=None, keep_alive=None):
        use_model = model or config.CHAT_MODEL
        keep = keep_alive if keep_alive is not None else config.KEEP_ALIVE_CHAT
        resp = requests.post(
            f"{self.base}/api/chat",
            json={
                "model": use_model,
                "messages": self._inject_images(messages, images),
                "stream": False, "think": False,
                "options": _ollama_options(keep),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def chat_stream(self, messages, model=None, images=None, keep_alive=None):
        use_model = model or config.CHAT_MODEL
        keep = keep_alive if keep_alive is not None else config.KEEP_ALIVE_CHAT
        resp = requests.post(
            f"{self.base}/api/chat",
            json={
                "model": use_model,
                "messages": self._inject_images(messages, images),
                "stream": True, "think": False,
                "options": _ollama_options(keep),
            },
            stream=True, timeout=600,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = data.get("message", {}).get("content", "")
            if content:
                yield content


# ============================================================
# OpenAI 兼容 Provider（覆盖绝大多数云模型）
# ============================================================

@_register("openai_compat")
class OpenAICompatProvider(BaseProvider):
    """OpenAI Chat Completions 协议
    支持：OpenAI、DeepSeek、MiniMax、Moonshot Kimi、智谱 GLM、阿里百炼、
          硅基流动 Together、Groq、字节豆包（火山方舟 OpenAI 兼容端点）等

    配置：
      llm_base      — API Base URL（如 https://api.deepseek.com/v1）
      llm_api_key   — API Key
      chat_model    — 模型名（如 deepseek-chat / moonshot-v1-8k / glm-4）
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base = cfg.get("llm_base") or cfg.get("openai_base") or "https://api.deepseek.com/v1"
        if not self.base.endswith("/chat/completions"):
            # 确保 base 指向 /v1 级别，自动补全
            pass
        self.api_key = cfg.get("llm_api_key") or cfg.get("openai_api_key") or ""
        self.temperature = cfg.get("llm_temperature", 0.7)

    def _url(self):
        base = self.base.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = base + "/chat/completions"
        return base

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, messages, model, stream=False, images=None):
        # OpenAI 格式的图片注入：最后一条 user message 里用 image_url
        msgs = self._inject_images_openai(messages, images)
        use_model = model or self.cfg.get("chat_model") or "deepseek-chat"
        body = {
            "model": use_model,
            "messages": msgs,
            "stream": stream,
            "temperature": self.temperature,
        }
        return body

    @staticmethod
    def _inject_images_openai(messages, images):
        if not images:
            return messages
        # OpenAI 图片格式：data:image/png;base64,XXX
        image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                          for img in images]
        msgs = [dict(m) for m in messages]
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                orig_content = msgs[i].get("content", "")
                msgs[i]["content"] = [
                    {"type": "text", "text": orig_content if isinstance(orig_content, str) else ""},
                ] + image_contents
                break
        return msgs

    def chat_once(self, messages, model=None, timeout=60, images=None):
        resp = requests.post(
            self._url(), headers=self._headers(),
            json=self._body(messages, model, stream=False, images=images),
            timeout=timeout,
        )
        if resp.status_code == 401:
            raise RuntimeError(f"API Key 无效（401），请检查 llm_api_key")
        if resp.status_code == 404:
            raise RuntimeError(f"API Endpoint 不存在（404），请检查 llm_base URL")
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def chat_stream(self, messages, model=None, images=None):
        resp = requests.post(
            self._url(), headers=self._headers(),
            json=self._body(messages, model, stream=True, images=images),
            stream=True, timeout=600,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


# ============================================================
# 百度文心一言 ERNIE
# ============================================================

@_register("baidu_ernie")
class BaiduErnieProvider(BaseProvider):
    """百度文心一言
    配置：
      llm_api_key    — API Key
      llm_base       — Secret Key（百度用 API Key + Secret Key 换 access_token）
      chat_model     — ERNIE-4.0-Turbo-8K / ERNIE-4.0-8K / ERNIE-3.5-8K
    注意：百度的 endpoint URL 里直接带 access_token，需要先换 token
    """

    TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    TOKEN_CACHE = {"token": "", "expire": 0}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.api_key = cfg.get("llm_api_key") or ""
        self.secret_key = cfg.get("llm_base") or cfg.get("openai_base") or ""

    def _get_access_token(self):
        now = time.time()
        cache = self.__class__.TOKEN_CACHE
        if cache["token"] and now < cache["expire"] - 60:
            return cache["token"]
        if not self.api_key or not self.secret_key:
            raise RuntimeError("百度文心需要配置 llm_api_key 和 llm_base（Secret Key）")
        resp = requests.post(self.TOKEN_URL, params={
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        expire = now + data.get("expires_in", 2592000)
        self.__class__.TOKEN_CACHE = {"token": token, "expire": expire}
        return token

    def _model_url(self, model):
        """百度模型 endpoint 映射"""
        m = (model or self.cfg.get("chat_model") or "ernie-4.0-turbo-8k").lower()
        # 百度标准 endpoint
        return f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{m}"

    def chat_once(self, messages, model=None, timeout=60, images=None):
        token = self._get_access_token()
        url = self._model_url(model) + f"?access_token={token}"
        # 百度需要 system 消息独立
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)
        body = {"messages": user_msgs}
        if system_msg:
            body["system"] = system_msg
        resp = requests.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error_code" in data:
            raise RuntimeError(f"百度 ERNIE 错误: {data.get('error_msg', data)}")
        return data.get("result", "").strip()

    def chat_stream(self, messages, model=None, images=None):
        token = self._get_access_token()
        url = self._model_url(model) + f"?access_token={token}"
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)
        body = {"messages": user_msgs, "stream": True}
        if system_msg:
            body["system"] = system_msg
        resp = requests.post(url, json=body, stream=True, timeout=600)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
            else:
                payload = line.strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            content = chunk.get("result", "")
            if content:
                yield content


# ============================================================
# 阿里通义 DashScope
# ============================================================

@_register("aliyun_dashscope")
class AliyunDashScopeProvider(BaseProvider):
    """阿里通义 DashScope 百炼平台
    配置：
      llm_api_key   — DashScope API Key（在阿里云百炼控制台申请）
      chat_model    — qwen-turbo / qwen-plus / qwen-max / qwen-long / qwen2.5-72b-instruct
    """
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.api_key = cfg.get("llm_api_key") or ""

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, messages, model, stream=False):
        use_model = model or self.cfg.get("chat_model") or "qwen-plus"
        return {
            "model": use_model,
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg.get("llm_temperature", 0.7),
        }

    def chat_once(self, messages, model=None, timeout=60, images=None):
        if not self.api_key:
            raise RuntimeError("阿里通义需要配置 llm_api_key")
        resp = requests.post(
            self.API_URL, headers=self._headers(),
            json=self._body(messages, model, stream=False),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "code" in data.get("message", {}):
            raise RuntimeError(f"DashScope 错误: {data['message'].get('message', data['message'])}")
        return data["choices"][0]["message"]["content"].strip()

    def chat_stream(self, messages, model=None, images=None):
        if not self.api_key:
            raise RuntimeError("阿里通义需要配置 llm_api_key")
        resp = requests.post(
            self.API_URL, headers=self._headers(),
            json=self._body(messages, model, stream=True),
            stream=True, timeout=600,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


# ============================================================
# 便捷函数：业务层统一入口
# ============================================================

def chat_once(messages, model=None, timeout=60, images=None, provider=None):
    """非流式调用，统一入口"""
    return get_provider(provider).chat_once(messages, model=model, timeout=timeout, images=images)


def chat_stream(messages, model=None, images=None, provider=None):
    """流式调用，统一入口"""
    yield from get_provider(provider).chat_stream(messages, model=model, images=images)


def test_connectivity(provider_name: str, base_url: str, api_key: str, model: str) -> dict:
    """三阶段连通性检测（和 app.py 的 test_llm endpoint 对齐）"""
    stages = []

    # 阶段 1：连接检测
    try:
        if provider_name == "ollama":
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            stages.append({"stage": "connect", "ok": r.status_code == 200,
                          "detail": f"HTTP {r.status_code}"})
            if r.status_code != 200:
                return {"ok": False, "stage": "connect", "message": f"Ollama 返回 {r.status_code}"}
            installed = [m["name"] for m in r.json().get("models", [])]
            stages.append({"stage": "install", "ok": True, "detail": f"{len(installed)} models"})
        elif provider_name in ("openai_compat", "aliyun_dashscope"):
            headers = {"Authorization": f"Bearer {api_key}"}
            # 发一个极短的请求，验证 key + endpoint
            r = requests.post(
                (base_url.rstrip("/") + "/chat/completions") if not base_url.endswith("/chat/completions") else base_url,
                headers=headers,
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=10,
            )
            if r.status_code in (200, 401, 403, 404):
                # 401/403 说明连通但 key 错；200 连通；404 endpoint 问题
                ok_connect = r.status_code != 404
                stages.append({"stage": "connect", "ok": ok_connect, "detail": f"HTTP {r.status_code}"})
                if r.status_code == 401:
                    return {"ok": False, "stage": "auth", "message": "API Key 无效（401）"}
                if r.status_code == 404:
                    return {"ok": False, "stage": "connect", "message": "Endpoint 不存在（404）"}
                if r.status_code != 200:
                    return {"ok": False, "stage": "chat",
                            "message": r.json().get("error", {}).get("message", str(r.text[:100]))}
            else:
                return {"ok": False, "stage": "connect", "message": f"HTTP {r.status_code}: {r.text[:100]}"}
        elif provider_name == "baidu_ernie":
            # 百度特殊：先用 api_key + secret_key 换 access_token
            r = requests.post(
                "https://aip.baidubce.com/oauth/2.0/token",
                params={"grant_type": "client_credentials",
                        "client_id": api_key, "client_secret": base_url},
                timeout=5,
            )
            if r.status_code != 200 or "access_token" not in r.json():
                return {"ok": False, "stage": "auth", "message": f"获取 access_token 失败: {r.text[:100]}"}
            stages.append({"stage": "auth", "ok": True, "detail": "access_token ok"})
            token = r.json()["access_token"]
            # 发一个极短请求
            r2 = requests.post(
                f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token={token}",
                json={"messages": [{"role": "user", "content": "hi"}]},
                timeout=10,
            )
            if r2.status_code != 200 or "error_code" in r2.json():
                return {"ok": False, "stage": "chat",
                        "message": r2.json().get("error_msg", r2.text[:100])}
        else:
            return {"ok": False, "stage": "connect", "message": f"未知 provider: {provider_name}"}

        # 阶段 2：实际调用（ollama 专属，云模型上面已经验证过了）
        if provider_name == "ollama":
            try:
                p = OllamaProvider({"ollama_base": base_url})
                sample = p.chat_once([{"role": "user", "content": "说一个字"}], model=model, timeout=15)
                stages.append({"stage": "chat", "ok": True, "detail": sample[:30]})
            except Exception as e:
                stages.append({"stage": "chat", "ok": False, "detail": str(e)[:100]})
                return {"ok": False, "stage": "chat", "message": str(e)}

        return {"ok": True, "stage": "chat",
                "message": "连接成功", "stages": stages,
                "installed": installed if provider_name == "ollama" else []}
    except Exception as e:
        return {"ok": False, "stage": "connect", "message": str(e)}
