# -*- coding: utf-8 -*-
"""Agent 工具注册表：内置工具 + function calling 解析"""
import math
import time

from core import ollama_client


def tool_search_knowledge(query, kb_id=None, **kwargs):
    """检索知识库：返回相关知识片段"""
    from core import rag
    context, sources, _images = rag.retrieve(query, kb_id=kb_id)
    if not context:
        return "知识库中未找到相关内容。"
    return context


def tool_calculator(expression, **kwargs):
    """计算器：计算数学表达式，仅允许安全字符"""
    allowed = set("0123456789+-*/().% ")
    if not set(expression) <= allowed:
        return "表达式包含非法字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def tool_get_current_time(**kwargs):
    """获取当前日期时间"""
    return time.strftime("当前时间：%Y-%m-%d %H:%M:%S")


def tool_get_weather(city, **kwargs):
    """查询天气（示例占位工具，可对接真实天气 API）"""
    return f"{city} 今日天气：晴，25°C（示例数据，请对接真实天气接口）"


def tool_system_info(**kwargs):
    """查询本机 CPU / 内存 / 磁盘的实时占用情况"""
    try:
        import psutil
    except ImportError:
        return "未安装 psutil 库，请先执行 pip install psutil"

    def _gb(n):
        return round(n / 1024 ** 3, 1)

    cpu_pct = psutil.cpu_percent(interval=1)
    cores_p = psutil.cpu_count(logical=False) or 0
    cores_l = psutil.cpu_count(logical=True) or 0
    mem = psutil.virtual_memory()
    # 注意：psutil 5.9.0 在部分 Windows 环境 disk_usage 会报错，改用标准库 shutil
    import shutil
    du = shutil.disk_usage("C:\\")
    disk_total, disk_free, disk_used = du.total, du.free, du.used
    disk_pct = round(disk_used / disk_total * 100, 1) if disk_total else 0

    return (
        f"本机实时状态：\n"
        f"- CPU：{cores_p} 物理核 / {cores_l} 逻辑线程，当前占用 {cpu_pct}%\n"
        f"- 内存：总 {_gb(mem.total)} GB，已用 {_gb(mem.used)} GB（{mem.percent}%），"
        f"可用 {_gb(mem.available)} GB\n"
        f"- C 盘：总 {_gb(disk_total)} GB，已用 {_gb(disk_used)} GB（{disk_pct}%），"
        f"剩余 {_gb(disk_free)} GB"
    )


def tool_gpu_info(**kwargs):
    """查询本机 NVIDIA 显卡的显存占用、利用率与温度"""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode("utf-8", "ignore").strip()
    except FileNotFoundError:
        return "本机未检测到 NVIDIA 显卡或未安装 nvidia-smi"
    except Exception as e:
        return f"读取显卡信息失败: {e}"
    if not out:
        return "未获取到显卡信息"

    lines = []
    for row in out.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 5:
            continue
        name, used, total, util, temp = parts[:5]
        try:
            pct = round(int(used) / int(total) * 100, 1) if int(total) else 0
        except ValueError:
            pct = 0
        lines.append(
            f"- {name}：显存 {used} / {total} MB（{pct}%），"
            f"GPU 利用率 {util}%，温度 {temp}°C"
        )
    return "本机显卡状态：\n" + "\n".join(lines) if lines else "未解析到显卡数据"


# 工具注册表：name -> {func, description, parameters}
TOOLS = {
    "search_knowledge": {
        "func": tool_search_knowledge,
        "label": "知识库检索",
        "description": "检索知识库中的相关内容，回答知识性问题时使用",
        "parameters": {
            "query": {"type": "string", "description": "检索关键词"},
            "kb_id": {"type": "string", "description": "知识库ID，可选"},
        },
    },
    "calculator": {
        "func": tool_calculator,
        "label": "计算器",
        "description": "计算数学表达式，用户问数学/计算/多少钱/多少数量时使用",
        "parameters": {
            "expression": {"type": "string", "description": "数学表达式，如 (100+50)*3"},
        },
    },
    "get_current_time": {
        "func": tool_get_current_time,
        "label": "当前时间",
        "description": "获取当前日期时间",
        "parameters": {},
    },
    "get_weather": {
        "func": tool_get_weather,
        "label": "天气查询",
        "description": "查询指定城市的天气",
        "parameters": {
            "city": {"type": "string", "description": "城市名称"},
        },
    },
    "system_info": {
        "func": tool_system_info,
        "label": "本机状态",
        "description": "查询本机电脑性能状态，用户问CPU、内存、磁盘、电脑卡不卡、硬件负载、剩余空间时使用",
        "parameters": {},   # 这个工具不需要参数
    },
    "gpu_info": {
        "func": tool_gpu_info,
        "label": "显卡状态",
        "description": "查询本机显卡状态，用户问显卡、GPU、显存、显卡温度、显卡占用率时使用",
        "parameters": {},   # 这个工具不需要参数
    }
}


def execute_tool(name, args):
    """执行工具调用，返回结果字符串"""
    spec = TOOLS.get(name)
    if not spec:
        return f"未知工具: {name}"
    try:
        return spec["func"](**args)
    except Exception as e:
        return f"工具执行失败: {e}"


# 零参数工具的关键词直路由：这类工具不需要从问题里抽取参数，
# 用关键词直接命中即可确定性执行，避免小模型探测不稳定、耗时长的缺点。
# 注意顺序：gpu_info 放前面，因为"电脑的显卡"这类问题会同时命中两组词
DIRECT_ROUTES = (
    ("gpu_info", ("显卡", "gpu", "显存", "显卡温度", "显卡占用", "显卡利用")),
    ("system_info", ("电脑", "卡不卡", "性能", "cpu", "内存", "磁盘", "硬盘",
                     "存储", "剩余空间", "占用率", "负载")),
    ("get_current_time", ("几点", "现在时间", "当前时间", "今天几号", "星期几", "什么日期")),
)


def route_tool(text):
    """关键词直路由：命中则返回可直接执行、无需参数的工具名，否则 None"""
    low = (text or "").lower()
    for name, kws in DIRECT_ROUTES:
        if any(k in low for k in kws):
            return name
    return None


# 需要模型从问题里抽取参数的工具（如计算表达式、城市名），才值得做一次模型探测
PROBE_KEYWORDS = (
    "计算", "算一下", "算算", "多少钱", "多少数量", "等于", "相乘", "相加", "减去", "除以",
    "天气", "气温", "下雨",
)


def should_probe(text):
    """判断是否值得为这个问题做一次模型探测（避免无谓的模型开销）"""
    low = (text or "").lower()
    return any(k in low for k in PROBE_KEYWORDS)


SYSTEM_PROMPT_SUFFIX = (
    "\n你可以使用以下工具来帮助回答：\n"
    "- search_knowledge：检索知识库\n"
    "- calculator：数学计算\n"
    "- get_current_time：当前时间\n"
    "- get_weather：查天气\n"
    "- system_info：查询本机 CPU/内存/磁盘占用\n"
    "- gpu_info：查询本机显卡显存/温度/利用率\n"
    "当需要使用工具时，在回答中用 JSON 格式输出工具调用，例如：\n"
    '{"tool": "calculator", "args": {"expression": "100*3"}}\n'
    "工具结果会以【工具结果】形式返回给你，你再基于结果给出最终回答。"
    "如果不需要工具，直接正常回答。"
)
