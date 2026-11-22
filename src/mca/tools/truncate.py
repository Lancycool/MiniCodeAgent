"""工具输出截断。"""

from __future__ import annotations


def truncate_middle(text: str, max_chars: int, head_ratio: float = 0.5) -> tuple[str, bool]:
    """保留头尾、省略中间。返回 (结果, 是否发生截断)。

    为什么保留头尾而不是只保留开头？
    命令输出/日志的开头通常是上下文（执行了什么），结尾通常是结论（最终报错、traceback、
    测试汇总）。只截开头会把最关键的错误信息丢掉。
    省略标记里写清楚省略了多少，让模型知道"信息不完整"，必要时用更精确的方式重新获取。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head_n = int(max_chars * head_ratio)
    tail_n = max_chars - head_n
    omitted = text[head_n : len(text) - tail_n]
    marker = (
        f"\n\n[... 输出过长，已省略中间 {len(omitted)} 个字符"
        f"（约 {omitted.count(chr(10))} 行）。如需查看，请用更精确的范围/条件重新获取 ...]\n\n"
    )
    tail = text[len(text) - tail_n :] if tail_n > 0 else ""
    return text[:head_n] + marker + tail, True
