"""时间辅助——纯标准库，便于单测替换。"""

from __future__ import annotations

import time


def now() -> float:
    """单调时钟，避免系统时间跳变影响窗口计算。"""
    return time.monotonic()


def format_duration(seconds: float) -> str:
    """把秒数格式化成人类可读的短串。"""
    if seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m{rest:.0f}s"


def format_clock(ts: float) -> str:
    """把 Unix 时间戳格式化为 HH:MM:SS，仅用于展示。"""
    return time.strftime("%H:%M:%S", time.localtime(ts))
