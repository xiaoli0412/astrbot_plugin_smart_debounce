"""smartdebounce —— 智能防抖插件的纯逻辑内核。

本包不 import astrbot，任何模块都可以在普通 Python 环境里直接单测。
"""

from .buffer import AddOutcome, BufferedMessage, MergeBuffer, MergePolicy
from .clock import format_clock, format_duration, now

__all__ = [
    "AddOutcome",
    "BufferedMessage",
    "MergeBuffer",
    "MergePolicy",
    "format_clock",
    "format_duration",
    "now",
]
