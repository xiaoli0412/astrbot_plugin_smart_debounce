"""缓冲核心——纯逻辑，不依赖 astrbot、不依赖 asyncio、不依赖事件对象。

设计要点
--------
判据是**时间**，不是语义。真实数据：用户连发间隔中位数 14s、最小 6s，
实际响应延迟中位数 7s。所以「等 N 秒，期间的新消息全部累积，然后一次性发」
比「判断这句话说完没」更贴近真实行为。

本模块只负责三件事：
1. 把消息按到达顺序存起来；
2. 在给定时刻回答「现在该不该放行、为什么」；
3. 给出合并后的文本与图片列表。

定时器、事件对象、session lock 全部由 main.py 负责，便于单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# flush 原因，用于日志与 /防抖 状态展示
REASON_WINDOW = "window"
REASON_MAX_WAIT = "max_wait"
REASON_BURST = "burst"
REASON_MEDIA = "media"


class AddOutcome(str, Enum):
    """一条消息加入缓冲区后的角色。"""

    LEADER = "leader"
    """缓冲区为空时的第一条：它负责等待窗口，其它消息并入它。"""

    FOLLOWER = "follower"
    """非第一条：并入 leader，自身事件应 stop_event。"""


@dataclass
class MergePolicy:
    """一次合并的判定参数（秒）。"""

    window: float = 9.0
    max_wait: float = 20.0
    burst_max: int = 6
    flush_on_media: bool = True

    def __post_init__(self) -> None:
        self.window = max(0.0, float(self.window))
        # max_wait 是「从第一条起算」的绝对硬上限，允许小于 window：
        # 小于时等价于「不管后面来多少条，到点必发」。
        self.max_wait = max(0.0, float(self.max_wait))
        self.burst_max = max(1, int(self.burst_max))

    @classmethod
    def from_config(cls, cfg) -> MergePolicy:
        """从 AstrBot 配置字典构造。缺失/非法值一律回退默认。"""

        def _num(key: str, default: float) -> float:
            try:
                val = cfg.get(key, default)
                return float(default if val is None else val)
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int) -> int:
            try:
                val = cfg.get(key, default)
                return int(default if val is None else val)
            except (TypeError, ValueError):
                return default

        return cls(
            window=_num("merge_window_seconds", 9.0),
            max_wait=_num("max_wait_seconds", 20.0),
            burst_max=_int("burst_max_messages", 6),
            flush_on_media=bool(cfg.get("flush_on_media", True)),
        )


@dataclass
class BufferedMessage:
    """缓冲区里的一条原始消息。"""

    text: str = ""
    image_urls: list[str] = field(default_factory=list)
    is_media: bool = False
    """是否含图片/文件/语音/视频等媒体段。"""

    arrival: float = 0.0
    """到达时刻（单调时钟）。"""

    message_id: str = ""
    """平台消息 id，仅用于日志排查。"""


@dataclass
class MergeResult:
    """一次合并快照，交给 on_llm_request 写回 req。"""

    text: str
    """全部文本按到达顺序换行拼接（用于展示/校验）。"""

    leader_text: str
    """第一条的文本。``req.prompt`` 此时已由它构造，追加时要跳过。"""

    followers_text: str
    """第二条及以后的文本换行拼接——真正要追加到 ``req.prompt`` 的部分。"""

    images: list[str]
    count: int
    waited: float
    reason: str
    first_arrival: float
    last_arrival: float


class MergeBuffer:
    """单个会话的合并缓冲区。

    用法::

        buf = MergeBuffer(policy)
        outcome = buf.add(msg, now)
        if outcome is AddOutcome.LEADER:
            ...  # 等待 buf.next_deadline()
        if buf.should_flush(now)[0]:
            result = buf.snapshot(now)
    """

    def __init__(self, policy: MergePolicy | None = None) -> None:
        self.policy = policy or MergePolicy()
        self._messages: list[BufferedMessage] = []

    # ---------- 只读属性 ----------

    @property
    def messages(self) -> list[BufferedMessage]:
        return self._messages

    @property
    def count(self) -> int:
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        return not self._messages

    @property
    def first_arrival(self) -> float:
        return self._messages[0].arrival if self._messages else 0.0

    @property
    def last_arrival(self) -> float:
        return self._messages[-1].arrival if self._messages else 0.0

    # ---------- 写入 ----------

    def add(self, message: BufferedMessage) -> AddOutcome:
        """追加一条消息，返回它在这一批里的角色。"""
        if not self._messages:
            self._messages.append(message)
            return AddOutcome.LEADER
        self._messages.append(message)
        return AddOutcome.FOLLOWER

    def reset(self) -> None:
        """清空缓冲区（放行后必须调用，避免下一批被污染）。"""
        self._messages.clear()

    # ---------- 时间判定 ----------

    def soft_deadline(self) -> float:
        """尾随窗口：最后一条到达时间 + window。"""
        return self.last_arrival + self.policy.window

    def hard_deadline(self) -> float:
        """硬上限：第一条到达时间 + max_wait。

        ``max_wait <= 0`` 视为「不设硬上限」（返回 inf），与旧插件「0 表示不超时」
        的习惯保持一致。
        """
        if self.policy.max_wait <= 0:
            return float("inf")
        return self.first_arrival + self.policy.max_wait

    def next_deadline(self) -> float:
        """下一次必须重新评估的时刻（取两者较早者）。"""
        return min(self.soft_deadline(), self.hard_deadline())

    def should_flush(self, now: float) -> tuple[bool, str]:
        """当前时刻是否应当放行，以及原因。"""
        if not self._messages:
            return False, ""

        # 硬上限优先级最高：到点必发，不论后面还在不在补充
        if now >= self.hard_deadline():
            return True, REASON_MAX_WAIT

        # 攒够条数立刻发——避免刷屏时无限延后
        if self.count >= self.policy.burst_max:
            return True, REASON_BURST

        # 媒体消息视为「发完了」——图文混发的常见模式
        if self.policy.flush_on_media and self._messages[-1].is_media:
            return True, REASON_MEDIA

        if now >= self.soft_deadline():
            return True, REASON_WINDOW

        return False, ""

    # ---------- 合并产物 ----------

    def merged_text(self) -> str:
        """所有文本按到达顺序用换行拼接（空文本跳过）。"""
        parts = [m.text.strip() for m in self._messages if m.text and m.text.strip()]
        return "\n".join(parts)

    def leader_text(self) -> str:
        """第一条消息的文本（``req.prompt`` 已包含它）。"""
        if not self._messages:
            return ""
        return (self._messages[0].text or "").strip()

    def followers_text(self) -> str:
        """第二条及以后的文本拼接——追加到 ``req.prompt`` 的部分。"""
        parts = [
            (m.text or "").strip()
            for m in self._messages[1:]
            if m.text and m.text.strip()
        ]
        return "\n".join(parts)

    def merged_images(self) -> list[str]:
        """所有图片 URL 去重（保序）。"""
        seen: set[str] = set()
        out: list[str] = []
        for msg in self._messages:
            for url in msg.image_urls:
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
        return out

    def snapshot(self, now: float, reason: str = "") -> MergeResult:
        """生成合并快照（不自动清空，调用方决定何时 reset）。"""
        if not reason:
            reason = self.should_flush(now)[1]
        waited = max(0.0, now - self.first_arrival) if self._messages else 0.0
        return MergeResult(
            text=self.merged_text(),
            leader_text=self.leader_text(),
            followers_text=self.followers_text(),
            images=self.merged_images(),
            count=self.count,
            waited=waited,
            reason=reason,
            first_arrival=self.first_arrival,
            last_arrival=self.last_arrival,
        )
