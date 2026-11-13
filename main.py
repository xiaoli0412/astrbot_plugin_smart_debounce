"""智能防抖 —— 用固定时间窗合并连发消息。

为什么不是语义判断
------------------
现有 astrbot_plugin_debounce 用 ONNX 判断「这句话说完没」，实测：
    「我今天去工房了」 0.308   ← 被判"没说完"，阻塞 20 秒
    「哦」             0.077
    「嗯」             0.042
    「你在干什么」     0.935   ← 立即发
真实用户数据（platform_message_history 93 条）：
    连发间隔中位数 14s、最小 6s（7 个样本）；实际响应延迟中位数 7s。
结论：用户是「发一句，等几秒，再补一句」，第一句语义本来就完整。
判据应该是时间，不是语义。

拦截点
------
主合并点是 ``on_llm_request(priority=190)``：把合并后的文本写回 ``req.prompt``、
图片收进 ``req.image_urls``。

但 AstrBot 4.29 的 ``on_llm_request`` 在 **session lock 内部** 触发
（``internal.py``：``async with session_lock_manager.acquire_lock(...)`` 之后才
``call_event_hook(OnLLMRequestEvent)``）。若在这里阻塞等窗口，同会话的下一条
消息会卡在锁上、根本到不了本钩子，永远合并不了。因此「等窗口」必须放在
**lock 之前** 的 ``on_waiting_llm_request``：那里登记到达、吸收后续消息并
``stop_event()``，等窗口结束后再让 leader 一路走到 ``on_llm_request`` 改写 req。
"""

from __future__ import annotations

import sys
from pathlib import Path

# AstrBot 以 `data.plugins.<dir>.main` 的虚线路径加载本文件，插件目录不是顶层包。
# 把目录塞进 sys.path，让 `smartdebounce.*` 有稳定导入名。
_PLUGIN_DIR = str(Path(__file__).resolve().parent)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import asyncio  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402

from astrbot.api import AstrBotConfig, logger  # noqa: E402
from astrbot.api.event import AstrMessageEvent, filter  # noqa: E402
from astrbot.api.message_components import File, Image, Record, Video  # noqa: E402
from astrbot.api.provider import ProviderRequest  # noqa: E402
from astrbot.api.star import Context, Star, register  # noqa: E402

from smartdebounce.buffer import (  # noqa: E402
    REASON_BURST,
    REASON_MAX_WAIT,
    REASON_MEDIA,
    REASON_WINDOW,
    AddOutcome,
    BufferedMessage,
    MergeBuffer,
    MergePolicy,
)
from smartdebounce.clock import format_duration, now as clock_now  # noqa: E402

PLUGIN_NAME = "astrbot_plugin_smart_debounce"
PLUGIN_VERSION = "1.0.0"

_REASON_LABEL = {
    REASON_WINDOW: "窗口到期",
    REASON_MAX_WAIT: "硬上限",
    REASON_BURST: "条数上限",
    REASON_MEDIA: "媒体消息",
}


class _SessionState:
    """单个 umo 的运行时状态（非持久化）。"""

    __slots__ = (
        "umo",
        "buffer",
        "leader_token",
        "flush_evt",
        "timer_task",
        "reason",
        "created",
        "released",
    )

    def __init__(self, umo: str, buffer: MergeBuffer, token: str) -> None:
        self.umo = umo
        self.buffer = buffer
        self.leader_token = token
        self.flush_evt = asyncio.Event()
        self.timer_task: asyncio.Task | None = None
        self.reason = ""
        self.created = clock_now()
        self.released = False


@register(
    PLUGIN_NAME,
    "xiaoli0412",
    "固定时间窗的消息防抖：把用户连发的话合并成一次 LLM 请求，替代语义判断",
    PLUGIN_VERSION,
)
class SmartDebounce(Star):
    """智能防抖（时间窗）。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._states: dict[str, _SessionState] = {}
        self._runtime_windows: dict[str, float] = {}
        self._records: deque[dict] = deque(maxlen=10)
        self._merge_seq = 0
        # PATCHED(local): 自适应窗口用的「消息到达历史」。
        # 每个 umo 存最近若干条消息的到达时间戳（time.time()），
        # 用来判断「最近是不是经常连发」。
        self._arrivals: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ 配置

    def _cfg(self, key: str, default=None):
        try:
            val = self.config.get(key, default)
        except Exception:
            return default
        return default if val is None else val

    def _enabled(self) -> bool:
        return bool(self._cfg("enabled", True))

    def _base_policy(self) -> MergePolicy:
        return MergePolicy.from_config(self.config)

    def _adaptive_window(self, umo: str, fallback: float) -> float:
        """PATCHED(local): 按最近的连发历史挑窗口。

        返回 fallback 表示「不启用自适应 / 数据不足」。
        """
        if not bool(self._cfg("adaptive_enabled", True)):
            return fallback

        try:
            memory = float(self._cfg("adaptive_memory_seconds", 600))
            gap = float(self._cfg("adaptive_burst_gap", 15))
            short_w = float(self._cfg("adaptive_short_window", 3))
            long_w = float(self._cfg("adaptive_long_window", fallback))
        except (TypeError, ValueError):
            return fallback

        now = __import__("time").time()
        hist = [t for t in (self._arrivals.get(umo) or []) if now - t <= memory]
        self._arrivals[umo] = hist

        if len(hist) < 2:
            # 没有历史 → 用短窗（第一次聊天不该等 9 秒）
            return short_w

        # 找最近一次「间隔 < gap」的连发
        for i in range(len(hist) - 1, 0, -1):
            if hist[i] - hist[i - 1] < gap:
                return long_w
        return short_w

    def _policy_for(self, umo: str) -> MergePolicy:
        """按会话计算实际策略（临时窗口 > 会话覆盖 > 自适应 > 全局默认）。"""
        base = self._base_policy()
        window = base.window

        if umo in self._runtime_windows:
            window = self._runtime_windows[umo]
        elif self._cfg("per_session_override", False):
            overrides = self._cfg("per_session_windows", {}) or {}
            if isinstance(overrides, dict) and umo in overrides:
                try:
                    window = float(overrides[umo])
                except (TypeError, ValueError):
                    self.logger.warning(
                        "[SmartDebounce] per_session_windows[%s] 不是数字，忽略", umo
                    )
        # 只有「没被临时窗口/会话覆盖显式指定」时才走自适应
        if umo not in self._runtime_windows:
            window = self._adaptive_window(umo, window)

        return MergePolicy(
            window=window,
            max_wait=base.max_wait,
            burst_max=base.burst_max,
            flush_on_media=base.flush_on_media,
        )

    # ------------------------------------------------------------ 消息抽取

    @staticmethod
    def _is_command(event: AstrMessageEvent) -> bool:
        """判断这条消息是不是指令。

        注意：WakingCheckStage 会把 wake_prefix（默认 "/"）从 event.message_str
        里剥掉，所以必须看原始 message_obj.message_str。命令处理器被激活时
        handlers_parsed_params 也会非空，作为第二重判据。
        """
        raw = (getattr(event.message_obj, "message_str", "") or "").strip()
        if raw.startswith("/"):
            return True
        return bool(event.get_extra("handlers_parsed_params"))

    @staticmethod
    def _extract(event: AstrMessageEvent) -> BufferedMessage:
        text = (event.message_str or "").strip()
        images: list[str] = []
        is_media = False
        for comp in event.message_obj.message or []:
            if isinstance(comp, Image):
                is_media = True
                url = getattr(comp, "url", "") or getattr(comp, "file", "") or ""
                if url:
                    images.append(url)
            elif isinstance(comp, (File, Record, Video)):
                is_media = True
        return BufferedMessage(
            text=text,
            image_urls=images,
            is_media=is_media,
            arrival=clock_now(),
            message_id=str(getattr(event.message_obj, "message_id", "") or ""),
        )

    def _dlog(self, msg: str, *args) -> None:
        if self._cfg("debug_log", False):
            self.logger.info("[SmartDebounce] " + msg, *args)

    # ------------------------------------------------------- 定时器（Task）

    async def _timer_loop(self, state: _SessionState) -> None:
        """尾随窗口定时器：条件满足时置位 flush_evt。

        每 50ms 重新评估一次，既能响应「新消息延长窗口」，也能响应
        「条数上限 / 媒体立即放行」，实现上比反复重建 Task 简单可靠。
        """
        buf = state.buffer
        try:
            while not state.released:
                now = clock_now()
                flush, reason = buf.should_flush(now)
                if flush:
                    state.reason = reason
                    state.flush_evt.set()
                    return
                remaining = max(0.0, buf.next_deadline() - now)
                await asyncio.sleep(min(0.05, max(0.01, remaining)))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("[SmartDebounce] 定时器异常")
            state.reason = REASON_WINDOW
            state.flush_evt.set()

    # --------------------------------------------------- 钩子 1：到达登记

    @filter.on_waiting_llm_request(priority=190)
    async def on_arrival(self, event: AstrMessageEvent) -> None:
        """session lock 之前触发——所有并发消息都能到这里。

        - 第一条（leader）：等窗口结束，然后放行到 on_llm_request。
        - 后续消息（follower）：并入 buffer 并 stop_event，不单独触发 LLM。
        """
        if not self._enabled():
            return

        # 同一个事件理论上只应经过本钩子一次；防止 ProcessStage 的
        # activated_handlers 分支重复调用导致自己把自己当 follower 掐掉。
        if event.get_extra("_sdb_seen"):
            return
        event.set_extra("_sdb_seen", True)

        # 指令、工具轮次一律放行，不进缓冲
        if self._cfg("skip_commands", True) and self._is_command(event):
            return
        if self._cfg("skip_tool_rounds", True) and event.get_extra(
            "provider_request"
        ) is not None:
            # 别的插件已经给出 ProviderRequest（通常是工具/续写轮次）
            return

        umo = event.unified_msg_origin
        # PATCHED(local): 记录到达时间，供自适应窗口判断连发
        self._arrivals.setdefault(umo, []).append(__import__("time").time())
        msg = self._extract(event)

        # 纯空消息（无文本无媒体）不参与，避免空合并
        if not msg.text and not msg.is_media:
            return

        self._sweep()

        state = self._states.get(umo)
        if state is not None and state.released:
            # leader 已放行但 on_llm_request 尚未执行（或被别处 stop），
            # 这条消息应另起一批，不能被已经关闭的 buffer 吞掉。
            self._states.pop(umo, None)
            state = None

        if state is None:
            token = f"{umo}#{time.time_ns()}"
            buf = MergeBuffer(self._policy_for(umo))
            buf.add(msg)
            state = _SessionState(umo, buf, token)
            self._states[umo] = state
            event.set_extra("_sdb_token", token)
            state.timer_task = asyncio.create_task(
                self._timer_loop(state), name=f"sdb-timer-{umo}"
            )
            self._dlog(
                "leader 登记 umo=%s text=%r window=%.1fs",
                umo,
                msg.text[:40],
                buf.policy.window,
            )
            try:
                await state.flush_evt.wait()
            finally:
                state.released = True
                if state.timer_task and not state.timer_task.done():
                    state.timer_task.cancel()
            result = buf.snapshot(clock_now(), state.reason or REASON_WINDOW)
            event.set_extra("_sdb_merge", result)
            self._dlog(
                "leader 放行 umo=%s count=%d waited=%.1fs reason=%s",
                umo,
                result.count,
                result.waited,
                result.reason,
            )
            return

        # follower：并入，然后掐掉自己的事件
        outcome = state.buffer.add(msg)
        if outcome is AddOutcome.FOLLOWER:
            # PATCHED(local): 出现 follower = 用户正在连发 → 立即把窗口延长到长窗，
            # 让后面几条也能被抓住。（尾随窗口按最后一条 + window 计算，
            # 所以改 policy.window 立即生效。）
            try:
                _long = float(self._cfg("adaptive_long_window", 9))
                if state.buffer.policy.window < _long:
                    state.buffer.policy.window = _long
                    self._dlog("延长窗口到 %.1fs（检测到连发）", _long)
            except (TypeError, ValueError, AttributeError):
                pass
            self._dlog(
                "follower 并入 umo=%s text=%r total=%d",
                umo,
                msg.text[:40],
                state.buffer.count,
            )
            event.set_extra("_sdb_token", state.leader_token)
            event.stop_event()

    # -------------------------------------------------- 钩子 2：合并改写

    @filter.on_llm_request(priority=190)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """session lock 内部触发——把合并结果写回 req。

        ``req.prompt`` 此时已由 build_main_agent 用 **leader 自己的** 文本
        （可能已加 prompt_prefix）构造完成，所以只追加 follower 的部分，
        既得到「全部文本换行拼接」，又不会丢掉前缀配置。
        """
        if not self._enabled():
            return

        result = event.get_extra("_sdb_merge")
        if result is None:
            return
        # 立刻消费掉，避免同一事件被 pipeline 二次送入时重复追加
        event.set_extra("_sdb_merge", None)

        token = event.get_extra("_sdb_token")
        state = self._states.get(event.unified_msg_origin)
        if state is not None and token == state.leader_token:
            # 合并完成，释放该会话状态，让下一条消息成为新 leader
            state.released = True
            self._states.pop(event.unified_msg_origin, None)

        # 工具轮次不改写（func_tool 在 agent 模式下恒非空，不能作为判据；
        # 真正的工具续写轮次带 tool_calls_result）
        if self._cfg("skip_tool_rounds", True) and (
            getattr(req, "tool_calls_result", None) is not None
        ):
            self._dlog("工具续写轮次，跳过改写")
            return

        # leader 自己的文本已经在 req.prompt 里，只追加其余消息。
        # 这样既得到「全部文本换行拼接」，又不会覆盖 prompt_prefix 等配置。
        followers_text = result.followers_text
        if followers_text:
            base = req.prompt or ""
            req.prompt = f"{base}\n{followers_text}" if base else followers_text

        if result.images:
            merged = list(req.image_urls or [])
            seen = set(merged)
            for url in result.images:
                if url not in seen:
                    seen.add(url)
                    merged.append(url)
            req.image_urls = merged

        self._record(result)
        if result.count > 1:
            self.logger.info(
                "[SmartDebounce] 合并 %d 条 → 1 次 LLM 请求 | 等待 %.1fs | 原因=%s | 文本=%r",
                result.count,
                result.waited,
                _REASON_LABEL.get(result.reason, result.reason),
                (req.prompt or "")[:120],
            )
        else:
            self._dlog(
                "单条放行 count=1 waited=%.1fs reason=%s",
                result.waited,
                result.reason,
            )

    # -------------------------------------------------------------- 记录

    def _record(self, result) -> None:
        self._merge_seq += 1
        self._records.append(
            {
                "seq": self._merge_seq,
                "at": time.time(),
                "count": result.count,
                "waited": result.waited,
                "reason": result.reason,
                "preview": (result.text or "").replace("\n", " / ")[:60],
            }
        )

    def _sweep(self) -> None:
        """清理长时间未释放的僵尸状态（leader 事件被别处 stop 时兜底）。"""
        if not self._states:
            return
        now = clock_now()
        for umo, state in list(self._states.items()):
            ttl = state.buffer.policy.max_wait + 120.0
            if state.released or (now - state.created) > ttl:
                if state.timer_task and not state.timer_task.done():
                    state.timer_task.cancel()
                self._states.pop(umo, None)
                self.logger.warning("[SmartDebounce] 清理僵尸缓冲 umo=%s", umo)

    # -------------------------------------------------------------- 指令

    @filter.command("防抖")
    async def cmd_debounce(self, event: AstrMessageEvent):
        """查看 / 调整防抖配置。用法：/防抖 | /防抖 窗口 12 | /防抖 状态"""
        raw = (event.message_str or "").strip()
        tokens = raw.split()
        # 去掉可能残留的指令名
        if tokens and tokens[0] in ("防抖",):
            tokens = tokens[1:]
        elif tokens and tokens[0].startswith("/防抖"):
            tokens = tokens[1:]

        if tokens and tokens[0] == "窗口":
            if len(tokens) < 2:
                yield event.plain_result("用法：/防抖 窗口 12（单位秒，0 表示不等待）")
                return
            try:
                window = float(tokens[1])
            except ValueError:
                yield event.plain_result(f"窗口值不是数字：{tokens[1]}")
                return
            if window < 0 or window > 300:
                yield event.plain_result("窗口需在 0–300 秒之间")
                return
            umo = event.unified_msg_origin
            self._runtime_windows[umo] = window
            self.logger.info(
                "[SmartDebounce] 临时窗口已改 umo=%s → %.1fs（未写配置文件）", umo, window
            )
            yield event.plain_result(
                f"本会话合并窗口已临时改为 {window:g}s（仅内存，重启失效）"
            )
            return

        if tokens and tokens[0] == "状态":
            yield event.plain_result(self._render_records())
            return

        yield event.plain_result(self._render_config(event.unified_msg_origin))

    def _render_config(self, umo: str) -> str:
        p = self._policy_for(umo)
        max_wait_text = "不限" if p.max_wait <= 0 else f"{p.max_wait:g}s（第一条起算）"
        lines = [
            f"智能防抖 v{PLUGIN_VERSION}",
            f"开关：{'开启' if self._enabled() else '关闭'}",
            f"合并窗口：{p.window:g}s（尾随，最后一条起算）",
            f"硬上限：{max_wait_text}",
            f"连发上限：{p.burst_max} 条",
            f"媒体立即放行：{'是' if p.flush_on_media else '否'}",
            f"跳过指令：{'是' if self._cfg('skip_commands', True) else '否'}"
            f" | 跳过工具轮次：{'是' if self._cfg('skip_tool_rounds', True) else '否'}",
        ]
        if umo in self._runtime_windows:
            lines.append(f"本会临时窗口：{self._runtime_windows[umo]:g}s")
        lines.append(f"活跃缓冲：{len(self._states)} 个会话")
        now = clock_now()
        for st in list(self._states.values())[:5]:
            lines.append(
                f"  · {st.umo}：{st.buffer.count} 条，已等 "
                f"{format_duration(now - st.created)}"
            )
        return "\n".join(lines)

    def _render_records(self) -> str:
        if not self._records:
            return "最近还没有发生合并。"
        lines = ["最近 10 次合并："]
        for rec in reversed(self._records):
            lines.append(
                f"#{rec['seq']} {time.strftime('%H:%M:%S', time.localtime(rec['at']))} "
                f"{rec['count']} 条 / 等 {rec['waited']:.1f}s / "
                f"{_REASON_LABEL.get(rec['reason'], rec['reason'])}\n"
                f"    {rec['preview']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------ WebUI（PATCHED local）
    async def initialize(self) -> None:
        """注册 WebUI 防抖面板所需 API。"""
        try:
            reg = getattr(self.context, "register_web_api", None)
            if not reg:
                return
            reg(
                "astrbot_plugin_smart_debounce/overview",
                self.api_overview,
                ["GET"],
                "防抖：配置 + 活跃缓冲 + 最近合并",
            )
            reg(
                "astrbot_plugin_smart_debounce/clear",
                self.api_clear,
                ["POST"],
                "防抖：清空统计记录",
            )
            self.logger.info("[SmartDebounce] WebUI API 已注册 2 条")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("[SmartDebounce] WebUI API 注册失败：%s", exc)

    async def api_overview(self):
        """一次性返回面板需要的全部数据。"""
        base = self._base_policy()
        sessions = []
        for umo, st in list(self._states.items()):
            try:
                sessions.append({
                    "umo": umo,
                    "count": st.buffer.count,
                    "waited": round(clock_now() - st.created, 1),
                    "soft_deadline": round(st.buffer.soft_deadline(), 1),
                    "hard_deadline": round(st.buffer.hard_deadline(), 1),
                    "preview": (st.buffer.merged_text() or "")[:80],
                })
            except Exception:
                continue

        records = []
        for r in list(self._records):
            records.append({
                "seq": r.get("seq"),
                "at": time.strftime("%H:%M:%S", time.localtime(r.get("at", 0))),
                "count": r.get("count"),
                "waited": round(float(r.get("waited") or 0), 1),
                "reason": _REASON_LABEL.get(r.get("reason"), r.get("reason")),
                "preview": r.get("preview"),
            })

        return {
            "ok": True,
            "data": {
                "config": {
                    "enabled": self._enabled(),
                    "merge_window_seconds": base.window,
                    "max_wait_seconds": base.max_wait,
                    "burst_max_messages": base.burst_max,
                    "flush_on_media": base.flush_on_media,
                },
                "sessions": sessions,
                "records": records,
                "total_merges": self._merge_seq,
                "runtime_windows": dict(self._runtime_windows),
            },
        }

    async def api_clear(self):
        self._records.clear()
        self._merge_seq = 0
        return {"ok": True, "data": {"cleared": True}}

    async def terminate(self) -> None:
        """插件卸载时取消所有定时器。"""
        for state in list(self._states.values()):
            state.released = True
            if state.timer_task and not state.timer_task.done():
                state.timer_task.cancel()
        self._states.clear()
        self.logger.info("[SmartDebounce] 已卸载，缓冲已清空")
