"""buffer.py 的单元测试——不依赖 astrbot、不需要事件对象、不需要联网。

运行：  python3 -m unittest discover -s tests -v
或：    python3 tests/test_buffer.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def msg(text="", images=None, media=False, at=0.0, mid=""):
    return BufferedMessage(
        text=text,
        image_urls=list(images or []),
        is_media=media,
        arrival=at,
        message_id=mid,
    )


class TestPolicy(unittest.TestCase):
    def test_defaults(self):
        p = MergePolicy()
        self.assertEqual(p.window, 9.0)
        self.assertEqual(p.max_wait, 20.0)
        self.assertEqual(p.burst_max, 6)
        self.assertTrue(p.flush_on_media)

    def test_max_wait_may_be_below_window(self):
        """max_wait 是硬上限，允许配得比 window 小（等价于到点必发）。"""
        p = MergePolicy(window=30, max_wait=5)
        self.assertEqual(p.max_wait, 5)
        buf = MergeBuffer(p)
        buf.add(msg("a", at=0))
        self.assertTrue(buf.should_flush(5.0)[0])
        self.assertEqual(buf.should_flush(5.0)[1], REASON_MAX_WAIT)

    def test_from_config_missing_keys_use_defaults(self):
        p = MergePolicy.from_config({})
        self.assertEqual((p.window, p.max_wait, p.burst_max), (9.0, 20.0, 6))

    def test_from_config_bad_types_fall_back(self):
        p = MergePolicy.from_config(
            {"merge_window_seconds": "abc", "burst_max_messages": None}
        )
        self.assertEqual(p.window, 9.0)
        self.assertEqual(p.burst_max, 6)


class TestMergeBuffer(unittest.TestCase):
    def test_first_is_leader_rest_are_followers(self):
        buf = MergeBuffer(MergePolicy())
        self.assertIs(buf.add(msg("a", at=0)), AddOutcome.LEADER)
        self.assertIs(buf.add(msg("b", at=1)), AddOutcome.FOLLOWER)
        self.assertIs(buf.add(msg("c", at=2)), AddOutcome.FOLLOWER)
        self.assertEqual(buf.count, 3)

    def test_merged_text_joins_with_newline(self):
        buf = MergeBuffer()
        buf.add(msg("我今天去工房了", at=0))
        buf.add(msg("哦", at=1))
        buf.add(msg("你在干什么", at=2))
        self.assertEqual(buf.merged_text(), "我今天去工房了\n哦\n你在干什么")

    def test_merged_text_skips_blank(self):
        buf = MergeBuffer()
        buf.add(msg("a", at=0))
        buf.add(msg("   ", at=1))
        buf.add(msg("b", at=2))
        self.assertEqual(buf.merged_text(), "a\nb")
        self.assertEqual(buf.followers_text(), "b")

    def test_leader_and_followers_split(self):
        buf = MergeBuffer()
        buf.add(msg("第一句", at=0))
        buf.add(msg("第二句", at=1))
        self.assertEqual(buf.leader_text(), "第一句")
        self.assertEqual(buf.followers_text(), "第二句")

    def test_followers_text_empty_for_single_message(self):
        buf = MergeBuffer()
        buf.add(msg("只有一句", at=0))
        self.assertEqual(buf.followers_text(), "")

    def test_images_dedup_and_keep_order(self):
        buf = MergeBuffer()
        buf.add(msg("a", images=["u1", "u2"], at=0))
        buf.add(msg("b", images=["u2", "u3"], at=1))
        self.assertEqual(buf.merged_images(), ["u1", "u2", "u3"])

    def test_reset(self):
        buf = MergeBuffer()
        buf.add(msg("a", at=0))
        buf.reset()
        self.assertTrue(buf.is_empty)
        self.assertEqual(buf.count, 0)
        self.assertEqual(buf.merged_text(), "")


class TestTiming(unittest.TestCase):
    def setUp(self):
        self.policy = MergePolicy(window=9, max_wait=20, burst_max=6, flush_on_media=True)

    def test_no_flush_before_window(self):
        buf = MergeBuffer(self.policy)
        buf.add(msg("a", at=100))
        self.assertEqual(buf.should_flush(108.9), (False, ""))

    def test_flush_at_window_boundary(self):
        buf = MergeBuffer(self.policy)
        buf.add(msg("a", at=100))
        self.assertEqual(buf.should_flush(109.0), (True, REASON_WINDOW))

    def test_trailing_window_extends_with_new_message(self):
        buf = MergeBuffer(self.policy)
        buf.add(msg("a", at=100))
        buf.add(msg("b", at=105))  # 尾随窗口应顺延到 114
        self.assertFalse(buf.should_flush(109)[0])
        self.assertTrue(buf.should_flush(114.0)[0])
        self.assertEqual(buf.should_flush(114.0)[1], REASON_WINDOW)

    def test_hard_cap_wins_over_trailing_window(self):
        buf = MergeBuffer(self.policy)
        buf.add(msg("a", at=100))
        # 一直有新消息把尾随窗口往后推
        for i in range(1, 5):
            buf.add(msg(f"m{i}", at=100 + i * 4))
        # 最后一条在 116，尾随到 125；但硬上限是 120
        self.assertEqual(buf.next_deadline(), 120.0)
        flush, reason = buf.should_flush(120.0)
        self.assertTrue(flush)
        self.assertEqual(reason, REASON_MAX_WAIT)

    def test_burst_limit_flushes_immediately(self):
        buf = MergeBuffer(MergePolicy(window=9, max_wait=20, burst_max=3))
        buf.add(msg("a", at=0))
        buf.add(msg("b", at=0.1))
        self.assertFalse(buf.should_flush(0.2)[0])
        buf.add(msg("c", at=0.2))
        flush, reason = buf.should_flush(0.2)
        self.assertTrue(flush)
        self.assertEqual(reason, REASON_BURST)

    def test_media_flushes_immediately(self):
        buf = MergeBuffer(self.policy)
        buf.add(msg("看图", images=["u1"], media=True, at=0))
        flush, reason = buf.should_flush(0.0)
        self.assertTrue(flush)
        self.assertEqual(reason, REASON_MEDIA)

    def test_media_ignored_when_flush_on_media_false(self):
        buf = MergeBuffer(
            MergePolicy(window=9, max_wait=20, burst_max=6, flush_on_media=False)
        )
        buf.add(msg("看图", images=["u1"], media=True, at=0))
        self.assertFalse(buf.should_flush(0.0)[0])

    def test_zero_window_flushes_immediately(self):
        buf = MergeBuffer(MergePolicy(window=0, max_wait=0.001, burst_max=6))
        buf.add(msg("a", at=0))
        self.assertTrue(buf.should_flush(0.0)[0])

    def test_zero_max_wait_means_no_hard_cap(self):
        """max_wait=0 视为不设硬上限（对齐旧插件「0 表示不超时」）。"""
        buf = MergeBuffer(MergePolicy(window=9, max_wait=0))
        buf.add(msg("a", at=0))
        self.assertEqual(buf.hard_deadline(), float("inf"))
        self.assertFalse(buf.should_flush(8.9)[0])
        self.assertTrue(buf.should_flush(9.0)[0])

    def test_empty_buffer_never_flushes(self):
        buf = MergeBuffer(self.policy)
        self.assertEqual(buf.should_flush(999), (False, ""))


class TestSnapshot(unittest.TestCase):
    def test_snapshot_fields(self):
        buf = MergeBuffer(MergePolicy(window=9, max_wait=20))
        buf.add(msg("a", images=["u1"], at=100))
        buf.add(msg("b", images=["u2"], at=103))
        snap = buf.snapshot(109.5, REASON_WINDOW)
        self.assertEqual(snap.count, 2)
        self.assertEqual(snap.text, "a\nb")
        self.assertEqual(snap.leader_text, "a")
        self.assertEqual(snap.followers_text, "b")
        self.assertEqual(snap.images, ["u1", "u2"])
        self.assertAlmostEqual(snap.waited, 9.5)
        self.assertEqual(snap.reason, REASON_WINDOW)

    def test_snapshot_infers_reason(self):
        buf = MergeBuffer(MergePolicy(window=9, max_wait=20))
        buf.add(msg("a", at=0))
        snap = buf.snapshot(9.0)
        self.assertEqual(snap.reason, REASON_WINDOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
