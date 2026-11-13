# 智能防抖（astrbot_plugin_smart_debounce）

用**固定时间窗**合并用户连发的消息，让「发一句 → 等几秒 → 再补一句」只触发一次 LLM 请求。

## 为什么做这个插件

现有 `astrbot_plugin_debounce` 用 ONNX 模型判断「这句话说完没」，阈值 0.75。实测分数：

| 消息 | 语义模型得分 | 结果 |
| --- | --- | --- |
| 我今天去工房了 | 0.308 | 判为「没说完」，阻塞 20 秒 |
| 哦 | 0.077 | 阻塞 |
| 嗯 | 0.042 | 阻塞 |
| 你在干什么 | 0.935 | 立即发 |
| 在吗 | 0.932 | 立即发 |

问题很明显：**判据选错了维度**。「我今天去工房了」语义上是完整的一句话，用户只是习惯隔几秒再补一句；而「哦」「嗯」这种看似不完整的话，用户根本不会再补。

从 `platform_message_history` 统计 93 条真实消息得到的事实：

- 用户连发间隔：**中位数 14 秒，最小 6 秒**（7 个样本）
- 实际响应延迟：中位数 7 秒
- bot 自己会分段回复（+2s、+3s 各一条），那是正常行为，不该被防抖吃掉

所以本插件改用**时间**判定：收到消息后固定等 N 秒，期间的新消息全部累积，然后一次性发。第一句本来就完整，等的是「还有没有下一句」，不是「这句完不完整」。

## 合并行为

```
用户: 我今天去工房了        ← t=0    成为 leader，开始等窗口
用户: 哦                    ← t=6    并入 buffer，事件被 stop，不单独发 LLM
用户: 在吗                  ← t=8    并入 buffer（窗口顺延到最后一条 + 9s）
                              t=17   放行：一次 LLM 请求，prompt 含三句话
```

- **尾随窗口**：每来一条新消息，窗口从这一条重新计时（默认 9s）
- **硬上限**：从第一条起算最多等 `max_wait_seconds`（默认 20s），到点必发
- **条数上限**：攒够 `burst_max_messages`（默认 6 条）立刻发
- **媒体立即放行**：图片/文件/语音/视频一到就发（图文混发时用户发完图就在等）
- **指令跳过**：以 `/` 开头的指令、工具续写轮次一律不缓冲

合并时：所有文本按到达顺序用换行拼接，图片 URL 全部收进 `req.image_urls`，最终写回 `req.prompt`。

## 参数怎么调

| 场景 | `merge_window_seconds` | `max_wait_seconds` | 说明 |
| --- | --- | --- | --- |
| 秒回型 | 5 | 15 | 助手感强，适合问答/客服。比最快连发间隔（实测 6s）略短，偶尔会拆成两次回复 |
| 平衡型（默认） | 9 | 20 | 覆盖大多数「等几秒再补一句」，单条消息最多等 9 秒 |
| 慢节奏 | 15 | 30 | 适合闲聊/创作搭子，几乎不会打断思路，但单条消息要等 15 秒 |

> 真实数据：连发间隔中位数 14s。想让**大部分**连发都合并进来，窗口应接近 14；但那样每条消息都要等 14 秒才回，体感很差。9 秒是「合并收益」和「响应速度」的折中——最小间隔 6s 的连发必合并，间隔 14s 的多数会拆成两次。

## 配置项

| 配置键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 总开关 |
| `merge_window_seconds` | float | `9` | 尾随合并窗口（秒），从最后一条消息起算 |
| `max_wait_seconds` | float | `20` | 绝对硬上限（秒），从第一条消息起算；`0` 表示不设上限 |
| `burst_max_messages` | int | `6` | 缓冲区达到该条数立即放行 |
| `flush_on_media` | bool | `true` | 图片/文件/语音/视频消息立即放行 |
| `skip_commands` | bool | `true` | 跳过以 `/` 开头的指令 |
| `skip_tool_rounds` | bool | `true` | 跳过工具续写轮次（带 `tool_calls_result`） |
| `per_session_override` | bool | `false` | 启用按会话覆盖窗口 |
| `per_session_windows` | dict | `{}` | `{umo: 秒}`，仅 `per_session_override` 开启时生效 |
| `debug_log` | bool | `false` | 输出每条消息的登记/并入/放行日志 |

`per_session_windows` 的键是 `unified_msg_origin`，格式 `platform:message_type:session_id`，例如：

```json
{
  "aiocqhttp:FriendMessage:123456789": 5,
  "aiocqhttp:GroupMessage:987654321": 15
}
```

## 指令

| 指令 | 作用 |
| --- | --- |
| `/防抖` | 查看当前配置 + 活跃 buffer 状态 |
| `/防抖 窗口 12` | 临时把本会话合并窗口改为 12 秒（只改内存，不写配置文件，重启失效） |
| `/防抖 状态` | 查看最近 10 次合并记录（合并了几条、等了多久、为什么放行） |

## 实现说明（踩坑记录）

### 1. 拦截点必须在 `on_waiting_llm_request`，不能只在 `on_llm_request`

AstrBot 4.29 的 `on_llm_request` 是在 **session lock 内部**触发的：

```python
# astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py
async with session_lock_manager.acquire_lock(event.unified_msg_origin):
    ...
    if await call_event_hook(event, EventType.OnLLMRequestEvent, req):
        return
```

如果在 `on_llm_request` 里 `await` 等 9 秒窗口，同会话的下一条消息会**卡在 session lock 上**，根本走不到自己的钩子，永远无法并入 buffer——结果是每条消息都各自等 9 秒、各自发一次 LLM，比不装插件还糟。

所以本插件拆成两个钩子：

- `on_waiting_llm_request(priority=190)`：**lock 之前**触发，所有并发消息都能到这里。负责登记到达、把后续消息并入 buffer 并 `stop_event()`、以及 leader 的窗口等待。
- `on_llm_request(priority=190)`：**lock 之内**触发，负责把合并结果真正写回 `req.prompt` / `req.image_urls`。

### 2. `func_tool` 不能用来判断"工具轮次"

`build_main_agent` 在 agent 模式下**总是**给 `req.func_tool` 挂上内置工具集，所以 `req.func_tool is not None` 恒为真。按字面实现 `skip_tool_rounds` 会让插件完全不工作。真正的工具续写轮次标志是 `req.tool_calls_result`，本插件用后者判定。

### 3. `message_str` 的 `/` 前缀会被剥掉

`WakingCheckStage` 会执行 `event.message_str = message_str[len(wake_prefix):].strip()`，所以到达插件时 `/防抖` 已经变成 `防抖`。判断"是不是指令"必须看原始 `event.message_obj.message_str`，或检查 `handlers_parsed_params`。

### 4. `_conf_schema.json` 必须写全

AstrBot 的 `check_config_integrity()` 会删除参考配置（由 schema 生成的默认值）中不存在的键，日志打印 `Config key removed`。少写一个键，用户配了也会被静默清掉。

## 已知限制

- **群聊**：AstrBot 的 `WakingCheckStage` 对未 @ 机器人、也无唤醒前缀的群消息直接 `stop_event()`，这类消息根本到不了插件。因此群聊里只能合并「每条都 @ 了机器人」或「都以 `/` 开头」的连发。私聊不受此限制。
- **多实例**：状态按 `unified_msg_origin` 隔离，多平台/多会话互不影响；但不做持久化，重启后缓冲区清空。
- **零新依赖**：全部基于 Python 标准库，内存占用可忽略（每个活跃会话只有一个小列表 + 一个 50ms 轮询的 Task）。

## 目录结构

```
astrbot_plugin_smart_debounce/
├── main.py                  # Star 类 + 两个钩子 + /防抖 指令（@register 必须在这里）
├── _conf_schema.json        # 配置 schema（键必须写全）
├── metadata.yaml
├── README.md
├── LICENSE                  # MIT
├── smartdebounce/
│   ├── __init__.py
│   ├── buffer.py            # 缓冲核心（纯逻辑，可单测）
│   └── clock.py             # 时间辅助
└── tests/
    └── test_buffer.py       # 22 个单测，不依赖 astrbot
```

## 许可

MIT
