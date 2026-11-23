"""Lesson 4 实战演示：用真正的 AgentLoop 状态机 + LoopGuard + SQLite Trace 修复 Bug，并回放全链路。

运行：
  python scripts/l4_agent_demo.py
  python scripts/l4_agent_demo.py --docker
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from mca.agent import AgentLoop, Budget, LoopGuard
from mca.config import LLMSettings
from mca.llm import OpenAICompatClient
from mca.observability import TraceStore, Tracer, format_run_replay, format_tool_stats
from mca.sandbox import DockerSandbox, LocalSandbox
from mca.tools import ToolRegistry, fix_tools

ISSUE = (
    "LRU 缓存（lru_cache.py）存在两个 Bug：\n"
    "1) 当访问已存在的 key（get）时，该 key 没有刷新为“最近使用”，导致后续淘汰了刚被访问过的热点 key；\n"
    "2) 当更新已存在的 key（put）且缓存已满时，错误地多淘汰了一个其他 key。\n"
    "请定位根因、修复并确保所有测试通过，完成后调用 submit。"
)


def create_lru_repo(root: Path) -> None:
    (root / "lru_cache.py").write_text(
        "from collections import OrderedDict\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int) -> None:\n"
        "        if capacity <= 0:\n"
        "            raise ValueError('capacity must be positive')\n"
        "        self.capacity = capacity\n"
        "        self._data: OrderedDict[str, int] = OrderedDict()\n\n"
        "    def get(self, key: str) -> int | None:\n"
        "        if key not in self._data:\n"
        "            return None\n"
        "        return self._data[key]\n\n"
        "    def put(self, key: str, value: int) -> None:\n"
        "        if len(self._data) >= self.capacity:\n"
        "            self._data.popitem(last=False)\n"
        "        self._data[key] = value\n",
        encoding="utf-8",
    )
    (root / "test_lru_cache.py").write_text(
        "from lru_cache import LRUCache\n\n"
        "def test_get_refreshes_recency():\n"
        "    c = LRUCache(2)\n"
        "    c.put('a', 1)\n"
        "    c.put('b', 2)\n"
        "    assert c.get('a') == 1   # 'a' 变为最近使用，'b' 变为最久未使用\n"
        "    c.put('c', 3)            # 应淘汰 'b' 而不是 'a'\n"
        "    assert c.get('a') == 1\n"
        "    assert c.get('b') is None\n"
        "    assert c.get('c') == 3\n\n"
        "def test_put_existing_key_does_not_evict_when_full():\n"
        "    c = LRUCache(2)\n"
        "    c.put('a', 1)\n"
        "    c.put('b', 2)\n"
        "    c.put('a', 10)           # 更新已有 key 不应淘汰 'b'\n"
        "    assert c.get('a') == 10\n"
        "    assert c.get('b') == 2\n"
        "    c.put('c', 3)            # 此时 'b' 刚被 get，应淘汰 'a'；若改顺序请看上面 get\n"
        "    assert c.get('b') == 2\n"
        "    assert c.get('c') == 3\n",
        encoding="utf-8",
    )


async def main(use_docker: bool) -> None:
    db_path = Path("runs/traces.db")
    store = TraceStore(db_path)

    with tempfile.TemporaryDirectory(prefix="mca_l4_src_") as src_dir:
        src_path = Path(src_dir)
        create_lru_repo(src_path)

        sandbox = DockerSandbox(src_path) if use_docker else LocalSandbox(src_path)
        async with sandbox as sb:
            loop = AgentLoop(
                llm=OpenAICompatClient(LLMSettings()),
                registry=ToolRegistry(fix_tools()),
                sandbox=sb,
                budget=Budget(max_steps=15, max_tokens=60_000, max_recovers=2),
                loop_guard=LoopGuard(enabled=True),
                tracer=Tracer(store),
            )
            res = await loop.run(ISSUE, task_id="demo_lru_cache", mode="fix")

    print("\n" + format_run_replay(store, res.run_id))
    print("\n=== 工具统计 SQL（累计） ===")
    print(format_tool_stats(store))
    print("\n=== 最终 git diff ===")
    print(res.final_patch or "(无改动)")
    print(f"\nTrace 已落库: {db_path.resolve()} (run_id={res.run_id})")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true")
    asyncio.run(main(use_docker=parser.parse_args().docker))
