"""MiniBench：自建多类别 Python Bug 评测集（含隐藏测试防作弊，对标 SWE-bench 评测范式）。

每道题包含：
- files：放入沙箱供 Agent 阅读与修复的仓库快照（含基础公开测试）
- hidden_tests：Agent 运行期间**不可见**的隐藏边界测试；仅在评测阶段注入沙箱做最终裁决，
  防止模型“面向公开用例硬编码”作弊。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchTask:
    id: str
    category: str
    issue: str
    files: dict[str, str]
    hidden_tests: dict[str, str]

    def materialize(self, dest: Path, *, include_hidden: bool = False) -> None:
        for rel, content in self.files.items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        if include_hidden:
            for rel, content in self.hidden_tests.items():
                p = dest / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")


def _distractor_file(prefix: str, n: int = 18) -> str:
    return "\n\n".join(f"def {prefix}_{i}(v: int) -> int:\n    return v + {i}" for i in range(1, n + 1)) + "\n"


def builtin_tasks() -> list[BenchTask]:
    return [
        # ---- Task 1: 边界条件与下限保护 ----
        BenchTask(
            id="t01_cart_boundary",
            category="boundary",
            issue=(
                "购物车结算（cart.py）有两个问题：\n"
                "1) 满 100 元享 9 折，但刚好 100 元时未触发折扣；\n"
                "2) 扣除定额优惠券后总价可能为负，正确行为是最低 0.0。"
            ),
            files={
                "cart.py": (
                    "def calculate_total(items: list[tuple[str, float, int]], coupon: float = 0.0) -> float:\n"
                    "    subtotal = sum(p * q for _, p, q in items)\n"
                    "    if subtotal > 100:\n"
                    "        subtotal *= 0.9\n"
                    "    return round(subtotal - coupon, 2)\n"
                ),
                "test_cart.py": (
                    "from cart import calculate_total\n"
                    "def test_exact_100():\n"
                    "    assert calculate_total([('book', 50.0, 2)]) == 90.0\n"
                    "def test_floor_zero():\n"
                    "    assert calculate_total([('pen', 5.0, 2)], coupon=50.0) == 0.0\n"
                ),
            },
            hidden_tests={
                "test_hidden_cart.py": (
                    "from cart import calculate_total\n"
                    "def test_hidden_edges():\n"
                    "    assert calculate_total([], coupon=10.0) == 0.0\n"
                    "    assert calculate_total([('a', 99.99, 1)], coupon=0.0) == 99.99\n"
                    "    assert calculate_total([('a', 100.0, 1)], coupon=90.0) == 0.0\n"
                    "    assert calculate_total([('a', 200.0, 1)], coupon=30.0) == 150.0\n"
                )
            },
        ),
        # ---- Task 2: 有状态数据结构（LRU Cache） ----
        BenchTask(
            id="t02_lru_cache",
            category="stateful",
            issue=(
                "LRUCache（lru_cache.py）存在两个 Bug：\n"
                "1) get(key) 命中后没有把该 key 刷新为最近使用；\n"
                "2) put(key, value) 更新已存在 key 且缓存已满时，错误地多淘汰了一个 key。"
            ),
            files={
                "lru_cache.py": (
                    "from collections import OrderedDict\n\n"
                    "class LRUCache:\n"
                    "    def __init__(self, capacity: int) -> None:\n"
                    "        self.capacity = capacity\n"
                    "        self._data: OrderedDict[str, int] = OrderedDict()\n\n"
                    "    def get(self, key: str) -> int | None:\n"
                    "        if key not in self._data:\n"
                    "            return None\n"
                    "        return self._data[key]\n\n"
                    "    def put(self, key: str, value: int) -> None:\n"
                    "        if len(self._data) >= self.capacity:\n"
                    "            self._data.popitem(last=False)\n"
                    "        self._data[key] = value\n"
                ),
                "test_lru.py": (
                    "from lru_cache import LRUCache\n"
                    "def test_get_refreshes():\n"
                    "    c = LRUCache(2); c.put('a', 1); c.put('b', 2)\n"
                    "    assert c.get('a') == 1\n"
                    "    c.put('c', 3)\n"
                    "    assert c.get('a') == 1 and c.get('b') is None\n"
                    "def test_put_existing_no_evict():\n"
                    "    c = LRUCache(2); c.put('a', 1); c.put('b', 2); c.put('a', 10)\n"
                    "    assert c.get('a') == 10 and c.get('b') == 2\n"
                ),
            },
            hidden_tests={
                "test_hidden_lru.py": (
                    "from lru_cache import LRUCache\n"
                    "def test_put_existing_also_refreshes_recency():\n"
                    "    c = LRUCache(2); c.put('a', 1); c.put('b', 2)\n"
                    "    c.put('a', 10)  # 'a' 变为最近使用，'b' 变为最久未使用\n"
                    "    c.put('c', 3)   # 应淘汰 'b'\n"
                    "    assert c.get('b') is None and c.get('a') == 10 and c.get('c') == 3\n"
                )
            },
        ),
        # ---- Task 3: 多文件跨模块 + 干扰文件（考验 AST 检索与 Repo Map） ----
        BenchTask(
            id="t03_multifile_checkout",
            category="cross_file",
            issue=(
                "电商结算系统中，VIP 折扣（apply_vip_discount）应该只对商品小计打折，运费（calc_shipping_fee）不打折。"
                "但目前运费也被打折了，请定位跨文件调用链并修复。"
            ),
            files={
                "models.py": (
                    "from dataclasses import dataclass\n"
                    "@dataclass\nclass Item:\n    name: str\n    price: float\n    qty: int\n"
                    "@dataclass\nclass Order:\n    items: list[Item]\n    is_vip: bool = False\n    region: str = 'CN'\n"
                ),
                "shipping.py": (
                    "def calc_shipping_fee(subtotal: float, region: str) -> float:\n"
                    "    return 0.0 if subtotal >= 200.0 else (15.0 if region == 'CN' else 40.0)\n"
                ),
                "discounts.py": (
                    "def apply_vip_discount(amount: float, is_vip: bool) -> float:\n"
                    "    return round(amount * 0.85, 2) if is_vip else amount\n"
                ),
                "checkout.py": (
                    "from discounts import apply_vip_discount\n"
                    "from models import Order\n"
                    "from shipping import calc_shipping_fee\n\n"
                    "def compute_order_total(order: Order) -> float:\n"
                    "    subtotal = sum(i.price * i.qty for i in order.items)\n"
                    "    fee = calc_shipping_fee(subtotal, order.region)\n"
                    "    return round(apply_vip_discount(subtotal + fee, order.is_vip), 2)\n"
                ),
                "legacy_noise_a.py": _distractor_file("report_a", 20),
                "legacy_noise_b.py": _distractor_file("report_b", 20),
                "test_checkout.py": (
                    "from checkout import compute_order_total\n"
                    "from models import Item, Order\n"
                    "def test_vip_excludes_shipping():\n"
                    "    o = Order([Item('book', 50.0, 2)], is_vip=True, region='CN')\n"
                    "    assert compute_order_total(o) == 100.0\n"
                ),
            },
            hidden_tests={
                "test_hidden_checkout.py": (
                    "from checkout import compute_order_total\n"
                    "from models import Item, Order\n"
                    "def test_overseas_vip():\n"
                    "    o = Order([Item('pen', 100.0, 1)], is_vip=True, region='US')\n"
                    "    assert compute_order_total(o) == 125.0  # 85 + 40\n"
                    "def test_regular_user():\n"
                    "    o = Order([Item('pen', 100.0, 1)], is_vip=False, region='CN')\n"
                    "    assert compute_order_total(o) == 115.0\n"
                )
            },
        ),
        # ---- Task 4: 数据清洗与 None/嵌套字典健壮性 ----
        BenchTask(
            id="t04_config_merger",
            category="data_handling",
            issue=(
                "deep_merge(base, override) 合并嵌套配置字典时有两个 Bug：\n"
                "1) 修改了传入的 base 原字典（存在副作用），应当返回新字典不修改入参；\n"
                "2) 当 override 中某键值为 None 时，应当忽略该 None 保留 base 的原值，而不是覆盖成 None。"
            ),
            files={
                "merger.py": (
                    "def deep_merge(base: dict, override: dict) -> dict:\n"
                    "    for k, v in override.items():\n"
                    "        if k in base and isinstance(base[k], dict) and isinstance(v, dict):\n"
                    "            base[k] = deep_merge(base[k], v)\n"
                    "        else:\n"
                    "            base[k] = v\n"
                    "    return base\n"
                ),
                "test_merger.py": (
                    "from merger import deep_merge\n"
                    "def test_no_mutate_and_skip_none():\n"
                    "    b = {'a': 1, 'db': {'host': 'localhost', 'port': 5432}}\n"
                    "    out = deep_merge(b, {'a': None, 'db': {'port': 3306}})\n"
                    "    assert out == {'a': 1, 'db': {'host': 'localhost', 'port': 3306}}\n"
                    "    assert b['db']['port'] == 5432\n"
                ),
            },
            hidden_tests={
                "test_hidden_merger.py": (
                    "from merger import deep_merge\n"
                    "def test_nested_none_and_new_key():\n"
                    "    b = {'db': {'host': 'h1', 'user': 'root'}}\n"
                    "    out = deep_merge(b, {'db': {'host': None, 'ssl': True}, 'extra': None})\n"
                    "    assert out == {'db': {'host': 'h1', 'user': 'root', 'ssl': True}}\n"
                    "    assert 'ssl' not in b['db']\n"
                )
            },
        ),
        # ---- Task 5: 滑动窗口限流器（时间边界） ----
        BenchTask(
            id="t05_rate_limiter",
            category="algorithm",
            issue=(
                "SlidingWindowLimiter（limiter.py）按窗口大小 window_s 限制最多 max_requests 次请求，"
                "但过期时间戳清理条件写反/边界错误：恰好在 now - window_s 之前的请求应该已经过期，"
                "且被拒绝的请求不应计入已用配额。"
            ),
            files={
                "limiter.py": (
                    "from collections import deque\n\n"
                    "class SlidingWindowLimiter:\n"
                    "    def __init__(self, max_requests: int, window_s: float) -> None:\n"
                    "        self.max_requests = max_requests\n"
                    "        self.window_s = window_s\n"
                    "        self._hits: deque[float] = deque()\n\n"
                    "    def allow(self, now: float) -> bool:\n"
                    "        cutoff = now - self.window_s\n"
                    "        while self._hits and self._hits[0] < cutoff:\n"
                    "            self._hits.popleft()\n"
                    "        self._hits.append(now)\n"
                    "        return len(self._hits) <= self.max_requests\n"
                ),
                "test_limiter.py": (
                    "from limiter import SlidingWindowLimiter\n"
                    "def test_rejected_not_counted_and_exact_window():\n"
                    "    lim = SlidingWindowLimiter(max_requests=2, window_s=10.0)\n"
                    "    assert lim.allow(0.0) is True\n"
                    "    assert lim.allow(1.0) is True\n"
                    "    assert lim.allow(2.0) is False  # 被拒绝不应占位\n"
                    "    assert lim.allow(10.0) is True  # t=0.0 在 t=10.0 时刚好过期\n"
                ),
            },
            hidden_tests={
                "test_hidden_limiter.py": (
                    "from limiter import SlidingWindowLimiter\n"
                    "def test_multiple_rejections_do_not_block_later():\n"
                    "    lim = SlidingWindowLimiter(max_requests=1, window_s=5.0)\n"
                    "    assert lim.allow(0.0) is True\n"
                    "    assert lim.allow(1.0) is False\n"
                    "    assert lim.allow(2.0) is False\n"
                    "    assert lim.allow(5.0) is True\n"
                )
            },
        ),
        # ---- Task 6: 跨文件参数解析与路由匹配 ----
        BenchTask(
            id="t06_path_router",
            category="cross_file",
            issue=(
                "简易路由系统（router.py + matcher.py）匹配动态路径参数如 `/users/{id}/posts/{pid}` 时：\n"
                "1) 末尾带斜杠 `/users/42/` 与 `/users/42` 应该等价匹配；\n"
                "2) 静态路由应优先于动态参数路由（如 `/users/me` 优先匹配静态 `/users/me` 而非 `/users/{id}`）。"
            ),
            files={
                "matcher.py": (
                    "def match_pattern(pattern: str, path: str) -> dict[str, str] | None:\n"
                    "    p_parts = pattern.split('/')\n"
                    "    u_parts = path.split('/')\n"
                    "    if len(p_parts) != len(u_parts):\n"
                    "        return None\n"
                    "    params: dict[str, str] = {}\n"
                    "    for pp, up in zip(p_parts, u_parts):\n"
                    "        if pp.startswith('{') and pp.endswith('}'):\n"
                    "            params[pp[1:-1]] = up\n"
                    "        elif pp != up:\n"
                    "            return None\n"
                    "    return params\n"
                ),
                "router.py": (
                    "from matcher import match_pattern\n\n"
                    "class Router:\n"
                    "    def __init__(self) -> None:\n"
                    "        self._routes: list[tuple[str, str]] = []\n\n"
                    "    def add(self, pattern: str, handler: str) -> None:\n"
                    "        self._routes.append((pattern, handler))\n\n"
                    "    def resolve(self, path: str) -> tuple[str, dict[str, str]] | None:\n"
                    "        for pattern, handler in self._routes:\n"
                    "            params = match_pattern(pattern, path)\n"
                    "            if params is not None:\n"
                    "                return handler, params\n"
                    "        return None\n"
                ),
                "misc_helpers.py": _distractor_file("helper", 16),
                "test_router.py": (
                    "from router import Router\n"
                    "def test_trailing_slash_and_static_priority():\n"
                    "    r = Router()\n"
                    "    r.add('/users/{id}', 'get_user')\n"
                    "    r.add('/users/me', 'get_me')\n"
                    "    assert r.resolve('/users/42/') == ('get_user', {'id': '42'})\n"
                    "    assert r.resolve('/users/me') == ('get_me', {})\n"
                ),
            },
            hidden_tests={
                "test_hidden_router.py": (
                    "from router import Router\n"
                    "def test_root_and_multi_param():\n"
                    "    r = Router()\n"
                    "    r.add('/', 'home')\n"
                    "    r.add('/users/{uid}/posts/{pid}', 'post')\n"
                    "    assert r.resolve('/') == ('home', {})\n"
                    "    assert r.resolve('/users/7/posts/99/') == ('post', {'uid': '7', 'pid': '99'})\n"
                )
            },
        ),
    ]
