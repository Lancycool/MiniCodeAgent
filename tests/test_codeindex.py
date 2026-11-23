"""Tree-sitter AST 索引、Repo Map 与 AST 检索工具单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mca.codeindex import CodeIndex, PythonASTParser
from mca.llm.types import ToolCall
from mca.tools import ToolContext, ToolRegistry, ast_tools


@pytest.fixture
def multi_file_repo(tmp_path: Path) -> Path:
    (tmp_path / "pricing.py").write_text(
        '"""定价核心模块。"""\n\n'
        "def apply_tax(amount: float, rate: float = 0.1) -> float:\n"
        '    """计算含税金额。"""\n'
        "    return amount * (1 + rate)\n\n"
        "class OrderCalculator:\n"
        "    def compute(self, base: float) -> float:\n"
        "        return apply_tax(base)\n",
        encoding="utf-8",
    )
    (tmp_path / "checkout.py").write_text(
        "from pricing import OrderCalculator, apply_tax\n\n"
        "def run_checkout(price: float) -> float:\n"
        "    calc = OrderCalculator()\n"
        "    return calc.compute(price) + apply_tax(10.0)\n",
        encoding="utf-8",
    )
    # 故意放一个语法写坏了一半的文件，验证 tree-sitter 的容错解析能力
    (tmp_path / "broken_wip.py").write_text(
        "def intact_func(x: int) -> int:\n"
        "    return x + 1\n\n"
        "def broken_func(:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return tmp_path


def test_tree_sitter_error_tolerant_parsing(multi_file_repo: Path):
    parser = PythonASTParser()
    syms, _ = parser.parse_file(multi_file_repo / "broken_wip.py", "broken_wip.py")
    names = [s.name for s in syms]
    # 即使文件后半段有语法错误，前半段完好的函数仍能被 tree-sitter 提取出来
    assert "intact_func" in names


def test_code_index_search_refs_and_repo_map(multi_file_repo: Path):
    idx = CodeIndex(multi_file_repo)
    idx.build()

    # 精确符号定义
    hits = idx.search_symbol("apply_tax")
    assert len(hits) == 1 and hits[0].file == "pricing.py" and hits[0].start_line == 3

    # 跨文件引用查找
    refs = idx.find_references("apply_tax")
    ref_files = {r.file for r in refs}
    assert "pricing.py" in ref_files and "checkout.py" in ref_files

    # Repo Map 排序：pricing.py 被跨文件引用且命中 focus_terms，应排在最前面
    rmap = idx.build_repo_map(max_chars=2000, focus_terms=["apply_tax"])
    assert "pricing.py:" in rmap and "def apply_tax" in rmap
    assert rmap.index("pricing.py:") < rmap.index("broken_wip.py:")


async def test_ast_tools_via_registry(multi_file_repo: Path):
    idx = CodeIndex(multi_file_repo)
    idx.build()
    ctx = ToolContext(workspace=multi_file_repo, code_index=idx)
    reg = ToolRegistry(ast_tools())

    r1 = await reg.execute(ToolCall("c1", "view_outline", '{"path": "pricing.py"}'), ctx)
    assert r1.ok and "class OrderCalculator:" in r1.content and "[L3-L5]" in r1.content

    r2 = await reg.execute(ToolCall("c2", "search_symbol", '{"name": "OrderCalculator"}'), ctx)
    assert r2.ok and "pricing.py:" in r2.content

    r3 = await reg.execute(ToolCall("c3", "find_references", '{"name": "compute"}'), ctx)
    assert r3.ok and "checkout.py:" in r3.content
