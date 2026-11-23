"""CodeIndex：仓库级 AST 符号索引与 Repo Map 生成器（借鉴 Aider 设计）。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from mca.codeindex.parser import PythonASTParser, Reference, Symbol
from mca.fs_utils import iter_files
from mca.security import is_sensitive_path


class CodeIndex:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self._parser = PythonASTParser()
        self._file_symbols: dict[str, list[Symbol]] = {}
        self._file_refs: dict[str, list[Reference]] = {}
        self._built = False

    def build(self) -> None:
        self._file_symbols.clear()
        self._file_refs.clear()
        for path in iter_files(self.workspace):
            if path.suffix != ".py" or is_sensitive_path(path):
                continue
            rel = path.relative_to(self.workspace).as_posix()
            syms, refs = self._parser.parse_file(path, rel)
            self._file_symbols[rel] = syms
            self._file_refs[rel] = refs
        self._built = True

    def ensure_built(self) -> None:
        if not self._built:
            self.build()

    def refresh_file(self, rel_path: str) -> None:
        """文件被修改后按单文件增量刷新索引，无需全仓重建。"""
        rel = rel_path.replace("\\", "/")
        p = self.workspace / rel
        if not p.exists() or p.suffix != ".py":
            self._file_symbols.pop(rel, None)
            self._file_refs.pop(rel, None)
            return
        syms, refs = self._parser.parse_file(p, rel)
        self._file_symbols[rel] = syms
        self._file_refs[rel] = refs

    # ---------------- 检索接口 ----------------

    def file_outline(self, rel_path: str) -> str:
        """返回单个文件的大纲（类与函数签名 + 行号范围），相比整文件读取节省大量 Token。"""
        rel = rel_path.replace("\\", "/")
        p = self.workspace / rel
        if not p.exists():
            return ""
        syms, _ = self._parser.parse_file(p, rel)
        if not syms:
            return f"[{rel}] (无顶层类或函数定义)"
        lines = [f"[{rel}]"]
        for s in syms:
            indent = "  " if s.kind == "method" else ""
            doc = f"  # {s.doc}" if s.doc else ""
            lines.append(f"{indent}{s.signature}  [L{s.start_line}-L{s.end_line}]{doc}")
        return "\n".join(lines)

    def search_symbol(self, name: str, *, limit: int = 30) -> list[Symbol]:
        self.ensure_built()
        exact: list[Symbol] = []
        partial: list[Symbol] = []
        q = name.lower()
        for syms in self._file_symbols.values():
            for s in syms:
                if s.name == name or s.qualified_name == name:
                    exact.append(s)
                elif q in s.name.lower() or q in s.qualified_name.lower():
                    partial.append(s)
        return (exact + partial)[:limit]

    def find_references(self, name: str, *, limit: int = 50) -> list[Reference]:
        self.ensure_built()
        hits: list[Reference] = []
        for refs in self._file_refs.values():
            for r in refs:
                if r.name == name:
                    hits.append(r)
                    if len(hits) >= limit:
                        return hits
        return hits

    # ---------------- Repo Map（按引用热度 + 任务关键词排序，预算内裁剪） ----------------

    def build_repo_map(
        self,
        *,
        max_chars: int = 4000,
        focus_terms: Iterable[str] = (),
    ) -> str:
        self.ensure_built()
        if not self._file_symbols:
            return "(仓库中未发现 Python 文件)"

        # 统计每个符号名定义在哪些文件，用于计算文件间的引用热度（简化 PageRank / 入度中心性）
        sym_to_files: dict[str, set[str]] = defaultdict(set)
        for rel, syms in self._file_symbols.items():
            for s in syms:
                sym_to_files[s.name].add(rel)

        file_score: Counter[str] = Counter()
        for rel, refs in self._file_refs.items():
            for r in refs:
                for target_file in sym_to_files.get(r.name, ()):
                    if target_file != rel:  # 跨文件引用加分
                        file_score[target_file] += 2
                    else:
                        file_score[target_file] += 1

        terms = {t.lower() for t in focus_terms if len(t) >= 3}
        for rel, syms in self._file_symbols.items():
            # 测试文件默认略降权，优先展示核心业务模块
            if Path(rel).name.startswith("test_") or "/tests/" in f"/{rel}":
                file_score[rel] -= 1
            haystack = (rel + " " + " ".join(s.name for s in syms)).lower()
            for t in terms:
                if t in haystack:
                    file_score[rel] += 10  # 命中 Issue 关键词显著提权

        ranked_files = sorted(
            self._file_symbols.keys(),
            key=lambda f: (-file_score[f], f),
        )

        blocks: list[str] = []
        used = 0
        omitted = 0
        for rel in ranked_files:
            syms = self._file_symbols[rel]
            if not syms:
                continue
            lines = [f"{rel}:"]
            for s in syms:
                indent = "    " if s.kind == "method" else "  "
                lines.append(f"{indent}{s.signature} [L{s.start_line}-L{s.end_line}]")
            block = "\n".join(lines)
            if used + len(block) + 2 > max_chars:
                omitted += 1
                continue
            blocks.append(block)
            used += len(block) + 2

        header = "=== Repo Map (AST 符号骨架，按引用热度排序) ==="
        footer = f"\n[已省略 {omitted} 个低优先级文件，可用 list_dir / search_symbol 查看]" if omitted else ""
        return header + "\n" + "\n\n".join(blocks) + footer


def extract_focus_terms(issue: str) -> list[str]:
    """从 Issue 文本中提取可能的标识符/文件名作为 Repo Map 加权词。"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue)
