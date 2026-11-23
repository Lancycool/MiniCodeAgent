"""基于 tree-sitter 的 Python AST 结构化解析器。

为什么用 tree-sitter 而不是标准库 ast？
1. 容错解析（Error-Tolerant）：即使文件改到一半存在语法错误，tree-sitter 仍能解析出其余完好的类与函数；
   而标准库 ast.parse() 遇到一处 SyntaxError 就会整文件崩溃。
2. 跨语言统一架构：换一个语法包（如 tree-sitter-go / tree-sitter-typescript）即可扩展到其他语言。
（注：edit_file 的语法预检仍用 ast.parse，因为它就是要严格拒绝任何语法错误。）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())

_IGNORED_REF_NAMES = {
    "self", "cls", "None", "True", "False", "int", "str", "float", "bool", "list",
    "dict", "set", "tuple", "len", "range", "print", "isinstance", "type", "super",
    "Exception", "ValueError", "TypeError", "KeyError", "RuntimeError", "return",
}


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # "class" | "function" | "method"
    file: str
    start_line: int
    end_line: int
    signature: str
    parent: str | None = None
    doc: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass(frozen=True)
class Reference:
    name: str
    file: str
    line: int
    snippet: str


class PythonASTParser:
    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse_source(self, source: str, rel_file: str) -> tuple[list[Symbol], list[Reference]]:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        lines = source.splitlines()

        symbols: list[Symbol] = []
        references: list[Reference] = []

        self._walk_defs(tree.root_node, source_bytes, lines, rel_file, parent_class=None, out=symbols)
        self._walk_refs(tree.root_node, source_bytes, lines, rel_file, out=references)
        return symbols, references

    def parse_file(self, path: Path, rel_file: str) -> tuple[list[Symbol], list[Reference]]:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return [], []
        return self.parse_source(text, rel_file)

    def _walk_defs(
        self,
        node: Node,
        src: bytes,
        lines: list[str],
        rel_file: str,
        *,
        parent_class: str | None,
        out: list[Symbol],
    ) -> None:
        for child in node.children:
            target = child
            if child.type == "decorated_definition":
                # 解包装饰器，取里面的 class_definition / function_definition
                inner = next(
                    (c for c in child.children if c.type in ("class_definition", "function_definition")),
                    None,
                )
                if inner is None:
                    continue
                target = inner

            if target.type == "class_definition":
                name_node = target.child_by_field_name("name")
                if name_node is None:
                    continue
                name = _node_text(name_node, src)
                sig = _extract_header(target, lines)
                doc = _extract_docstring_first_line(target, src)
                out.append(
                    Symbol(
                        name=name,
                        kind="class",
                        file=rel_file,
                        start_line=child.start_point.row + 1,
                        end_line=target.end_point.row + 1,
                        signature=sig,
                        parent=parent_class,
                        doc=doc,
                    )
                )
                body = target.child_by_field_name("body")
                if body is not None:
                    self._walk_defs(body, src, lines, rel_file, parent_class=name, out=out)

            elif target.type == "function_definition":
                name_node = target.child_by_field_name("name")
                if name_node is None:
                    continue
                name = _node_text(name_node, src)
                sig = _extract_header(target, lines)
                doc = _extract_docstring_first_line(target, src)
                out.append(
                    Symbol(
                        name=name,
                        kind="method" if parent_class else "function",
                        file=rel_file,
                        start_line=child.start_point.row + 1,
                        end_line=target.end_point.row + 1,
                        signature=sig,
                        parent=parent_class,
                        doc=doc,
                    )
                )

    def _walk_refs(
        self,
        node: Node,
        src: bytes,
        lines: list[str],
        rel_file: str,
        *,
        out: list[Reference],
    ) -> None:
        """收集函数/方法调用点（call 节点）与 import 引用。"""
        stack = [node]
        seen: set[tuple[str, int]] = set()
        while stack:
            cur = stack.pop()
            if cur.type == "call":
                fn = cur.child_by_field_name("function")
                ref_name = _extract_call_name(fn, src) if fn is not None else ""
                if ref_name and ref_name not in _IGNORED_REF_NAMES:
                    line_no = cur.start_point.row + 1
                    key = (ref_name, line_no)
                    if key not in seen:
                        seen.add(key)
                        snippet = lines[line_no - 1].strip()[:140] if line_no <= len(lines) else ""
                        out.append(Reference(name=ref_name, file=rel_file, line=line_no, snippet=snippet))
            stack.extend(reversed(cur.children))


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_header(node: Node, lines: list[str]) -> str:
    """提取 def/class 签名行（支持跨行签名，压缩为一行）。"""
    start_row = node.start_point.row
    body = node.child_by_field_name("body")
    end_row = body.start_point.row if body is not None else start_row
    # 如果 body 在下一行，只取到 body 前一行
    if body is not None and body.start_point.row > start_row:
        end_row = body.start_point.row - 1
    chunk = " ".join(lines[r].strip() for r in range(start_row, min(end_row + 1, len(lines))))
    return chunk[:160]


def _extract_docstring_first_line(node: Node, src: bytes) -> str:
    body = node.child_by_field_name("body")
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_children:
        expr = first.named_children[0]
        if expr.type == "string":
            raw = _node_text(expr, src).strip().strip("\"'")
            first_line = raw.splitlines()[0].strip() if raw else ""
            return first_line[:80]
    return ""


def _extract_call_name(fn_node: Node, src: bytes) -> str:
    if fn_node.type == "identifier":
        return _node_text(fn_node, src)
    if fn_node.type == "attribute":
        attr = fn_node.child_by_field_name("attribute")
        if attr is not None:
            return _node_text(attr, src)
    return ""
