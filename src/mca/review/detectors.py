"""自研 Python AST 坏味道检测器（确定性“事实”层）。

涵盖 8 类典型“屎山代码”坏味道：
  1. LongFunction     —— 超长函数
  2. DeepNesting      —— 控制流嵌套过深（箭头型代码）
  3. TooManyParams    —— 参数列表过长
  4. GodClass         —— 上帝类（方法过多或体量过大）
  5. LongIfElifChain  —— 冗长 if-elif 分支链
  6. BroadException   —— 裸 except 或 except Exception: pass 吞异常
  7. GlobalMutation   —— 函数内使用 global 修改全局状态
  8. DuplicateBlock   —— 归一化 AST（抹去变量名差异）后检测结构重复代码块
"""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from mca.review.models import Finding


@dataclass(frozen=True)
class DetectorThresholds:
    max_func_lines: int = 45
    max_nesting_depth: int = 3
    max_params: int = 5
    max_class_methods: int = 12
    max_class_lines: int = 300
    max_elif_branches: int = 5
    min_dup_statements: int = 4


def detect_smells_in_source(
    source: str,
    rel_file: str,
    thresholds: DetectorThresholds | None = None,
) -> tuple[list[Finding], dict[str, list[tuple[str, int, int, str]]]]:
    """扫描单个 Python 文件，返回 (Findings, 归一化代码块哈希映射供跨函数查重)。"""
    th = thresholds or DetectorThresholds()
    try:
        tree = ast.parse(source, filename=rel_file)
    except SyntaxError:
        return [], {}

    lines = source.splitlines()
    findings: list[Finding] = []
    block_hashes: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)

    def _snippet(start: int, end: int) -> str:
        lo = max(0, start - 1)
        hi = min(len(lines), min(end, start + 3))
        return "\n".join(lines[lo:hi])

    def _visit_func(node: ast.FunctionDef | ast.AsyncFunctionDef, parent_class: str | None) -> None:
        sym = f"{parent_class}.{node.name}" if parent_class else node.name
        start = node.lineno
        end = node.end_lineno or start
        length = end - start + 1

        # 1. LongFunction
        if length > th.max_func_lines:
            sev = "major" if length > th.max_func_lines * 2 else "minor"
            findings.append(
                Finding(
                    rule_id="LongFunction",
                    source="detector",
                    file=rel_file,
                    start_line=start,
                    end_line=end,
                    symbol=sym,
                    severity=sev,
                    message=f"函数 {sym} 长达 {length} 行（阈值 {th.max_func_lines}），职责可能过重",
                    metrics={"lines": length},
                    evidence=_snippet(start, end),
                )
            )

        # 2. TooManyParams（排除 self / cls）
        all_args = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        effective_params = [a for a in all_args if a not in ("self", "cls")]
        if len(effective_params) > th.max_params:
            findings.append(
                Finding(
                    rule_id="TooManyParams",
                    source="detector",
                    file=rel_file,
                    start_line=start,
                    end_line=start,
                    symbol=sym,
                    severity="minor",
                    message=f"函数 {sym} 有 {len(effective_params)} 个参数（阈值 {th.max_params}），建议封装为 dataclass/参数对象",
                    metrics={"params": len(effective_params)},
                    evidence=_snippet(start, start + 1),
                )
            )

        # 3. DeepNesting
        depth = max_nesting_depth(node)
        if depth > th.max_nesting_depth:
            sev = "major" if depth >= th.max_nesting_depth + 2 else "minor"
            findings.append(
                Finding(
                    rule_id="DeepNesting",
                    source="detector",
                    file=rel_file,
                    start_line=start,
                    end_line=end,
                    symbol=sym,
                    severity=sev,
                    message=f"函数 {sym} 控制流最大嵌套深度为 {depth}（阈值 {th.max_nesting_depth}），建议用卫语句提前返回或拆分函数",
                    metrics={"nesting": depth},
                    evidence=_snippet(start, end),
                )
            )

        # 4~7. 函数内部遍历：LongIfElifChain / BroadException / GlobalMutation
        for sub in ast.walk(node):
            if isinstance(sub, ast.If):
                branches = _count_elif_branches(sub)
                if branches >= th.max_elif_branches:
                    findings.append(
                        Finding(
                            rule_id="LongIfElifChain",
                            source="detector",
                            file=rel_file,
                            start_line=sub.lineno,
                            end_line=sub.end_lineno or sub.lineno,
                            symbol=sym,
                            severity="minor",
                            message=f"存在 {branches} 层 if-elif 分支链，建议改用字典分派或策略模式",
                            metrics={"branches": branches},
                            evidence=_snippet(sub.lineno, sub.lineno + 2),
                        )
                    )
            elif isinstance(sub, ast.ExceptHandler):
                is_bare = sub.type is None
                is_broad = isinstance(sub.type, ast.Name) and sub.type.id in ("Exception", "BaseException")
                is_swallowed = len(sub.body) == 1 and isinstance(sub.body[0], ast.Pass)
                if is_bare or (is_broad and is_swallowed):
                    findings.append(
                        Finding(
                            rule_id="BroadException",
                            source="detector",
                            file=rel_file,
                            start_line=sub.lineno,
                            end_line=sub.end_lineno or sub.lineno,
                            symbol=sym,
                            severity="major",
                            message="存在裸 except 或 `except Exception: pass` 静默吞异常，会掩盖真实故障",
                            evidence=_snippet(sub.lineno, sub.lineno + 2),
                        )
                    )
            elif isinstance(sub, ast.Global):
                findings.append(
                    Finding(
                        rule_id="GlobalMutation",
                        source="detector",
                        file=rel_file,
                        start_line=sub.lineno,
                        end_line=sub.lineno,
                        symbol=sym,
                        severity="minor",
                        message=f"使用 global 修改全局状态 ({', '.join(sub.names)})，破坏可测试性与并发安全",
                        evidence=_snippet(sub.lineno, sub.lineno + 1),
                    )
                )

        # 8. 归一化 AST 指纹（用于检测跨函数复制粘贴的重复代码）
        if len(node.body) >= th.min_dup_statements:
            norm_hash = _normalized_body_hash(node.body)
            block_hashes[norm_hash].append((rel_file, start, end, sym))

    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _visit_func(top, parent_class=None)
        elif isinstance(top, ast.ClassDef):
            methods = [n for n in top.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            c_start = top.lineno
            c_end = top.end_lineno or c_start
            c_lines = c_end - c_start + 1
            if len(methods) > th.max_class_methods or c_lines > th.max_class_lines:
                findings.append(
                    Finding(
                        rule_id="GodClass",
                        source="detector",
                        file=rel_file,
                        start_line=c_start,
                        end_line=c_end,
                        symbol=top.name,
                        severity="major",
                        message=f"类 {top.name} 包含 {len(methods)} 个方法、共 {c_lines} 行，疑似上帝类，建议按职责拆分",
                        metrics={"methods": len(methods), "lines": c_lines},
                        evidence=_snippet(c_start, c_start + 2),
                    )
                )
            for m in methods:
                _visit_func(m, parent_class=top.name)

    return findings, block_hashes


def max_nesting_depth(node: ast.AST) -> int:
    """计算函数内控制流（if / for / while / with / try）的最大嵌套深度。"""
    control_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)

    def _walk(cur: ast.AST, depth: int) -> int:
        best = depth
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # 不跨进嵌套定义
            if isinstance(child, control_types):
                # elif 在 Python AST 中是 If 嵌在 orelse 里，不额外增加视觉嵌套深度
                if isinstance(cur, ast.If) and isinstance(child, ast.If) and child in cur.orelse:
                    best = max(best, _walk(child, depth))
                else:
                    best = max(best, _walk(child, depth + 1))
            else:
                best = max(best, _walk(child, depth))
        return best

    return _walk(node, 0)


def _count_elif_branches(node: ast.If) -> int:
    count = 1
    cur = node
    while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
        count += 1
        cur = cur.orelse[0]
    return count


def _normalized_body_hash(stmts: list[ast.stmt]) -> str:
    """将语句列表的 AST 归一化（所有变量名替换成统一占位符），检测结构相同的复制粘贴代码。"""

    class _Normalizer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    mod = ast.Module(body=[_Normalizer().visit(ast.fix_missing_locations(s)) for s in stmts], type_ignores=[])
    dumped = ast.dump(mod, include_attributes=False)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]
