"""Pydantic JSON Schema → LLM 友好的 tool schema。

Pydantic 生成的 Schema 很"啰嗦"（每个字段都有 title、Optional 变成 anyOf[..., null]），
对弱一些的模型是噪声，还会浪费 Token。这里做一次清洗。
"""

from __future__ import annotations

from typing import Any


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "$defs" in schema:
        # 工具参数刻意保持扁平：嵌套模型会让模型更容易生成错误参数
        raise ValueError("Tool params must be flat (nested models are not supported)")
    return _clean(schema, in_properties=False)


def _clean(node: Any, *, in_properties: bool) -> Any:
    if isinstance(node, list):
        return [_clean(x, in_properties=False) for x in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        # 注意：在 "properties" 下，key 是字段名 —— 一个叫 title 的字段不能被删掉！
        if not in_properties and key == "title" and isinstance(value, str):
            continue
        if not in_properties and key == "default" and value is None:
            continue
        out[key] = _clean(value, in_properties=(key == "properties" and not in_properties))

    # Optional[X] 在 Pydantic 中是 anyOf: [X, {type: null}]，折叠成 X
    any_of = out.get("anyOf")
    if not in_properties and isinstance(any_of, list):
        non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
        if len(non_null) == 1 and len(non_null) < len(any_of):
            out.pop("anyOf")
            out = {**non_null[0], **out}
    return out
