"""底层文件遍历工具（无上层依赖，避免 codeindex / sandbox / tools 之间产生循环导入）。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build",
}


def iter_files(root: Path) -> Iterator[Path]:
    """遍历目录，剪枝忽略目录；排序保证结果确定（便于复现与缓存命中）。"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            yield Path(dirpath) / name
