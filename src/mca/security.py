"""敏感信息防护：两道防线。

防线 1（访问控制）：敏感文件禁止被工具读取/搜索 —— 从源头不让密钥进入上下文。
防线 2（输出脱敏）：所有工具输出在回传模型前扫描密钥模式并打码 ——
          兜住防线 1 漏掉的情况（如密钥被硬编码在 config.py、打印在日志里）。

背景：L2 演示中模型为"检查环境变量"主动 read_file('.env')，API Key 随工具结果
被发送给了模型服务商。见 docs/debug_log.md #3。
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

_SENSITIVE_NAME_PATTERNS = (
    ".env", ".env.*", "*.env",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".netrc", ".pypirc", ".npmrc", ".git-credentials",
    "credentials", "credentials.*", "secrets.*", "*.secret",
)
# 模板文件不含真实密钥，且对模型理解配置有帮助
_ALLOWED_NAMES = {".env.example", ".env.sample", ".env.template"}


def is_sensitive_path(path: Path) -> bool:
    name = path.name.lower()
    if name in _ALLOWED_NAMES:
        return False
    return any(fnmatch(name, pattern) for pattern in _SENSITIVE_NAME_PATTERNS)


_REDACTED = "[REDACTED]"

# (?<![A-Za-z0-9]) 防止误伤，如 "risk-assessment-framework" 中的 "sk-..."
_TOKEN_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-.]{16,}"),          # OpenAI / DeepSeek / 百炼 风格
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),                  # AWS Access Key
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,}"),        # GitHub Token
    re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),      # Slack Token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]
# key = value 形式：只打码值，保留键名，模型仍能理解"这里配置了一个密钥"。
# 值必须"像密钥"：仅由密钥常见字符组成且含数字 —— 避免把代码表达式
# （如 password = get_password_from_vault()、self.api_key = settings.api_key）误打码，
# 否则代码 Agent 读到的代码就是残缺的。脱敏永远是"漏报 vs 误报"的权衡。
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b[\w.-]*(?:api[_-]?key|secret|token|password|passwd|pwd)[\w.-]*\b\s*[:=]\s*['\"]?)"
    r"(?=[A-Za-z0-9_\-./+=]*\d)([A-Za-z0-9_\-./+=]{8,})(?![\w(])"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """返回 (脱敏后的文本, 打码次数)。"""
    count = 0

    def _sub_token(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return _REDACTED

    def _sub_assignment(m: re.Match[str]) -> str:
        nonlocal count
        if m.group(2) == _REDACTED:
            return m.group(0)
        count += 1
        return m.group(1) + _REDACTED

    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(_sub_token, text)
    text = _ASSIGNMENT_PATTERN.sub(_sub_assignment, text)
    return text, count
