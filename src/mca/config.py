"""集中式配置。

所有可调参数都从环境变量 / .env 读取，代码里不出现任何密钥或魔法数字。
后续的沙箱、上下文、预算等配置也会以独立的 Settings 类加在这里。
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM 相关配置，环境变量前缀 MCA_LLM_。"""

    model_config = SettingsConfigDict(
        env_prefix="MCA_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SecretStr：打印 / 记日志时显示为 '**********'，防止密钥意外泄露到 Trace 里
    api_key: SecretStr
    #  base_url: str = "https://api.deepseek.com"
    base_url: str = "https://ws-63tuqwgnvti8179g.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v4.1-flash"

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    timeout_s: float = Field(120.0, gt=0)

    # 重试策略（我们自己实现重试，SDK 内置重试会被关闭）
    max_attempts: int = Field(5, ge=1)
    retry_base_delay_s: float = Field(1.0, gt=0)
    retry_max_delay_s: float = Field(30.0, gt=0)

    # 部分"思考型"模型在 tool calling 多轮中要求把 reasoning_content 原样回传
    send_reasoning_content: bool = False
