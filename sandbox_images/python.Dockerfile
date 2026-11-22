# MiniCodeAgent 沙箱基础镜像
# 构建命令：docker build -t mca-python:3.11 -f sandbox_images/python.Dockerfile .
FROM python:3.11-slim

# 预装 git 和 pytest，避免容器断网后无法安装测试工具
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest

# 创建非 root 用户（UID 1000），对应 DockerSandbox 的 user="1000:1000"
RUN useradd -m -u 1000 -s /bin/bash agent \
    && mkdir -p /workspace && chown agent:agent /workspace

USER agent
WORKDIR /workspace
