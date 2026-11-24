# SWE-bench Lite 10 题代表集（pallets/flask + pytest-dev/pytest）专用离线 Docker 镜像
# 基于已有 mca-swebench-flask:3.11 增量构建（约 15 秒），全程支持 network_mode="none" 断网评测。
#
# 构建命令：
#   docker build -t mca-swebench-py:3.11 -f sandbox_images/swebench_py.Dockerfile .
FROM mca-swebench-flask:3.11

USER root
RUN pip install --no-cache-dir \
    attrs>=22.2 \
    py>=1.11 \
    pluggy>=1.2 \
    iniconfig>=2.0 \
    packaging>=23.0 \
    tomli>=2.0 \
    exceptiongroup>=1.1 \
    wcwidth>=0.2 \
    xmlschema==2.5.0 \
    hypothesis==6.82.0

ENV PYTHONPATH=/workspace/src
USER agent
WORKDIR /workspace
