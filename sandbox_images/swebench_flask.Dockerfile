# SWE-bench Lite (pallets/flask) 专用离线评测沙箱镜像
# 基于已有 mca-python:3.11 构建，预装 Flask 2.x 纯 Python 依赖，
# 运行时保持 network_mode="none" 彻底断网，通过 PYTHONPATH=/workspace/src 直接加载沙箱内源码。
#
# 构建命令：
#   docker build -t mca-swebench-flask:3.11 -f sandbox_images/swebench_flask.Dockerfile .
FROM mca-python:3.11

USER root
RUN pip install --no-cache-dir \
    werkzeug==2.3.7 \
    jinja2==3.1.2 \
    itsdangerous==2.1.2 \
    click==8.1.3 \
    blinker==1.6.2 \
    markupsafe==2.1.3 \
    pytest==7.4.3

ENV PYTHONPATH=/workspace/src
USER agent
WORKDIR /workspace
