# 知识库服务镜像：仅包含 API + UI 层（推理走宿主 Windows 的 LMStudio）
FROM python:3.12-slim

WORKDIR /app

# 系统依赖（pypdf / 部分包可能需编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认启动 API；UI 通过 docker-compose 的 command 覆盖或另起服务
EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
