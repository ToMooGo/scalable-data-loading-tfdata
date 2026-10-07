# Training pipeline: runs `python run_flow.py --config <config>` against the MLflow and Prefect servers.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MPLBACKEND=Agg \
    HOME=/tmp

WORKDIR /app
COPY services/pipeline/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY flows ./flows
COPY run_flow.py ./

# /app/data holds housing.csv, the shards and the TFDS cache (a named volume); world-writable so the
# host user can fill it
RUN mkdir -p /app/data /app/reports && chmod 777 /app/data

# compose runs this image as the host user (HOST_UID) so ./reports stays yours; HOME=/tmp is writable
CMD ["python", "run_flow.py", "--config", "configs/full_flow_config.yaml"]
