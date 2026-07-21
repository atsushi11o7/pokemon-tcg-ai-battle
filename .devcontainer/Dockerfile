FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/workspace \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# GitHub CLI
RUN mkdir -p -m 755 /etc/apt/keyrings \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# uv インストール + Python 3.12 を uv 管理下にインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && /root/.local/bin/uv python install 3.12

WORKDIR /workspace

# layer cache 用に依存定義を先にコピー
COPY pyproject.toml ./
COPY uv.loc[k] ./

# uv で Python + 依存をインストール
# キャッシュマウントでuvのダウンロード/ビルドキャッシュをホストに永続化
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project; \
    else \
        uv sync --no-install-project; \
    fi

# Jupyter カーネル登録
RUN /opt/venv/bin/python -m ipykernel install \
    --name ptcg \
    --display-name "Python (ptcg)" \
    --sys-prefix

# Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

CMD ["bash"]