# One image for everything, started with compose.yaml: the worker by default
# (converts the clips the web app on Vercel queues), and the `web` and `cli`
# services. Runs as an unprivileged user.
FROM python:3.12-slim-trixie

# ffmpeg for the audio; node for yt-dlp's YouTube challenge solver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

RUN useradd --create-home worker
USER worker
WORKDIR /home/worker/app

# Dependencies first, so a code change reinstalls only the project.
COPY --chown=worker pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra pipeline --no-install-project
COPY --chown=worker src ./src
RUN uv sync --locked --no-dev --extra pipeline
ENV PATH=/home/worker/app/.venv/bin:$PATH

# Below the STT server sharing the machine.
CMD ["nice", "-n", "10", "python", "-m", "worker.convert"]
