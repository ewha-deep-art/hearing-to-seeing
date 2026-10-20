# The worker (`hearing-to-seeing-worker`): converts the clips the web app on
# Vercel queues. Runs as an unprivileged user with no host files mounted.
# Started with compose.yaml.
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

# Below the STT server sharing the machine.
CMD ["nice", "-n", "10", ".venv/bin/hearing-to-seeing-worker"]
