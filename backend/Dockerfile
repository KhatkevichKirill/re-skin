# syntax=docker/dockerfile:1
#
# Two stages (AICP-153 / AICP-158):
#   builder — has the compiler toolchain, installs the hashed lock into a venv
#   runtime — only ffmpeg/fonts/runtime libs + that venv; no compilers, non-root
#
# The base is pinned by digest so a rebuild is reproducible. To move it, pull
# the new tag, take its digest from `docker image inspect --format
# '{{json .RepoDigests}}'`, and regenerate backend/requirements.lock with the
# same Python minor (see the header of backend/requirements.txt).
ARG PYTHON_IMAGE=python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# ---------------------------------------------------------------- builder ---
FROM ${PYTHON_IMAGE} AS builder

# build-essential only matters if pip ever has to fall back to an sdist; every
# package in the lock currently ships a cp312 manylinux wheel. It never reaches
# the runtime stage either way.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libseccomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY backend/requirements.lock /tmp/requirements.lock
# --require-hashes: every artifact (transitives included) must match the lock.
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

# Native policy is compiled once and remains root-owned, without setuid/caps.
COPY backend/sandbox/media_exec.c /tmp/media_exec.c
RUN mkdir -p /opt/media-sandbox \
    && cc -O2 -Wall -Wextra -Werror -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
       -Wl,-z,relro,-z,now /tmp/media_exec.c -lseccomp -o /opt/media-sandbox/reskin-media-exec \
    && cc -O2 -Wall -Wextra -Werror -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
       -fPIC -shared -Wl,-z,relro,-z,now /tmp/media_exec.c -lseccomp \
       -o /opt/media-sandbox/reskin-media-policy.so \
    && sha256sum /tmp/media_exec.c | cut -d ' ' -f 1 > /opt/media-sandbox/media-source.sha256

# ---------------------------------------------------------------- app -------
# The application image. Every service builds the LAST stage (`runtime`,
# below), which is this stage unchanged; `brand-asr` (AICP-389) adds the
# hashed ASR wheels on top of it for the brand-asr container only.
FROM ${PYTHON_IMAGE} AS app

# Runtime system dependencies: ffmpeg for video processing; libgl1/libglib2.0-0
# for OpenCV (insightface pulls the non-headless opencv-python wheel too).
# Fonts (DejaVu and Noto CJK) are required for libass to render burned captions
# via fontconfig. Missing fonts render as boxes without failing ffmpeg.
# fonts-noto-cjk is ~100 MB (required for Japanese caption rendering).
# libgomp1 is the OpenMP runtime onnxruntime/scikit-image load; it used to
# arrive implicitly with build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-noto-cjk \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. The UID/GID is fixed (10001) because the host's
# bind-mounted ./data and ./secrets must be owned by it — see the AICP-158
# deploy runbook in deploy/container-hardening.md before changing it.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid ${APP_GID} reskin \
    && useradd --system --uid ${APP_UID} --gid ${APP_GID} \
       --home-dir /home/reskin --create-home --shell /usr/sbin/nologin reskin \
    # InsightFace downloads/reads buffalo_l under ~/.insightface (named volume
    # mounted here). Pre-creating it owned by reskin means a FRESH volume
    # inherits that ownership; an existing root-owned one needs the chown in
    # the runbook.
    && mkdir -p /home/reskin/.insightface /app/data \
    && chown -R reskin:reskin /home/reskin /app/data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/media-sandbox/ /usr/local/libexec/
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/reskin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Code stays root-owned and read-only to the runtime user. Byte-compile it at
# build time because the runtime user cannot write __pycache__ under /app.
COPY backend/app/ ./app/
COPY worker/ ./worker/
# AICP-297: the frozen holdout manifest the knowledge importer verifies by sha256.
COPY docs/knowledge/holdout-manifest-v1.json ./docs/knowledge/
RUN python -m compileall -q /app/app /app/worker

# The data/ directory is NOT baked into the image — it is bind-mounted at
# runtime (./data:/app/data) so api, worker and nginx share the same media.

USER reskin:reskin

# Default command: run FastAPI app.  Override in docker-compose for the worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info"]

# ------------------------------------------------------------ asr-deps ---
# AICP-389: the pinned faster-whisper stack of the assistant ASR sidecar
# (backend/requirements-asr-cp312.lock, every wheel hashed; see
# deploy/assistant/ASR-README.md), installed into its own prefix. It is put on
# sys.path only inside the seccomp'd ASR child (assistant_asr.LocalASR), never
# in the application's interpreter.
FROM ${PYTHON_IMAGE} AS asr-deps
COPY backend/requirements-asr-cp312.lock /tmp/requirements-asr.lock
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: \
        --require-hashes --target /opt/asr -r /tmp/requirements-asr.lock

# ----------------------------------------------------------- brand-asr ---
# The brand-asr container (worker/brand_asr_worker.py): the app plus /opt/asr
# and the model manifest. The model itself is a read-only mount (/model).
FROM app AS brand-asr
COPY --from=asr-deps /opt/asr /opt/asr
COPY deploy/assistant/asr-small-manifest.json /opt/asr-small-manifest.json
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CMD ["python", "-m", "worker.brand_asr_worker"]

# ---------------------------------------------------------------- runtime ---
# The default build target of every other service: the app stage as is.
FROM app AS runtime
