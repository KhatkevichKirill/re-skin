# Standalone offline ASR qualification/runtime image, never used by API workers.
# Supply the reviewed existing base image digest explicitly at build time.
ARG ASR_BASE_IMAGE
FROM ${ASR_BASE_IMAGE}
USER root
COPY media-source.sha256 /tmp/expected-media-source.sha256
# Refuse older bases at build time: adapters require the exact reviewed ABI.
RUN test -x /usr/local/libexec/reskin-media-exec \
    && test -r /usr/local/libexec/reskin-media-policy.so \
    && cmp /tmp/expected-media-source.sha256 /usr/local/libexec/media-source.sha256 \
    && rm /tmp/expected-media-source.sha256
COPY wheels /wheels
COPY requirements-asr-cp312.lock /requirements-asr.lock
RUN python -m pip install --disable-pip-version-check --no-cache-dir --no-index --find-links /wheels --require-hashes --target /opt/asr -r /requirements-asr.lock && rm -rf /wheels
COPY assistant_asr_worker.py /opt/assistant_asr_worker.py
COPY asrapp/ /app/app/
COPY asr-small-manifest.json /opt/asr-small-manifest.json
ENV PYTHONPATH=/opt/asr HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
USER 10001:10001
ENTRYPOINT ["python", "-m", "app.assistant_asr_service"]
