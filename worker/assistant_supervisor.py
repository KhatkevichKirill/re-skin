"""Run as the existing unprivileged application UID, separately from RQ."""
from pathlib import Path
import logging
import os
import signal
import threading


def main():
    from app.assistant_access import ProjectAllowlist
    from app.assistant_broker_client import BrokerClient
    from app.assistant_store import AssistantArtifactStorage
    from app.assistant_supervisor import AssistantSupervisor
    from app.assistant_turns import AssistantTurnService
    from app.config import settings
    from app.db import SessionLocal
    from app.storage import data_dir_root

    logging.basicConfig(level=logging.INFO)
    if getattr(settings, "ASSISTANT_ENABLED", False) is not True:
        raise SystemExit("Assistant is disabled")
    policy = ProjectAllowlist.from_settings(settings)
    if not policy.has_scope:
        raise SystemExit("Assistant project scope is empty or invalid")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    service = AssistantTurnService(storage=AssistantArtifactStorage(data_dir_root()), policy=policy, enabled=True,
        agent_enabled=getattr(settings, "ASSISTANT_AGENT_ENABLED", False))
    enricher = None
    asr_socket = getattr(settings, "ASSISTANT_ASR_SOCKET", "")
    if asr_socket:
        from app.assistant_asr_service import ASRSidecarClient
        from app.assistant_enrichment import AssistantContextEnricher
        enricher = AssistantContextEnricher(SessionLocal, service=service,
            asr=ASRSidecarClient(Path(asr_socket), service.storage.root))
    media_factory = None
    if getattr(settings, "ASSISTANT_AGENT_ENABLED", False):
        from app.assistant_agent_media import AgentMediaService
        agent_asr = ASRSidecarClient(Path(asr_socket), service.storage.root) if asr_socket else None
        def media_factory(*, context, cancelled):
            return AgentMediaService(SessionLocal, storage=service.storage, policy=policy,
                target_version_id=context.base_version_id, asr=agent_asr, cancelled=cancelled,
                timeout_sec=1200)
    supervisor = AssistantSupervisor(SessionLocal,
        service=service, enricher=enricher, media_factory=media_factory,
        edit_timeout_sec=getattr(settings, "ASSISTANT_EDIT_TIMEOUT_SEC", 600),
        broker=BrokerClient(Path(getattr(settings, "ASSISTANT_BROKER_SOCKET", "/run/reskin-assistant/broker.sock"))),
        owner=f"supervisor-{os.getpid()}", stopping=stop.is_set)
    while not stop.is_set():
        worked = supervisor.run_once()
        if not worked:
            stop.wait(1)


if __name__ == "__main__":
    main()
