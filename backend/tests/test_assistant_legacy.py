"""Legacy imports prove bytes, never infer old renders from mutable options."""
import pytest
from sqlalchemy import select

from tests.test_assistant_executor import setup
from app.assistant_access import ProjectAllowlist
from app.assistant_legacy import import_legacy_run, import_legacy_caption
from app.assistant_models import AssistantVersion
from app.assistant_store import get_version_artifact, verify_artifact, AssistantStoreError
from app.assistant_subtitles import get_subtitle_context
from app.models import Run, RunCaption


def test_import_probes_pinned_duration_and_reopen_ignores_legacy_mutation(setup):
    factory,storage,source,(rid,pid)=setup
    with factory.begin() as session:
        run=session.get(Run,rid)
        run.result_duration_sec=None
        session.flush()
        version=import_legacy_run(session,run,actor="operator",storage=storage,policy=ProjectAllowlist([pid]))
        assert get_version_artifact(session,version.id).metadata_json["duration_sec"]==3
        assert run.result_duration_sec is None
        original=version.id
    source.unlink()
    with factory.begin() as session:
        version=import_legacy_run(session,session.get(Run,rid),actor="operator",storage=storage,policy=ProjectAllowlist([pid]))
        assert version.id==original
        assert verify_artifact(get_version_artifact(session,original),storage=storage).exists()


def caption_fixture(setup):
    from app.captions import build_ass
    from app.media import burn_subtitles
    factory,storage,source,(rid,pid)=setup
    directory=storage.data_root/"caption"
    directory.mkdir()
    ass=directory/"captions.ass"
    ass.write_text(build_ass([dict(start=.5,end=1.5,text="Original line")],width=160,height=120,duration_sec=3,language="en"))
    output=directory/"result.mp4"
    burn_subtitles(str(source),str(ass),str(output),duration_sec=3,has_audio=True)
    with factory.begin() as session:
        caption=RunCaption(run_id=rid,status="done",source_type="run",output_path=str(output),lines=[dict(start=0,end=2,text="Unsaved changed line")])
        session.add(caption);session.flush()
        return caption.id,ass


def test_caption_import_proves_saved_ass_and_ignores_mutated_lines(setup):
    cid,_=caption_fixture(setup)
    factory,storage,_,(rid,pid)=setup
    with factory.begin() as session:
        version=import_legacy_caption(session,session.get(Run,rid),cid,actor="operator",storage=storage,policy=ProjectAllowlist([pid]))
        track=get_subtitle_context(session,version.id,storage=storage)
        assert "Original" in " ".join(c.text for c in track.cues)
        assert "Unsaved" not in " ".join(c.text for c in track.cues)
        assert version.sequence==2
        assert len(session.scalars(select(AssistantVersion)).all())==2


def test_caption_import_rejects_ass_that_does_not_reproduce_output(setup):
    cid,ass=caption_fixture(setup)
    ass.write_text(ass.read_text().replace("Original", "Changed"))
    factory,storage,_,(rid,pid)=setup
    with factory.begin() as session:
        with pytest.raises(AssistantStoreError,match="cannot be reproduced"):
            import_legacy_caption(session,session.get(Run,rid),cid,actor="operator",storage=storage,policy=ProjectAllowlist([pid]))
        assert len(session.scalars(select(AssistantVersion)).all())==0
