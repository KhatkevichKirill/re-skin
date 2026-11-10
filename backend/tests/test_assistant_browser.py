"""Real browser checks with an intercepted API; no model, production or paid calls."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from jinja2 import Template

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).parents[1] / "app"
RUN = "11111111-1111-4111-8111-111111111111"
V1 = "22222222-2222-4222-8222-222222222222"
V2 = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def ui():
    state = dict(run_id=RUN, selected_version_id=V2, latest_version_id=V2, selection_epoch=1,
        versions=[dict(id=v, sequence=i, status="succeeded", fps="30/1", full_url=f"/media/{v}", preview_url=f"/preview/{v}") for i,v in enumerate([V1,V2],1)],
        messages=[], turns=[], jobs=[], version_capabilities=[], preparation_receipts=[], run_result=dict(revision="original",version_id=None,drive_available=True,pending=None,history=[]))
    calls=[]
    def main_result():
        url=f'/api/v2/runs/{RUN}/result?v={state["run_result"]["revision"]}'
        return f'<video id="main-result" src="{url}"></video><a id="main-download" href="{url}" download>Download MP4</a>'
    html='<meta charset="utf-8"><script src="/static/htmx.min.js" defer></script><script src="/static/run_status.js" defer></script>'+Template((ROOT/"templates/v2/partials/run_assistant.html").read_text()).render(run={"id":RUN})+f'<div id="run-status-content" data-run-id="{RUN}">'+main_result()+'</div>'
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={"width":390,"height":844})
        def route(request):
            url=request.request.url
            if url.endswith("/static/assistant.js"):
                return request.fulfill(content_type="application/javascript",body=(ROOT/"static/assistant.js").read_text())
            if url.endswith("/static/htmx.min.js"):
                return request.fulfill(content_type="application/javascript",body=(ROOT/"static/htmx.min.js").read_text())
            if url.endswith("/static/run_status.js"):
                return request.fulfill(content_type="application/javascript",body=(ROOT/"static/run_status.js").read_text())
            if url.endswith(f"/v2/runs/{RUN}/status-fragment"):
                calls.append(("/status-fragment",None))
                return request.fulfill(content_type="text/html",body=main_result())
            if f"/api/v2/runs/{RUN}/result?" in url:
                return request.fulfill(status=204)
            if "/api/v2/assistant/" in url:
                parsed=urlsplit(url)
                path=parsed.path.split(RUN,1)[1]
                body=request.request.post_data_json if request.request.method=="POST" else None
                calls.append((path+('?' + parsed.query if parsed.query else ''),body))
                if path=="/select":
                    assert body["expected_epoch"]==state["selection_epoch"]
                    state["selected_version_id"]=body["version_id"]; state["selection_epoch"]+=1
                if path=="/turns":
                    state["messages"].append(dict(id="m1",role="operator",content=body["instruction"]))
                    state["turns"].append(dict(id="t1",status="queued",base_version_id=body["base_version_id"],operator_message_id="m1",assistant_message_id=None,edit_job_id=None,reply=None))
                if path=="/turns/t1/cancel": state["turns"][0]["status"]="cancelled"
                if path=="/turns/t1/accept":
                    state["turns"][0]["edit_job_id"]="j1"
                    state["turns"][0]["accepted_operation_indexes"]=body.get("operation_indexes",list(range(len((state["turns"][0].get("reply",{}).get("plan") or {}).get("operations",[])))))
                    state["jobs"]=[dict(id="j1",turn_id="t1",status="queued",phase="resolve")]
                if path=="/result":
                    assert body["expected_selection_epoch"]==state["selection_epoch"]
                    assert body["expected_result_revision"]==state["run_result"]["revision"]
                    receipt=dict(id="r1",version_id=body["version_id"],idempotency_key=body["idempotency_key"],created_at="2026-10-27T12:00:00Z",
                        before_url=f"/api/v2/assistant/runs/{RUN}/result-history/r1/before",full_url=f"/api/v2/assistant/runs/{RUN}/result-history/r1/after",
                        delivery="drive_updated" if state["run_result"]["drive_available"] else "local_only",is_current=True,duration_sec=10,noop=False)
                    state["run_result"].update(revision="published-"+body["idempotency_key"],version_id=body["version_id"],pending=None,history=[receipt])
                if path==f"/versions/{V1}/prepare-editable":
                    prepared=next((version for version in state["versions"] if version.get("preparation_source_version_id")==V1),None)
                    if not prepared:
                        assert body["expected_selection_epoch"]==state["selection_epoch"]
                        prepared=editable_copy()
                        state["versions"].append(prepared)
                        state["versions"][0]["prepared_version_id"]=V2
                        state.update(selected_version_id=V2,latest_version_id=V2)
                    state["preparation_receipts"].append(dict(idempotency_key=body["idempotency_key"],source_version_id=V1,version_id=V2,created_at="2026-10-27T12:00:00Z"))
                result=state if path=="/context" else {}
                if path=="/context":
                    bound_ids=set(parse_qs(parsed.query).get("bound_version_ids",[]))
                    bound_ids.update(turn["base_version_id"] for turn in state["turns"])
                    pool={version["id"]:version for version in state["versions"]+state["version_capabilities"]}
                    capabilities=[dict(id=version["id"],sequence=version["sequence"],status=version["status"],needs_editing_preparation=version.get("needs_editing_preparation",False))
                        for version in pool.values() if version["id"] in bound_ids]
                    result={**state,"version_capabilities":capabilities}
                if path=="/result": result=receipt
                if path==f"/versions/{V1}/prepare-editable": result=prepared
                return request.fulfill(content_type="application/json",body=json.dumps(result))
            if "/media/" in url or "/preview/" in url: return request.fulfill(status=204)
            return request.fulfill(content_type="text/html",body=html)
        page.route("**/*",route)
        page.goto("https://assistant.test/")
        yield page,state,calls
        browser.close()


def open_chat(page,version=V2):
    page.get_by_role("button",name="Open edit chat").click()
    playwright.expect(page.locator('[data-field="panel"]')).to_be_visible()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(version)


def editable_copy():
    return dict(id=V2,sequence=2,status="succeeded",fps="30/1",duration_sec=29.833333,
        full_url=f"/media/{V2}",preview_url=f"/preview/{V2}",needs_editing_preparation=False,preparation_source_version_id=V1)


def original_needs_preparation(state,existing_copy=False):
    original=state["versions"][0]
    original.update(needs_editing_preparation=True,prepared_version_id=V2 if existing_copy else None,
        editing_preparation=dict(kind="audio_tail",added_frames=3,added_duration_sec=0.1,target_duration_sec=29.833333,audio_copied=True))
    state["versions"]=[original]+([editable_copy()] if existing_copy else [])
    state.update(selected_version_id=V1,latest_version_id=V2 if existing_copy else V1)


def test_analysis_preview_discloses_sampling_and_subset_scope(ui):
    page, state, calls = ui
    state['turns'] = [dict(id='t1',status='succeeded',base_version_id=V2,
        operator_message_id=None,assistant_message_id=None,edit_job_id=None,reply=None,
        analysis={'phase':'final','steps':[{'step_index':0,'status':'completed','action':'inspect_frames',
            'observation':{'frames':[{'timestamp_sec':7.0}]}}],
            'candidate_url':'/media/candidate'})]
    open_chat(page)
    playwright.expect(page.get_by_text('Video analysis · 1 steps')).to_be_visible()
    playwright.expect(page.locator('[data-field="history"] video')).to_have_attribute('src','/media/candidate')
    playwright.expect(page.get_by_text('Trial edit for the complete proposal.',exact=False)).to_be_visible()
    page.get_by_text('Video analysis · 1 steps').click()
    playwright.expect(page.get_by_text('Frames cover sampled moments.',exact=False)).to_be_visible()
    assert not any(path.endswith('/accept') for path,_ in calls)


def test_branch_pin_request_cancel_and_reopen(ui):
    page,state,calls=ui
    open_chat(page)
    page.locator('[data-field="versions"]').select_option(V1)
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V1}")
    page.locator('[data-field="instruction"]').fill("Fade ending <img src=x onerror=alert(1)>")
    page.get_by_role("button",name="Send request").click()
    playwright.expect(page.get_by_text("Request: queued")).to_be_visible()
    request=next(body for path,body in calls if path=="/turns")
    assert request["base_version_id"]==V1 and request["idempotency_key"]
    assert page.locator('[data-field="history"] img').count()==0
    page.get_by_role("button",name="Cancel",exact=True).click()
    playwright.expect(page.get_by_text("Request: cancelled")).to_be_visible()
    page.reload()
    page.get_by_role("button",name="Open edit chat").click()
    playwright.expect(page.get_by_role("button",name="Retry request")).to_be_visible()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V1)


def test_review_proposal_execution_progress_and_comparison(ui):
    page,state,calls=ui
    state["messages"]=[dict(id="m1",role="operator",content="Fade ending"),dict(id="m2",role="assistant",content="Fade the last second; audio preserved.")]
    state["turns"]=[dict(id="t1",status="succeeded",base_version_id=V2,operator_message_id="m1",assistant_message_id="m2",edit_job_id=None,reply={"kind":"propose"})]
    state["versions"][1]["preview_metadata"]={"preview_mode":"before_after","switch_sec":3}
    open_chat(page)
    playwright.expect(page.get_by_role("button",name="Render this edit")).to_be_visible()
    assert not any(path.endswith("/accept") for path,_ in calls)
    page.get_by_text("Compare with an earlier version", exact=True).click()
    page.locator('[data-field="compare-versions"]').select_option(V1)
    playwright.expect(page.locator('[data-field="compare-player"]')).to_have_attribute("src",f"/media/{V1}")
    page.get_by_role("button",name="Render this edit").click()
    playwright.expect(page.get_by_text("Edit: queued · checking sources")).to_be_visible()
    playwright.expect(page.get_by_role("button",name="Cancel",exact=True)).to_be_visible()
    playwright.expect(page.locator('[data-field="preview-note"]')).to_contain_text("Before, then after")
    assert page.locator('[data-field="history"] button').count()==1


@pytest.mark.parametrize('visual,fade,expected', [
    ({'kind':'black'},0,'Use a black picture'),
    ({'kind':'freeze_selected','timestamp_sec':24.958333},0.5,'Any text burned into that frame stays frozen'),
])
def test_extended_ending_discloses_picture_audio_and_caption_changes_before_atomic_acceptance(ui,visual,fade,expected):
    page,state,calls=ui
    state['versions'][1]['duration_sec']=25.708333
    operation=dict(kind='extend_ending',start_sec=25,end_sec=26.5,visual=visual,
        source={'asset_id':'44444444-4444-4444-8444-444444444444','version_id':V1},
        donor_start_sec=25,fade_out_sec=fade)
    state['turns']=[dict(id='t1',status='succeeded',base_version_id=V2,operator_message_id=None,
        assistant_message_id=None,edit_job_id=None,reply={'kind':'propose','plan':{'operations':[operation]}})]
    open_chat(page)
    playwright.expect(page.get_by_text('Replace and extend ending',exact=False)).to_be_visible()
    playwright.expect(page.get_by_text('Final duration: 26.5 seconds. Adds about 0.792 seconds.')).to_be_visible()
    playwright.expect(page.get_by_text(expected,exact=False)).to_be_visible()
    playwright.expect(page.get_by_text('Use source audio from 25 to 26.500 seconds at normal speed.',exact=False)).to_be_visible()
    playwright.expect(page.get_by_text('Existing editable captions in the replaced ending are removed',exact=False)).to_be_visible()
    if fade:
        playwright.expect(page.get_by_text('Fade the held picture to black over the last 0.5 seconds.')).to_be_visible()
    assert not any(path.endswith('/accept') for path,_ in calls)
    assert page.get_by_role('button',name='Apply only change',exact=False).count()==0
    page.get_by_role('button',name='Render this edit',exact=True).click()
    accepted=[body for path,body in calls if path.endswith('/accept')]
    assert accepted==[{}]


def test_retry_response_loss_keeps_same_request_identity_across_reopen(ui):
    page,state,_=ui
    state["messages"]=[dict(id="m1",role="operator",content="Fade ending")]
    state["turns"]=[dict(id="t1",status="failed",base_version_id=V2,operator_message_id="m1",assistant_message_id=None,edit_job_id=None,reply=None)]
    requests=[]
    def retry(route):
        requests.append(route.request.post_data_json)
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Response unavailable"}')
    page.route("**/turns/t1/retry",retry)
    open_chat(page)
    page.get_by_role("button",name="Retry request").click()
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Response unavailable")
    page.reload();open_chat(page)
    page.get_by_role("button",name="Retry request").click()
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Response unavailable")
    assert len(requests)==2
    assert requests[0]["idempotency_key"]==requests[1]["idempotency_key"]


def test_player_marks_preserve_exact_source_frame_boundaries(ui):
    page,state,calls=ui
    state["versions"][1]["fps"]="30"
    open_chat(page)
    page.get_by_text("Select a section or source (optional)",exact=True).click()
    page.locator('[data-field="player"]').evaluate("el => Object.defineProperty(el, 'currentTime', {value:37/30, configurable:true})")
    page.get_by_role("button",name="Mark start").click()
    page.locator('[data-field="player"]').evaluate("el => Object.defineProperty(el, 'currentTime', {value:2, configurable:true})")
    page.get_by_role("button",name="Mark end").click()
    page.locator('[data-field="instruction"]').fill("Remove selected passage")
    page.get_by_role("button",name="Send request").click()
    playwright.expect(page.get_by_text("Request: queued")).to_be_visible()
    body=next(body for path,body in calls if path=="/turns")
    assert "1.2333333333333334 to 2 seconds" in body["instruction"]


def test_lost_submit_response_does_not_rebase_saved_request(ui):
    page,state,_=ui
    requests=[]
    def lost_response(route):
        requests.append(route.request.post_data_json)
        state["selected_version_id"]=V1
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Response unavailable"}')
    page.route("**/runs/"+RUN+"/turns",lost_response)
    open_chat(page)
    page.locator('[data-field="instruction"]').fill("Fade the ending")
    page.get_by_role("button",name="Send request").click()
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Response unavailable")
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V1,timeout=6000)
    page.reload()
    page.get_by_role("button",name="Open edit chat").click()
    playwright.expect(page.get_by_role("button",name="Retry saved request")).to_be_visible()
    page.get_by_role("button",name="Retry saved request").click()
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Response unavailable")
    assert len(requests)==2 and requests[0]==requests[1]
    assert requests[1]["base_version_id"]==V2


def test_older_history_keeps_current_selection_playable(ui):
    page,state,_=ui
    state["history_cursors"]={"before_version_sequence":2,"before_turn_id":None}
    state["versions"]=state["versions"][1:]
    open_chat(page)
    def older(route):
        value={**state,"versions":[dict(id=V1,sequence=1,status="succeeded",fps="30",full_url=f"/media/{V1}",preview_url=f"/preview/{V1}")],
               "selected_version":state["versions"][0],"latest_version":state["versions"][0],"history_cursors":{}}
        route.fulfill(content_type="application/json",body=json.dumps(value))
    page.route("**/context?*",older)
    page.get_by_role("button",name="Earlier history",exact=True).click()
    playwright.expect(page.get_by_text("Showing an earlier page of this conversation.")).to_be_visible()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    assert page.locator('[data-field="versions"] option').count()==2
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V2}")
    page.get_by_role("button",name="Latest history",exact=True).click()
    playwright.expect(page.get_by_text("Showing an earlier page of this conversation.")).not_to_be_visible()


def test_definite_invalid_request_can_be_corrected(ui):
    page,_,_=ui
    page.route("**/runs/"+RUN+"/turns",lambda route: route.fulfill(status=422,content_type="application/json",body='{"detail":"Invalid passage"}'))
    open_chat(page)
    page.locator('[data-field="instruction"]').fill("Invalid draft")
    page.get_by_role("button",name="Send request").click()
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Invalid passage")
    playwright.expect(page.locator('[data-field="instruction"]')).to_be_editable()
    page.locator('[data-field="instruction"]').fill("Corrected draft")
    playwright.expect(page.get_by_role("button",name="Send request")).to_be_enabled()


def test_render_rejection_stays_next_to_action_through_polling_and_can_retry(ui):
    page,state,calls=ui
    state["messages"]=[dict(id="m1",role="operator",content="Fade ending")]
    state["turns"]=[dict(id="t1",status="succeeded",base_version_id=V2,operator_message_id="m1",assistant_message_id=None,edit_job_id=None,reply={"kind":"propose"})]
    pending=[]
    page.route("**/turns/t1/accept",lambda route: pending.append(route))
    open_chat(page)
    player=page.locator('[data-field="player"]')
    player.evaluate("el => { window.savedPlayer = el; window.mediaLoads = 0; el.load = () => window.mediaLoads++; }")
    button=page.get_by_role("button",name="Render this edit",exact=True)
    button.click()
    submitting=page.get_by_role("button",name="Submitting…",exact=True)
    playwright.expect(submitting).to_be_disabled()
    playwright.expect(submitting).to_have_attribute("aria-busy","true")
    playwright.expect(submitting.locator("..").get_by_role("status")).to_have_text("Submitting request…")
    # Even a synthetic second click cannot submit while the request is pending.
    submitting.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert len(pending)==1 and pending[0].request.method=="POST"
    pending[0].fulfill(status=422,content_type="application/json",body='{"detail":"Fade durations exceed this image passage"}')
    feedback=button.locator("..").get_by_role("status")
    playwright.expect(feedback).to_contain_text("Fade durations exceed this image passage")
    playwright.expect(button).to_be_enabled()
    playwright.expect(button).to_have_attribute("aria-busy","false")
    # Force a real polling refresh to rebuild the history, not just reuse its signature.
    state["messages"][0]["content"]="Fade ending (saved)"
    playwright.expect(page.get_by_text("Fade ending (saved)",exact=True)).to_be_visible(timeout=6000)
    playwright.expect(feedback).to_contain_text("Fade durations exceed this image passage")
    assert page.evaluate("window.savedPlayer === document.querySelector('[data-field=player]') && window.mediaLoads === 0")
    # Retry the same acceptance endpoint, which queues one edit job.
    page.unroute("**/turns/t1/accept")
    button.click()
    playwright.expect(page.get_by_text("Edit: queued · checking sources")).to_be_visible()
    playwright.expect(page.get_by_role("button",name="Render this edit",exact=True)).to_have_count(0)
    playwright.expect(page.get_by_role("button",name="Cancel",exact=True)).to_be_enabled()
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_enabled()
    assert len([path for path,_ in calls if path=="/turns/t1/accept"])==1
    assert state["turns"][0]["edit_job_id"]=="j1"


def proposed_pair(state):
    state["messages"]=[dict(id="m1",role="operator",content="Fade out and back in")]
    state["turns"]=[dict(id="t1",status="succeeded",base_version_id=V1,operator_message_id="m1",assistant_message_id=None,edit_job_id=None,
        reply={"kind":"propose","plan":{"operations":[
            {"kind":"image_fade","direction":"out","start_sec":1,"end_sec":2},
            {"kind":"image_fade","direction":"in","start_sec":2,"end_sec":3}]}})]


@pytest.mark.parametrize("choice,indexes",[("Apply all changes",None),("Apply only change 1",[0]),("Apply only change 2",[1])])
def test_proposal_choices_send_selected_operations(ui,choice,indexes):
    page,state,calls=ui
    proposed_pair(state)
    open_chat(page)
    playwright.expect(page.locator('[data-field="history"] ol li')).to_have_text([
        "Fade picture out (1–2 seconds)","Fade picture in (2–3 seconds)"])
    playwright.expect(page.get_by_text("Picture outside each fade interval and audio are preserved.",exact=True)).to_be_visible()
    page.get_by_role("button",name=choice,exact=True).click()
    playwright.expect(page.get_by_text("Edit: queued · checking sources")).to_be_visible()
    body=next(body for path,body in calls if path=="/turns/t1/accept")
    assert body==({} if indexes is None else {"operation_indexes":indexes})
    playwright.expect(page.get_by_text("Selected changes: " + ("1, 2" if indexes is None else str(indexes[0]+1)),exact=True)).to_be_visible()
    playwright.expect(page.get_by_role("button",name="Apply all changes",exact=True)).to_have_count(0)
    assert len(state["turns"][0]["reply"]["plan"]["operations"])==2


def test_custom_option_preserves_draft_and_pins_proposal_base_without_post(ui):
    page,state,calls=ui
    proposed_pair(state)
    open_chat(page)
    instruction=page.locator('[data-field="instruction"]')
    instruction.fill("Keep my draft")
    before=len(calls)
    page.get_by_role("button",name="Custom option",exact=True).click()
    playwright.expect(instruction).to_be_focused()
    playwright.expect(instruction).to_have_value("Keep my draft\n\nCustom option for proposal t1 on version 1 ("+V1+"): describe which changes to combine or adjust.")
    assert not any(body is not None for _,body in calls[before:])
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    page.get_by_text("Select a section or source (optional)",exact=True).click()
    for action in ["start","end","clear-range","sources"]:
        playwright.expect(page.locator('[data-action="'+action+'"]')).to_be_disabled()
    playwright.expect(page.locator('[data-field="sources"]')).to_be_disabled()
    playwright.expect(page.locator('[data-field="composer-base"]')).to_contain_text("This request uses version 1.")
    page.locator('[data-field="player"]').evaluate("el => Object.defineProperty(el, 'currentTime', {value:1, configurable:true})")
    page.locator('[data-action="start"]').evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    state["messages"][0]["content"]="Saved original proposal"
    playwright.expect(page.get_by_text("Saved original proposal",exact=True)).to_be_visible(timeout=6000)
    playwright.expect(page.locator('[data-action="start"]')).to_be_disabled()
    instruction.fill("Apply change 2 more slowly")
    page.get_by_role("button",name="Send request",exact=True).click()
    playwright.expect(page.get_by_text("Request: queued",exact=True)).to_be_visible()
    body=next(body for path,body in calls if path=="/turns")
    assert body["base_version_id"]==V1
    assert body["proposal_turn_id"]=="t1"
    assert body["instruction"]=="Apply change 2 more slowly"
    assert state["turns"][0]["edit_job_id"] is None
    assert "selected_passage" not in body and body["sources"]==[]
    playwright.expect(page.locator('[data-field="composer-base"]')).to_be_hidden()
    playwright.expect(page.locator('[data-action="start"]')).to_be_enabled()
    instruction.fill("Now edit the selected version")
    page.get_by_role("button",name="Send request",exact=True).click()
    playwright.expect(instruction).to_have_value("")
    requests=[body for path,body in calls if path=="/turns"]
    assert len(requests)==2 and requests[1]["base_version_id"]==V2
    assert "proposal_turn_id" not in requests[1]


def test_restored_custom_draft_keeps_proposal_binding_after_correctable_rejection(ui):
    page,state,calls=ui
    proposed_pair(state)
    attempts=[]
    def reject(route):
        attempts.append(route.request.post_data_json)
        code=503 if len(attempts)==1 else 409
        route.fulfill(status=code,content_type="application/json",body=json.dumps({"detail":"Queue unavailable"}))
    page.route("**/runs/"+RUN+"/turns",reject)
    open_chat(page)
    page.get_by_role("button",name="Custom option",exact=True).click()
    instruction=page.locator('[data-field="instruction"]')
    instruction.fill("Only change 2, more slowly")
    page.get_by_role("button",name="Send request",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Retry saved request",exact=True)).to_be_enabled()
    page.reload();open_chat(page)
    page.get_by_role("button",name="Retry saved request",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_enabled()
    playwright.expect(instruction).to_be_editable()
    playwright.expect(page.locator('[data-field="composer-base"]')).to_contain_text("This request uses version 1.")
    page.get_by_text("Select a section or source (optional)",exact=True).click()
    playwright.expect(page.locator('[data-action="start"]')).to_be_disabled()
    playwright.expect(page.locator('[data-field="sources"]')).to_be_disabled()
    assert attempts[0]==attempts[1]
    assert attempts[0]["base_version_id"]==V1 and attempts[0]["proposal_turn_id"]=="t1"
    page.unroute("**/runs/"+RUN+"/turns",reject)
    instruction.fill("Only change 2, slightly more slowly")
    page.get_by_role("button",name="Send request",exact=True).click()
    playwright.expect(instruction).to_have_value("")
    body=next(body for path,body in calls if path=="/turns")
    assert body["base_version_id"]==V1 and body["proposal_turn_id"]=="t1"
    assert body["idempotency_key"]!=attempts[0]["idempotency_key"]
    assert body["sources"]==[] and "selected_passage" not in body


def test_use_selected_earlier_version_refreshes_main_player_and_download_and_keeps_draft(ui):
    page,state,calls=ui
    state["run_result"]["version_id"]=V2
    open_chat(page)
    button=page.get_by_role("button",name="Use as Run result",exact=True)
    playwright.expect(button).to_be_disabled()
    playwright.expect(page.locator('[data-field="versions"] option:checked')).to_have_text("Version 2 (Run result)")
    page.locator('[data-field="versions"]').select_option(V1)
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V1}")
    playwright.expect(button).to_be_enabled()
    playwright.expect(page.locator('[data-field="run-result-warning"]')).to_contain_text("replaces the content behind the existing Google Drive link")
    page.locator('[data-field="instruction"]').fill("Keep this unsent draft")
    button.click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Run result saved")
    body=next(body for path,body in calls if path=="/result")
    assert body["version_id"]==V1 and body["expected_selection_epoch"]==2
    assert body["expected_result_revision"]=="original" and body["idempotency_key"]
    expected=f'/api/v2/runs/{RUN}/result?v=published-'+body["idempotency_key"]
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",expected)
    playwright.expect(page.locator('#main-download')).to_have_attribute("href",expected)
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("Keep this unsent draft")
    playwright.expect(page.locator('[data-field="versions"] option:checked')).to_have_text("Version 1 (Run result)")
    playwright.expect(button).to_be_disabled()
    page.get_by_text("Run result history",exact=True).click()
    playwright.expect(page.get_by_role("link",name="Previous result",exact=True)).to_have_attribute("href",f"/api/v2/assistant/runs/{RUN}/result-history/r1/before")
    playwright.expect(page.get_by_role("link",name="Used result",exact=True)).to_have_attribute("href",f"/api/v2/assistant/runs/{RUN}/result-history/r1/after")
    assert len([path for path,_ in calls if path=="/status-fragment"])==1
    assert not any(path=="/turns" for path,_ in calls)
    assert len(state["versions"])==2 and state["latest_version_id"]==V2


def test_run_result_pending_blocks_duplicate_click_and_discloses_local_delivery(ui):
    page,state,calls=ui
    state["run_result"]["drive_available"]=False
    requests=[]
    page.route("**/runs/"+RUN+"/result",lambda route: requests.append(route))
    open_chat(page)
    playwright.expect(page.locator('[data-field="run-result-warning"]')).to_contain_text("updates the local result only")
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    pending=page.get_by_role("button",name="Setting Run result…",exact=True)
    playwright.expect(pending).to_be_disabled()
    playwright.expect(pending).to_have_attribute("aria-busy","true")
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Updating the Run result")
    pending.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert len(requests)==1
    requests[0].fulfill(status=409,content_type="application/json",body='{"detail":"Another result update is in progress; refresh and retry"}')
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Another result update is in progress")
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_enabled()
    assert not any(path=="/status-fragment" for path,_ in calls)
    playwright.expect(page.locator('#main-download')).to_have_attribute("href",f"/api/v2/runs/{RUN}/result?v=original")


def test_run_result_conflict_refreshes_current_marker_without_automatic_retry(ui):
    page,state,calls=ui
    requests=[]
    def conflict(route):
        requests.append(route.request.post_data_json)
        state["run_result"].update(revision="other-tab",version_id=V2)
        route.fulfill(status=409,content_type="application/json",body='{"detail":"The Run result changed. Review it before trying again."}')
    page.route("**/runs/"+RUN+"/result",conflict)
    open_chat(page)
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("The Run result changed")
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.")
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_disabled()
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",f"/api/v2/runs/{RUN}/result?v=other-tab")
    state["messages"]=[dict(id="m1",role="operator",content="History remains available")]
    state["turns"]=[dict(id="t1",status="succeeded",base_version_id=V2,operator_message_id="m1",assistant_message_id=None,edit_job_id=None,reply=None)]
    playwright.expect(page.get_by_text("History remains available",exact=True)).to_be_visible(timeout=6000)
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("The Run result changed",timeout=6000)
    assert len(requests)==1


def test_ambiguous_run_result_retry_retains_identity_after_reopen_and_selection_change(ui):
    page,state,_=ui
    requests=[]
    def unavailable(route):
        requests.append(route.request.post_data_json)
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Delivery outcome unavailable"}')
    page.route("**/runs/"+RUN+"/result",unavailable)
    open_chat(page)
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Retry Run result request",exact=True)).to_be_enabled()
    state["selected_version_id"]=V1; state["selection_epoch"]=2
    page.reload()
    page.get_by_role("button",name="Open edit chat",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V1)
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("bound to version 2")
    page.get_by_role("button",name="Retry Run result request",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Delivery outcome unavailable")
    assert len(requests)==2 and requests[0]==requests[1]
    assert requests[1]["version_id"]==V2 and requests[1]["expected_selection_epoch"]==1


@pytest.mark.parametrize("status_code",[409,503])
def test_server_pending_run_result_recovers_original_request_without_local_storage(ui,status_code):
    page,state,_=ui
    state["selected_version_id"]=V1
    state["selection_epoch"]=4
    state["run_result"]["pending"]=dict(id="intent1",version_id=V2,created_at="2026-10-27T12:00:00Z",
        idempotency_key="saved-server-request",expected_selection_epoch=1,expected_result_revision="original")
    requests=[]
    def retry(route):
        requests.append(route.request.post_data_json)
        route.fulfill(status=status_code,content_type="application/json",body='{"detail":"Drive reconciliation still required"}')
    page.route("**/runs/"+RUN+"/result",retry)
    page.get_by_role("button",name="Open edit chat",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("bound to version 2")
    page.get_by_role("button",name="Retry Run result request",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Drive reconciliation still required")
    assert requests==[dict(version_id=V2,idempotency_key="saved-server-request",expected_selection_epoch=1,expected_result_revision="original")]


def test_run_result_receipt_reconciles_lost_response_without_duplicate_write(ui):
    page,state,_=ui
    requests=[]
    def lost(route):
        body=route.request.post_data_json; requests.append(body)
        state["run_result"].update(version_id=V2,revision="saved-elsewhere",history=[dict(id="r1",version_id=V2,
            idempotency_key=body["idempotency_key"],is_current=True,delivery="drive_updated",before_url="/media/before",full_url="/media/after")])
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Response lost"}')
    page.route("**/runs/"+RUN+"/result",lost)
    open_chat(page)
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Run result saved",timeout=6000)
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_disabled()
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",f"/api/v2/runs/{RUN}/result?v=saved-elsewhere")
    # Opening saved publication history also repairs a page loaded before the
    # other tab's publication, without replaying the mutation.
    page.reload(); open_chat(page)
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",f"/api/v2/runs/{RUN}/result?v=saved-elsewhere")
    assert len(requests)==1


def test_run_result_requires_selected_completed_version(ui):
    page,state,calls=ui
    state["versions"][1]["status"]="pending"
    page.get_by_role("button",name="Open edit chat",exact=True).click()
    button=page.get_by_role("button",name="Use as Run result",exact=True)
    playwright.expect(button).to_be_disabled()
    button.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert not any(path=="/result" for path,_ in calls)


@pytest.mark.parametrize("receipt_is_current",[True,False])
def test_successful_run_result_waits_for_fresh_context_before_another_publication(ui,receipt_is_current):
    page,state,calls=ui
    if not receipt_is_current:
        state["run_result"]["version_id"]=V1
        def historical_receipt(route):
            body=route.request.post_data_json; calls.append(("/result",body))
            route.fulfill(content_type="application/json",body=json.dumps(dict(id="old-receipt",version_id=V2,
                idempotency_key=body["idempotency_key"],is_current=False)))
        page.route("**/runs/"+RUN+"/result",historical_receipt)
    open_chat(page)
    context_available=False
    def context(route):
        if context_available: route.fallback()
        else: route.fulfill(status=503,content_type="application/json",body='{"detail":"Context unavailable"}')
    page.route("**/runs/"+RUN+"/context*",context)
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    confirming=page.get_by_role("button",name="Confirming Run result…",exact=True)
    playwright.expect(confirming).to_be_disabled()
    playwright.expect(confirming).to_have_attribute("aria-busy","true")
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Run result saved. Reconnecting")
    requests=[body for path,body in calls if path=="/result"]
    assert len(requests)==1
    confirming.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert [body for path,body in calls if path=="/result"]==requests
    context_available=True
    button=page.get_by_role("button",name="Use as Run result",exact=True)
    playwright.expect(button).to_be_visible(timeout=6000)
    if receipt_is_current: playwright.expect(button).to_be_disabled()
    else: playwright.expect(button).to_be_enabled()
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version "+("2" if receipt_is_current else "1")+".")
    playwright.expect(page.locator('[data-field="run-result-status"]')).not_to_contain_text("Reconnecting")
    assert len([path for path,_ in calls if path=="/result"])==1


@pytest.mark.parametrize("stale_after_recovery",[False,True])
def test_context_started_before_publication_cannot_replace_its_reconciled_state(ui,stale_after_recovery):
    page,state,calls=ui
    open_chat(page)
    stale_snapshot=json.loads(json.dumps(state))
    held=[]
    context_available=False
    def context(route):
        if not held: held.append(route)
        elif context_available: route.fallback()
        else: route.fulfill(status=503,content_type="application/json",body='{"detail":"Context unavailable"}')
    page.route("**/runs/"+RUN+"/context*",context)
    # Hold the poll already in flight before the operator publishes.
    with page.expect_request("**/runs/"+RUN+"/context*",timeout=6000):
        page.locator('[data-field="instruction"]').focus()
    page.wait_for_timeout(50)  # Dispatch the intercepted route after its request event.
    assert len(held)==1
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    confirming=page.get_by_role("button",name="Confirming Run result…",exact=True)
    playwright.expect(confirming).to_be_disabled()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Run result saved. Reconnecting")
    if not stale_after_recovery:
        held[0].fulfill(content_type="application/json",body=json.dumps(stale_snapshot))
        page.wait_for_timeout(150)
        playwright.expect(confirming).to_be_disabled()
        confirming.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        assert len([path for path,_ in calls if path=="/result"])==1
    context_available=True
    if stale_after_recovery:
        # A version-selection refresh can finish before the held poll.
        page.locator('[data-field="versions"]').select_option(V2)
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.",timeout=6000)
    if stale_after_recovery:
        held[0].fulfill(content_type="application/json",body=json.dumps(stale_snapshot))
        page.wait_for_timeout(150)
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.")
    playwright.expect(page.locator('[data-field="versions"] option:checked')).to_have_text("Version 2 (Run result)")
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_disabled()
    assert len([path for path,_ in calls if path=="/result"])==1


@pytest.mark.parametrize("lose_context",[False,True])
def test_legacy_status_poll_cannot_restore_old_player_and_controls_after_publication(ui,lose_context):
    page,state,calls=ui
    open_chat(page)
    held=[]
    context_available=not lose_context
    def context(route):
        if context_available: route.fallback()
        else: route.fulfill(status=503,content_type="application/json",body='{"detail":"Context unavailable"}')
    page.route("**/runs/"+RUN+"/context*",context)
    stale_url=f'/api/v2/runs/{RUN}/result?v=original'
    stale_html=f'<video id="main-result" src="{stale_url}"></video><a id="main-download" href="{stale_url}">Download MP4</a><button id="stale-segment-controls">Regenerate segment</button>'
    def fragments(route):
        if not held: held.append(route)
        else: route.fallback()
    page.route("**/v2/runs/"+RUN+"/status-fragment",fragments)
    page.evaluate("""runId => {
      window.statusSwapAttempts = 0;
      document.addEventListener('htmx:beforeSwap', event => {
        if (event.detail.target.id === 'run-status-content') window.statusSwapAttempts++;
      });
      const poller = document.createElement('div');
      poller.id = 'legacy-range-poller';
      poller.setAttribute('hx-get', '/v2/runs/' + runId + '/status-fragment');
      poller.setAttribute('hx-trigger', 'refresh');
      poller.setAttribute('hx-target', '#run-status-content');
      poller.setAttribute('hx-swap', 'innerHTML');
      document.getElementById('run-status-content').append(poller);
      htmx.process(poller);
      htmx.trigger(poller, 'refresh');
    }""",RUN)
    page.wait_for_timeout(50)
    assert len(held)==1
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    if lose_context:
        playwright.expect(page.get_by_role("button",name="Confirming Run result…",exact=True)).to_be_disabled()
        held[0].fulfill(content_type="text/html",body=stale_html)
        page.wait_for_function("window.statusSwapAttempts === 1")
        playwright.expect(page.locator('#stale-segment-controls')).to_have_count(0)
        context_available=True
        page.locator('[data-field="versions"]').select_option(V2)
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.")
    published_url=f'/api/v2/runs/{RUN}/result?v='+state["run_result"]["revision"]
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",published_url)
    # The obsolete legacy response arrives after the publication's refresh.
    if not lose_context:
        held[0].fulfill(content_type="text/html",body=stale_html)
        page.wait_for_function("window.statusSwapAttempts === 2")
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",published_url)
    playwright.expect(page.locator('#main-download')).to_have_attribute("href",published_url)
    playwright.expect(page.locator('#stale-segment-controls')).to_have_count(0)
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.")
    assert len([path for path,_ in calls if path=="/result"])==1


@pytest.mark.parametrize("newer_status",[503,200])
def test_newer_status_response_fences_older_only_after_successful_swap(ui,newer_status):
    page,state,calls=ui
    open_chat(page)
    held=[]
    def fragments(route):
        if not held: held.append(route)
        else:
            url=f'/api/v2/runs/{RUN}/result?v='+state["run_result"]["revision"]+'&status=newer'
            html=f'<video id="main-result" src="{url}"></video><a id="main-download" href="{url}">Download MP4</a>'
            route.fulfill(status=newer_status,content_type="text/html",body=html if newer_status==200 else "Status temporarily unavailable")
    page.route("**/v2/runs/"+RUN+"/status-fragment",fragments)
    page.evaluate("""() => {
      window.fragmentSwapAttempts = 0;
      document.addEventListener('htmx:beforeSwap', event => {
        if (event.detail.target.id === 'run-status-content') window.fragmentSwapAttempts++;
      });
    }""")
    page.get_by_role("button",name="Use as Run result",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-current"]')).to_have_text("Current Run result: version 2.")
    page.wait_for_timeout(50)
    assert len(held)==1
    # A later failed request must not disqualify the valid waiting response;
    # a later successfully applied response must take precedence over it.
    with page.expect_response(lambda response: response.url.endswith("/status-fragment") and response.status==newer_status):
        page.evaluate("""runId => {
          const poller = document.createElement('div');
          poller.setAttribute('hx-get', '/v2/runs/' + runId + '/status-fragment');
          poller.setAttribute('hx-trigger', 'refresh');
          poller.setAttribute('hx-target', '#run-status-content');
          poller.setAttribute('hx-swap', 'innerHTML');
          document.getElementById('run-status-content').append(poller);
          htmx.process(poller);
          htmx.trigger(poller, 'refresh');
        }""",RUN)
    published_url=f'/api/v2/runs/{RUN}/result?v='+state["run_result"]["revision"]
    if newer_status==200:
        playwright.expect(page.locator('#main-result')).to_have_attribute("src",published_url+'&status=newer')
    held[0].fulfill(content_type="text/html",body=f'<video id="main-result" src="{published_url}"></video><a id="main-download" href="{published_url}">Download MP4</a>')
    page.wait_for_function("window.fragmentSwapAttempts === 2")
    expected_url=published_url+('&status=newer' if newer_status==200 else '')
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",expected_url)
    playwright.expect(page.locator('#main-download')).to_have_attribute("href",expected_url)
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_disabled()
    assert len([path for path,_ in calls if path=="/result"])==1


def test_unprepared_original_is_playable_with_persistent_editing_and_publication_gates(ui):
    page,state,calls=ui
    original_needs_preparation(state)
    open_chat(page,V1)
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V1}")
    playwright.expect(page.locator('[data-field="full"]')).to_have_attribute("href",f"/media/{V1}")
    disclosure=page.locator('[data-field="preparation-disclosure"]')
    playwright.expect(disclosure).to_contain_text("3 frames (0.100 seconds)")
    playwright.expect(disclosure).to_contain_text("29.833333 seconds")
    playwright.expect(disclosure).to_contain_text("Picture is re-encoded; all original AAC audio is copied unchanged")
    playwright.expect(disclosure).to_contain_text("original version and main Run result remain unchanged")
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_disabled()
    publish=page.get_by_role("button",name="Use as Run result",exact=True)
    playwright.expect(publish).to_be_disabled()
    for action in ["start","end","clear-range","sources"]:
        playwright.expect(page.locator('[data-action="'+action+'"]')).to_be_disabled()
    # Generic action cleanup and synthetic events must preserve capability gates.
    page.locator('[data-field="versions"]').select_option(V1)
    playwright.expect(page.get_by_role("button",name="Prepare editable copy",exact=True)).to_be_enabled()
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_disabled()
    publish.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    page.locator('[data-field="form"]').evaluate("el => el.dispatchEvent(new Event('submit', {bubbles:true,cancelable:true}))")
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Prepare and select an editable copy")
    assert not any(path=="/turns" or path=="/result" or path.endswith("/prepare-editable") for path,_ in calls)


@pytest.mark.parametrize("profile",["audio_tail","audio_tail_grid_gap"])
def test_preparation_disclosure_distinguishes_gap_fill_and_final_frame_holds(ui,profile):
    page,state,calls=ui
    original_needs_preparation(state)
    if profile=="audio_tail_grid_gap":
        state["versions"][0]["editing_preparation"].update(kind=profile,gap_fill_frames=1,tail_hold_frames=2,
            gap_start_sec=29.7,source_final_pts_sec=892/30,source_declared_duration_sec=892/30,output_frame_count=895)
    open_chat(page,V1)
    disclosure=page.locator('[data-field="preparation-disclosure"]')
    if profile=="audio_tail_grid_gap":
        playwright.expect(disclosure).to_contain_text("Fill 1 missing picture frame at 29.700000 seconds using the preceding frame")
        playwright.expect(disclosure).to_contain_text("retain the original final frame at 29.733333 seconds, then hold it for 2 more frames")
        playwright.expect(disclosure).to_contain_text("All original frame times are preserved")
        playwright.expect(disclosure).to_contain_text("Output: 895 frames, 29.833333 seconds")
        playwright.expect(disclosure).not_to_contain_text("holding the last frame for 3 frames")
    else:
        playwright.expect(disclosure).to_contain_text("holding the last frame for 3 frames (0.100 seconds)")
        playwright.expect(disclosure).not_to_contain_text("missing picture frame")
    playwright.expect(disclosure).to_contain_text("Picture is re-encoded; all original AAC audio is copied unchanged")
    playwright.expect(disclosure).to_contain_text("original version and main Run result remain unchanged")
    playwright.expect(page.get_by_role("button",name="Prepare editable copy",exact=True)).to_be_enabled()
    assert not any(path.endswith("/prepare-editable") or path=="/result" for path,_ in calls)
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="send"]')).to_be_enabled()
    assert len([path for path,_ in calls if path.endswith("/prepare-editable")])==1
    assert not any(path in ["/result","/status-fragment","/turns"] for path,_ in calls)


def test_gap_preparation_retry_preserves_disclosure_and_exact_request_after_reopen(ui):
    page,state,calls=ui
    original_needs_preparation(state)
    state["versions"][0]["editing_preparation"].update(kind="audio_tail_grid_gap",gap_fill_frames=1,tail_hold_frames=2,
        gap_start_sec=29.7,source_final_pts_sec=892/30,source_declared_duration_sec=892/30,output_frame_count=895)
    requests=[]
    def unavailable(route):
        requests.append(route.request.post_data_json)
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Preparation could not be confirmed"}')
    page.route("**/prepare-editable",unavailable)
    open_chat(page,V1)
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Retry preparation request",exact=True)).to_be_enabled()
    page.reload();open_chat(page,V1)
    playwright.expect(page.locator('[data-field="preparation-disclosure"]')).to_contain_text("retain the original final frame at 29.733333 seconds, then hold it for 2 more frames")
    page.get_by_role("button",name="Retry preparation request",exact=True).click()
    playwright.expect(page.locator('[data-field="preparation-status"]')).to_contain_text("Preparation could not be confirmed")
    assert len(requests)==2 and requests[0]==requests[1]
    assert not any(path in ["/result","/status-fragment","/turns"] for path,_ in calls)


def test_preparation_preserves_draft_and_main_then_original_requires_explicit_copy_selection(ui):
    page,state,calls=ui
    original_needs_preparation(state)
    open_chat(page,V1)
    page.locator('[data-field="instruction"]').fill("Keep this draft for the editable copy")
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_enabled()
    playwright.expect(page.get_by_role("button",name="Use as Run result",exact=True)).to_be_enabled()
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("Keep this draft for the editable copy")
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",f"/api/v2/runs/{RUN}/result?v=original")
    requests=[body for path,body in calls if path.endswith("/prepare-editable")]
    assert len(requests)==1 and requests[0]["expected_selection_epoch"]==1 and requests[0]["idempotency_key"]
    assert state["versions"][0]["needs_editing_preparation"] and len(state["versions"])==2
    assert not any(path in ["/result","/status-fragment","/turns"] for path,_ in calls)
    page.locator('[data-field="versions"]').select_option(V1)
    playwright.expect(page.get_by_role("button",name="Select editable copy",exact=True)).to_be_enabled()
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_disabled()
    page.reload();open_chat(page,V1)
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V1}")
    page.get_by_role("button",name="Select editable copy",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    assert len([path for path,_ in calls if path.endswith("/prepare-editable")])==1


@pytest.mark.parametrize("detail",[
    "Preparation response unavailable",
    "Preparation could not be confirmed; retry the same saved preparation request",
])
def test_pending_preparation_keeps_exact_identity_across_reopen_and_selection_change(ui,detail):
    page,state,calls=ui
    original_needs_preparation(state)
    requests=[]
    def lost(route):
        requests.append((route.request.url,route.request.post_data_json))
        route.fulfill(status=503,content_type="application/json",body=json.dumps({"detail":detail}))
    page.route("**/prepare-editable",lost)
    open_chat(page,V1)
    page.locator('[data-field="instruction"]').fill("Keep my draft through preparation")
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Retry preparation request",exact=True)).to_be_enabled()
    page.locator('[data-field="instruction"]').fill("Keep my updated draft through preparation")
    state["versions"].append(dict(id=V2,sequence=2,status="succeeded",fps="30/1",full_url=f"/media/{V2}",preview_url=f"/preview/{V2}"))
    state.update(selected_version_id=V2,latest_version_id=V2,selection_epoch=2)
    page.reload();open_chat(page,V2)
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("Keep my updated draft through preparation")
    playwright.expect(page.locator('[data-field="preparation-status"]')).to_contain_text("request uses version 1")
    assert len(requests)==1  # Fresh context must not retry an unknown commit automatically.
    page.get_by_role("button",name="Retry preparation request",exact=True).click()
    playwright.expect(page.locator('[data-field="preparation-status"]')).to_contain_text(detail)
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("Keep my updated draft through preparation")
    assert len(requests)==2 and requests[0]==requests[1]
    assert requests[0][0].endswith(f"/versions/{V1}/prepare-editable") and requests[0][1]["expected_selection_epoch"]==1
    assert requests[0][1]["idempotency_key"]
    assert not any(path in ["/result","/status-fragment","/turns"] for path,_ in calls)


def test_preparation_pending_blocks_duplicate_click_and_known_success_waits_for_context(ui):
    page,state,calls=ui
    original_needs_preparation(state)
    held=[]
    page.route("**/prepare-editable",lambda route: held.append(route))
    open_chat(page,V1)
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    pending=page.get_by_role("button",name="Preparing editable copy…",exact=True)
    playwright.expect(pending).to_be_disabled()
    playwright.expect(pending).to_have_attribute("aria-busy","true")
    pending.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert len(held)==1
    page.route("**/context*",lambda route: route.fulfill(status=503,content_type="application/json",body='{"detail":"Context unavailable"}'))
    held[0].fulfill(content_type="application/json",body=json.dumps(editable_copy()))
    select=page.get_by_role("button",name="Select editable copy",exact=True)
    playwright.expect(select).to_be_disabled()
    playwright.expect(page.locator('[data-field="preparation-status"]')).to_contain_text("Reconnecting")
    page.locator('[data-action="prepare-editable"]').evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    assert len(held)==1
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_disabled()
    assert not any(path=="/result" for path,_ in calls)


def test_preparation_receipt_resolves_lost_response_without_reselection_or_second_render(ui):
    page,state,_=ui
    original_needs_preparation(state)
    requests=[]
    def lost(route):
        body=route.request.post_data_json;requests.append(body)
        state["versions"].append(editable_copy());state["versions"][0]["prepared_version_id"]=V2
        state.update(latest_version_id=V2,selection_epoch=2)
        state["preparation_receipts"].append(dict(idempotency_key=body["idempotency_key"],source_version_id=V1,version_id=V2,created_at="2026-10-27T12:00:00Z"))
        route.fulfill(status=503,content_type="application/json",body='{"detail":"Response lost"}')
    page.route("**/prepare-editable",lost)
    open_chat(page,V1)
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Select editable copy",exact=True)).to_be_enabled(timeout=6000)
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V1)
    playwright.expect(page.get_by_role("button",name="Send request",exact=True)).to_be_disabled()
    assert len(requests)==1


@pytest.mark.parametrize("binding",["custom","saved"])
def test_unprepared_effective_request_base_is_blocked_even_with_ready_selected_player(ui,binding):
    page,state,calls=ui
    original_needs_preparation(state,existing_copy=True)
    state["selected_version_id"]=V2
    proposed_pair(state)
    if binding=="saved":
        body=dict(base_version_id=V1,instruction="Saved original-base edit",sources=[],idempotency_key="saved-unprepared-request")
        page.evaluate("value => sessionStorage.setItem('reskin-assistant-pending:'+value.run,JSON.stringify(value.pending))",dict(run=RUN,pending=dict(body=body,draft=body["instruction"])))
        page.reload()
    open_chat(page,V2)
    if binding=="custom":
        page.get_by_role("button",name="Custom option",exact=True).click()
        page.locator('[data-field="instruction"]').fill("Edit the original proposal")
    playwright.expect(page.locator('[data-field="send"]')).to_be_disabled()
    playwright.expect(page.get_by_role("button",name="Apply all changes",exact=True)).to_be_disabled()
    page.locator('[data-field="form"]').evaluate("el => el.dispatchEvent(new Event('submit', {bubbles:true,cancelable:true}))")
    playwright.expect(page.locator('[data-field="status"]')).to_contain_text("Prepare and select an editable copy")
    playwright.expect(page.locator('[data-field="send"]')).to_be_disabled()
    assert not any(path=="/turns" or path.endswith("/accept") for path,_ in calls)
    draft=page.locator('[data-field="instruction"]').input_value()
    page.get_by_role("button",name="Use selected version for this draft",exact=True).click()
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value(draft)
    playwright.expect(page.locator('[data-field="send"]')).to_be_enabled()
    assert not any(path=="/turns" for path,_ in calls)
    page.locator('[data-field="send"]').click()
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("")
    request=next(body for path,body in calls if path=="/turns")
    assert request["base_version_id"]==V2 and "proposal_turn_id" not in request


def test_preparation_success_invalidates_a_context_poll_started_on_original(ui):
    page,state,calls=ui
    original_needs_preparation(state)
    open_chat(page,V1)
    stale=json.loads(json.dumps(state));held=[]
    def context(route):
        if not held: held.append(route)
        else: route.fallback()
    page.route("**/context*",context)
    with page.expect_request("**/context*",timeout=6000):
        page.locator('[data-field="instruction"]').focus()
    page.wait_for_timeout(50)
    assert len(held)==1
    page.get_by_role("button",name="Prepare editable copy",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="send"]')).to_be_enabled()
    held[0].fulfill(content_type="application/json",body=json.dumps(stale))
    page.wait_for_timeout(150)
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="send"]')).to_be_enabled()
    playwright.expect(page.locator('[data-field="player"]')).to_have_attribute("src",f"/media/{V2}")
    assert not any(path=="/result" or path=="/status-fragment" for path,_ in calls)


def test_existing_prepared_copy_can_be_selected_when_not_on_history_page(ui):
    page,state,calls=ui
    original_needs_preparation(state,existing_copy=True)
    state["versions"]=state["versions"][:1]
    def context(route):
        value=dict(state)
        if state["selected_version_id"]==V2: value["selected_version"]=editable_copy()
        route.fulfill(content_type="application/json",body=json.dumps(value))
    page.route("**/context*",context)
    open_chat(page,V1)
    playwright.expect(page.locator('[data-field="versions"] option')).to_have_count(1)
    page.get_by_role("button",name="Select editable copy",exact=True).click()
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="send"]')).to_be_enabled()
    assert not any(path.endswith("/prepare-editable") for path,_ in calls)


def saved_off_page_request(page,state,*,capability="ready"):
    original=state["versions"][0]
    state["versions"]=state["versions"][1:]
    state["version_capabilities"]=[] if capability=="unknown" else [{**original,"needs_editing_preparation":capability=="unprepared"}]
    body=dict(base_version_id=V1,instruction="Keep the supported historical base",sources=[],idempotency_key="off-page-saved-request")
    page.evaluate("value => sessionStorage.setItem('reskin-assistant-pending:'+value.run,JSON.stringify(value.pending))",
        dict(run=RUN,pending=dict(body=body,draft=body["instruction"])))
    page.reload()
    return body


def test_saved_supported_off_page_base_retries_exact_request_and_keeps_history_cursor(ui):
    page,state,calls=ui
    body=saved_off_page_request(page,state)
    state["history_cursors"]={"before_version_sequence":2,"before_turn_id":None}
    open_chat(page)
    playwright.expect(page.locator('[data-field="versions"] option')).to_have_count(1)
    playwright.expect(page.get_by_role("button",name="Retry saved request",exact=True)).to_be_enabled()
    playwright.expect(page.locator('[data-field="composer-base"]')).to_contain_text("This request uses version 1.")
    page.get_by_role("button",name="Earlier history",exact=True).click()
    playwright.expect(page.get_by_role("button",name="Retry saved request",exact=True)).to_be_enabled()
    queries=[parse_qs(urlsplit(path).query) for path,_ in calls if urlsplit(path).path=="/context"]
    assert any(query.get("bound_version_ids")==[V1] and query.get("before_version_sequence")==["2"] for query in queries)
    page.get_by_role("button",name="Retry saved request",exact=True).click()
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("")
    assert [request for path,request in calls if path=="/turns"]==[body]
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="versions"] option')).to_have_count(1)


@pytest.mark.parametrize("capability",["unprepared","unknown"])
def test_off_page_base_without_confirmed_editability_preserves_saved_request(ui,capability):
    page,state,calls=ui
    body=saved_off_page_request(page,state,capability=capability)
    open_chat(page)
    playwright.expect(page.locator('[data-field="send"]')).to_be_disabled()
    reset=page.get_by_role("button",name="Use selected version for this draft",exact=True)
    if capability=="unprepared":
        playwright.expect(reset).to_be_visible()
        playwright.expect(page.locator('[data-field="editing-capability"]')).to_contain_text("unprepared version")
    else:
        playwright.expect(reset).not_to_be_visible()
        playwright.expect(page.locator('[data-field="editing-capability"]')).to_contain_text("has not been confirmed")
        page.locator('[data-action="use-selected-base"]').evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    saved=page.evaluate("run => JSON.parse(sessionStorage.getItem('reskin-assistant-pending:'+run))",RUN)
    assert saved["body"]==body and saved["draft"]==body["instruction"]
    playwright.expect(page.locator('[data-field="instruction"]')).to_have_value(body["instruction"])
    assert not any(path=="/turns" for path,_ in calls)


def test_pending_publication_recovers_off_page_supported_version_with_same_request(ui):
    page,state,calls=ui
    original=state["versions"][0]
    state["versions"]=state["versions"][1:]
    state["version_capabilities"]=[original]
    body=dict(version_id=V1,expected_selection_epoch=1,expected_result_revision="original",idempotency_key="off-page-publication")
    page.evaluate("value => sessionStorage.setItem('reskin-assistant-result:'+value.run,JSON.stringify(value.body))",dict(run=RUN,body=body))
    page.reload();open_chat(page)
    playwright.expect(page.locator('[data-field="versions"] option')).to_have_count(1)
    page.get_by_role("button",name="Retry Run result request",exact=True).click()
    playwright.expect(page.locator('[data-field="run-result-status"]')).to_contain_text("Run result saved")
    assert [request for path,request in calls if path=="/result"]==[body]
    playwright.expect(page.locator('#main-result')).to_have_attribute("src",f"/api/v2/runs/{RUN}/result?v=published-off-page-publication")


@pytest.mark.parametrize("choice",["render","custom"])
def test_old_proposal_uses_supported_off_page_base_without_expanding_version_page(ui,choice):
    page,state,calls=ui
    state["version_capabilities"]=[state["versions"][0]]
    state["versions"]=state["versions"][1:]
    proposed_pair(state)
    open_chat(page)
    playwright.expect(page.get_by_role("button",name="Apply all changes",exact=True)).to_be_enabled()
    if choice=="render":
        page.get_by_role("button",name="Apply all changes",exact=True).click()
        playwright.expect(page.get_by_text("Edit: queued · checking sources")).to_be_visible()
        assert [body for path,body in calls if path=="/turns/t1/accept"]==[{}]
    else:
        page.get_by_role("button",name="Custom option",exact=True).click()
        page.locator('[data-field="instruction"]').fill("Keep the historical proposal base")
        page.get_by_role("button",name="Send request",exact=True).click()
        playwright.expect(page.locator('[data-field="instruction"]')).to_have_value("")
        body=next(body for path,body in calls if path=="/turns")
        assert body["base_version_id"]==V1 and body["proposal_turn_id"]=="t1"
    playwright.expect(page.locator('[data-field="versions"]')).to_have_value(V2)
    playwright.expect(page.locator('[data-field="versions"] option')).to_have_count(1)
