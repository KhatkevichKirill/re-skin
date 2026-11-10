/* Run-scoped private beta. Render model/user text only through textContent. */
(() => {
  'use strict';
  const root = document.getElementById('run-assistant');
  if (!root) return;
  const field = name => root.querySelector('[data-field="' + name + '"]');
  const base = '/api/v2/assistant/runs/' + encodeURIComponent(root.dataset.runId);
  let snapshot = null, opened = false, busy = false, timer = null, start = null, end = null;
  let historyQuery = '', composerBaseVersion = null, composerProposalTurnId = null;
  let refreshGeneration = 0;
  let versionCapabilities = new Map();
  const actionFeedback = new Map();
  let pendingRequest = null, catalog = [], selectedMedia = null, historySignature = null;
  const pendingKey = 'reskin-assistant-pending:' + root.dataset.runId;
  try { pendingRequest = JSON.parse(sessionStorage.getItem(pendingKey) || 'null'); } catch (_) { pendingRequest = null; }
  const resultKey = 'reskin-assistant-result:' + root.dataset.runId;
  let pendingResult = null, resultBusy = false, resultAwaitingContext = false, resultFeedback = '', resultRevision = null;
  try { pendingResult = JSON.parse(sessionStorage.getItem(resultKey) || 'null'); } catch (_) { pendingResult = null; }
  const preparationKey = 'reskin-assistant-preparation:' + root.dataset.runId;
  let pendingPreparation = null, preparationBusy = false, preparationAwaitingContext = false, preparationFeedback = '';
  const preparedCopies = new Map();
  try { pendingPreparation = JSON.parse(sessionStorage.getItem(preparationKey) || 'null'); } catch (_) { pendingPreparation = null; }
  function savePendingPreparation(value) {
    pendingPreparation = value;
    try { if (value) sessionStorage.setItem(preparationKey, JSON.stringify(value)); else sessionStorage.removeItem(preparationKey); } catch (_) { /* Retain the exact identity in memory. */ }
  }
  function versionReady(versionId) {
    const version = versionCapabilities.get(versionId);
    return !!version && version.status === 'succeeded' && !version.needs_editing_preparation;
  }
  function versionNeedsPreparation(versionId) {
    const version = versionCapabilities.get(versionId);
    return !!version && version.needs_editing_preparation === true;
  }
  function composerBase() {
    return pendingRequest ? pendingRequest.body.base_version_id : composerBaseVersion || (snapshot && snapshot.selected_version_id);
  }
  function updateCapabilities() {
    field('send').disabled = busy || preparationAwaitingContext || !versionReady(composerBase());
    const baseCapability = versionCapabilities.get(composerBase());
    const resetBase = snapshot && versionReady(snapshot.selected_version_id) && composerBase() !== snapshot.selected_version_id && versionNeedsPreparation(composerBase());
    field('editing-capability').textContent = !snapshot || versionReady(composerBase()) ? ''
      : !baseCapability ? 'The base version’s editing capability has not been confirmed. Its saved request identity and draft are preserved; reconnect or reload to continue.'
      : !versionNeedsPreparation(composerBase()) ? 'This request’s base version is not ready for editing. Its saved request identity and draft are preserved.'
      : resetBase ? 'This draft is still bound to an unprepared version. Use selected version for this draft to start a separate request. Earlier requests and proposals remain unchanged.'
      : 'This request’s base version needs an editable copy before editing. Prepare and select a copy to continue; your draft is preserved.';
    const reset = root.querySelector('[data-action="use-selected-base"]');
    reset.hidden = !resetBase; reset.disabled = busy || preparationAwaitingContext;
    renderPreparation(); updateComposerBase(); renderRunResult();
  }
  function renderPreparation() {
    const selected = snapshot && snapshot.versions.find(item => item.id === snapshot.selected_version_id);
    const source = pendingPreparation ? snapshot && snapshot.versions.find(item => item.id === pendingPreparation.version_id) : selected;
    const metadata = pendingPreparation ? pendingPreparation.preparation : source && source.editing_preparation;
    const preparedId = selected && (selected.prepared_version_id || preparedCopies.get(selected.id));
    field('editing-preparation').hidden = !pendingPreparation && !(selected && selected.needs_editing_preparation) && !preparationAwaitingContext;
    const prepare = root.querySelector('[data-action="prepare-editable"]');
    const select = root.querySelector('[data-action="select-prepared"]');
    prepare.hidden = !!preparedId && !pendingPreparation;
    select.hidden = !preparedId || !!pendingPreparation;
    select.disabled = busy || preparationAwaitingContext;
    prepare.disabled = busy || preparationBusy || preparationAwaitingContext || (!pendingPreparation && !(selected && selected.needs_editing_preparation && metadata));
    prepare.textContent = preparationBusy ? 'Preparing editable copy…' : preparationAwaitingContext ? 'Confirming editable copy…' : pendingPreparation ? 'Retry preparation request' : 'Prepare editable copy';
    prepare.setAttribute('aria-busy', String(preparationBusy || preparationAwaitingContext));
    field('preparation-disclosure').textContent = metadata && Number(metadata.gap_fill_frames) > 0
      ? 'Fill ' + metadata.gap_fill_frames + ' missing picture frame at ' + Number(metadata.gap_start_sec).toFixed(6)
        + ' seconds using the preceding frame, retain the original final frame at ' + Number(metadata.source_final_pts_sec).toFixed(6)
        + ' seconds, then hold it for ' + metadata.tail_hold_frames + ' more frames. All original frame times are preserved.'
        + ' Picture is re-encoded; all original AAC audio is copied unchanged. Output: ' + metadata.output_frame_count
        + ' frames, ' + Number(metadata.target_duration_sec).toFixed(6)
        + ' seconds. The original version and main Run result remain unchanged.'
      : metadata
      ? 'Prepare an editable copy by holding the last frame for ' + metadata.added_frames + ' frames (' + Number(metadata.added_duration_sec).toFixed(3)
        + ' seconds), extending the picture to ' + Number(metadata.target_duration_sec).toFixed(6)
        + ' seconds. Picture is re-encoded; all original AAC audio is copied unchanged. The original version and main Run result remain unchanged.'
      : 'The original version remains available for playback. Prepare an editable copy to continue editing.';
    const recovery = pendingPreparation ? 'This saved preparation request uses version ' + (source ? source.sequence : pendingPreparation.version_id)
      + '. Changing the selected version does not change this request.' : '';
    field('preparation-status').textContent = preparationBusy ? 'Preparing the copy locally. This can take up to four minutes; your original and draft are preserved.'
      : [preparationFeedback, preparedId ? 'An editable copy already exists. Select editable copy to use it.' : '', recovery].filter(Boolean).join(' ');
  }
  function savePendingResult(value) {
    pendingResult = value;
    try { if (value) sessionStorage.setItem(resultKey, JSON.stringify(value)); else sessionStorage.removeItem(resultKey); } catch (_) { /* Keep this request identity in memory. */ }
  }
  function savePending(value) {
    pendingRequest = value;
    try { if (value) sessionStorage.setItem(pendingKey, JSON.stringify(value)); else sessionStorage.removeItem(pendingKey); } catch (_) { /* In-memory identity still survives response loss. */ }
    field('instruction').readOnly = !!value;
    field('send').textContent = value ? 'Retry saved request' : 'Send request';
    updateCapabilities();
  }
  const status = text => { field('status').textContent = text; };
  const key = () => crypto.randomUUID();
  const failure = code => ({
    audio_fit_requires_review: 'The source audio could not fit naturally. Choose a wider source window ending in silence, or review another source.',
    local_edit_failed: 'The local edit could not finish. Check the selected passage and source, then retry or describe a different edit.',
    local_edit_timeout: 'The edit reached its processing time limit. Original versions are preserved; retry this edit without re-planning.',
    planning_timeout: 'The assistant reached its time limit. Your versions are preserved; you can retry the request.',
    interrupted: 'Processing was interrupted. You can retry from the same version.',
    planning_failed: 'The assistant could not prepare this edit. Check source availability and retry, or describe a more specific passage.',
    runtime_planner_failed: 'Claude could not complete this request. Check the app login and included subscription capacity before retrying. No paid fallback was used.',
  }[code] || 'Could not complete: ' + code.replaceAll('_', ' '));
  async function api(path, body) {
    const options = {credentials: 'same-origin', cache: 'no-store'};
    if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const response = await fetch(base + path, options);
    const value = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(typeof value.detail === 'string' ? value.detail : 'Request failed (' + response.status + ').');
      error.status = response.status;
      throw error;
    }
    return value;
  }
  async function confirmPending() {
    try { await api('/turns', pendingRequest.body); }
    catch (error) {
      // A definite client rejection may be corrected. Network/5xx response
      // loss retains the exact saved body and key until reconciled.
      if (error.status >= 400 && error.status < 500) {
        // A restored draft still belongs to its original proposal after a
        // correctable rejection releases the saved request identity.
        composerBaseVersion = pendingRequest.body.base_version_id;
        composerProposalTurnId = pendingRequest.body.proposal_turn_id || null;
        savePending(null);
      }
      throw error;
    }
    savePending(null); field('instruction').value = ''; composerBaseVersion = null; composerProposalTurnId = null; updateComposerBase();
    historyQuery = ''; historySignature = null;
  }
  function node(tag, text, parent) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (parent) parent.append(element);
    return element;
  }
  function updateActionFeedback() {
    root.querySelectorAll('[data-feedback-key]').forEach(container => {
      const feedback = actionFeedback.get(container.dataset.feedbackKey) || {};
      const button = container.querySelector('button');
      button.disabled = !!feedback.pending || !!(button.dataset.editingBase && !versionReady(button.dataset.editingBase));
      button.textContent = feedback.pending ? 'Submitting…' : button.dataset.label;
      button.setAttribute('aria-busy', String(!!feedback.pending));
      const message = container.querySelector('[role="status"]');
      message.textContent = feedback.pending ? 'Submitting request…' : feedback.error || '';
      message.hidden = !message.textContent;
    });
  }
  function action(parent, label, fn, feedbackKey, editingBase) {
    const container = node('div', undefined, parent);
    container.dataset.feedbackKey = feedbackKey;
    const button = node('button', label, container);
    button.type = 'button'; button.className = 'btn btn-secondary btn-sm';
    button.dataset.label = label;
    if (editingBase) button.dataset.editingBase = editingBase;
    const message = node('p', '', container);
    message.setAttribute('role', 'status'); message.setAttribute('aria-live', 'polite');
    message.hidden = true;
    button.addEventListener('click', () => perform(fn, feedbackKey));
    updateActionFeedback();
    return button;
  }
  async function perform(fn, feedbackKey) {
    if (busy) return;
    busy = true; updateCapabilities();
    if (feedbackKey) actionFeedback.set(feedbackKey, {pending:true});
    updateActionFeedback();
    try {
      await fn();
      if (feedbackKey) actionFeedback.delete(feedbackKey);
      await refresh();
    }
    catch (error) {
      const message = error.message + ' Your request and completed versions are preserved.';
      status(message);
      if (feedbackKey) actionFeedback.set(feedbackKey, {error:message});
    }
    finally {
      busy = false; updateCapabilities();
      updateActionFeedback();
    }
  }
  function setVideo(element, url) {
    if (element.getAttribute('src') !== url) { element.src = url; element.load(); }
  }
  function renderVersions(data) {
    const complete = data.versions.filter(version => version.status === 'succeeded');
    const compare = field('compare-versions').value;
    for (const name of ['versions', 'compare-versions']) {
      const select = field(name); select.replaceChildren();
      complete.forEach(version => {
        const option = node('option', 'Version ' + version.sequence + (data.run_result && data.run_result.version_id === version.id ? ' (Run result)' : ''), select); option.value = version.id;
      });
    }
    field('versions').value = data.selected_version_id;
    field('compare-versions').value = complete.some(v => v.id === compare) ? compare : (complete[0] || {}).id || '';
    const selected = complete.find(version => version.id === data.selected_version_id);
    if (selected) {
      setVideo(field('player'), selected.full_url); setVideo(field('preview'), selected.preview_url);
      field('full').href = selected.full_url;
      const metadata = selected.preview_metadata || {};
      field('preview-note').textContent = metadata.preview_mode === 'before_after'
        ? 'Before, then after. The edited passage starts at ' + metadata.switch_sec + ' seconds in this preview.'
        : 'Preview of this version. Review picture, speech, and subtitle timing before using the result.';
      if (metadata.preview_truncated) field('preview-note').textContent += ' This preview covers only part of the edit; compare the full versions to review the rest.';
      if (metadata.preview_quality === 'proxy_max_dimension_640') field('preview-note').textContent += ' This comparison uses a smaller picture for playback. Open the full result for its original dimensions.';
      if (selectedMedia !== selected.id) {
        start = end = null; renderRange(); catalog = []; field('sources').replaceChildren(new Option('Let the assistant clarify', ''));
        field('source-player').pause(); field('source-player').removeAttribute('src'); field('source-player').hidden = true;
        selectedMedia = selected.id;
      }
    }
    renderComparison();
  }
  function renderRunResult() {
    const result = snapshot && snapshot.run_result;
    const button = root.querySelector('[data-action="use-run-result"]');
    if (!result) { field('run-result-controls').hidden = true; return; }
    field('run-result-controls').hidden = false;
    const selected = snapshot.versions.find(version => version.id === snapshot.selected_version_id && version.status === 'succeeded');
    const current = snapshot.versions.find(version => version.id === result.version_id);
    field('run-result-current').textContent = result.version_id
      ? 'Current Run result: version ' + (current ? current.sequence : result.version_id) + '.'
      : 'The current Run result is separate from these assistant versions.';
    field('run-result-warning').textContent = 'This makes ' + (pendingResult ? 'the requested version’s' : 'the selected') + ' full video the main Run result, download, and public link.'
      + (result.drive_available ? ' It also replaces the content behind the existing Google Drive link.' : ' No Google Drive file is connected; this updates the local result only.')
      + ' Previous results are retained below. After using an assistant result, use a new or copied Run for segment regeneration.';
    const pendingVersion = pendingResult && snapshot.versions.find(version => version.id === pendingResult.version_id);
    const recovery = pendingResult ? 'An earlier Run result request is bound to version ' + (pendingVersion ? pendingVersion.sequence : pendingResult.version_id)
      + '. Retry that request to confirm its outcome; changing the selection does not change it.' : '';
    button.textContent = resultBusy ? 'Setting Run result…' : resultAwaitingContext ? 'Confirming Run result…' : pendingResult ? 'Retry Run result request' : 'Use as Run result';
    button.disabled = busy || resultBusy || resultAwaitingContext || preparationAwaitingContext || !versionReady(pendingResult ? pendingResult.version_id : selected && selected.id) || (!pendingResult && (!selected || result.version_id === selected.id));
    button.setAttribute('aria-busy', String(resultBusy || resultAwaitingContext));
    field('run-result-status').textContent = resultBusy ? 'Updating the Run result. Keep this request open while the result is saved…' : [resultFeedback, recovery].filter(Boolean).join(' ');
    const receipts = field('run-result-receipts'); receipts.replaceChildren();
    field('run-result-history').hidden = !(result.history || []).length;
    (result.history || []).forEach(receipt => {
      const row = node('p', undefined, receipts);
      const version = snapshot.versions.find(item => item.id === receipt.version_id);
      node('span', 'Version ' + (version ? version.sequence : receipt.version_id) + (receipt.is_current ? ' · current Run result' : '') + ' · ', row);
      const before = node('a', 'Previous result', row); before.href = receipt.before_url; before.target = '_blank'; before.rel = 'noopener';
      node('span', ' · ', row);
      const after = node('a', 'Used result', row); after.href = receipt.full_url; after.target = '_blank'; after.rel = 'noopener';
      if (receipt.delivery === 'local_only') node('span', ' · local result only', row);
    });
  }
  async function refreshMainResult() {
    if (!document.getElementById('run-status-content')) return;
    if (!window.reskinRunStatus) throw new Error('Run result saved. Reload this page to refresh the main player and download.');
    window.reskinRunStatus.invalidate();
    await window.reskinRunStatus.refresh();
  }
  function reconcileRunResult(data) {
    const result = data.run_result;
    if (!result) return false;
    if (pendingResult && (result.history || []).some(receipt => receipt.idempotency_key === pendingResult.idempotency_key)) {
      savePendingResult(null); resultFeedback = 'Run result saved. Previous results remain available in Run result history.';
    }
    if (result.pending) {
      const pending = result.pending;
      savePendingResult({version_id:pending.version_id, expected_selection_epoch:pending.expected_selection_epoch,
        expected_result_revision:pending.expected_result_revision, idempotency_key:pending.idempotency_key});
    }
    const changed = resultRevision === null ? (result.history || []).some(receipt => receipt.is_current) : resultRevision !== result.revision;
    resultRevision = result.revision;
    return changed;
  }
  function renderComparison() {
    const version = snapshot && snapshot.versions.find(v => v.id === field('compare-versions').value);
    if (version) setVideo(field('compare-player'), version.full_url);
  }
  function composerUsesDifferentBase() {
    const baseVersion = pendingRequest ? pendingRequest.body.base_version_id : composerBaseVersion;
    return !!(snapshot && baseVersion && baseVersion !== snapshot.selected_version_id);
  }
  function updateComposerBase() {
    let notice = field('composer-base');
    if (!notice) {
      notice = node('p'); notice.dataset.field = 'composer-base';
      notice.setAttribute('role', 'status');
      field('instruction').before(notice);
    }
    const baseVersion = pendingRequest ? pendingRequest.body.base_version_id : composerBaseVersion;
    notice.hidden = !baseVersion;
    const version = versionCapabilities.get(baseVersion);
    notice.textContent = baseVersion ? 'This request uses version ' + (version ? version.sequence : baseVersion) + '.'
      + (composerUsesDifferentBase() ? ' Section and source controls are unavailable while the player shows another version. Select this version above to use them.' : '') : '';
    root.querySelectorAll('[data-action="start"], [data-action="end"], [data-action="clear-range"], [data-action="sources"]').forEach(button => {
      button.disabled = busy || !versionReady(snapshot && snapshot.selected_version_id) || composerUsesDifferentBase();
    });
    field('sources').disabled = busy || !versionReady(snapshot && snapshot.selected_version_id) || composerUsesDifferentBase();
  }
  function describeOperation(operation) {
    const names = {trim:'Keep passage', cut:'Remove passage', replace_footage:'Replace footage',
      audio_splice:'Replace audio passage', full_track_audio:'Replace full audio track', subtitle_edit:'Edit subtitles',
      extend_ending:'Replace and extend ending'};
    const name = operation.kind === 'image_fade' ? 'Fade picture ' + operation.direction : names[operation.kind] || operation.kind.replaceAll('_', ' ');
    const interval = Number.isFinite(operation.start_sec) && Number.isFinite(operation.end_sec)
      ? ' (' + operation.start_sec + '–' + operation.end_sec + ' seconds)' : '';
    return name + interval;
  }
  function customizeProposal(turn) {
    if (pendingRequest) throw new Error('Confirm the saved request before describing another option.');
    const sequence = (versionCapabilities.get(turn.base_version_id) || {}).sequence;
    const reference = 'Custom option for proposal ' + turn.id + ' on version ' + (sequence || 'from earlier history')
      + ' (' + turn.base_version_id + '): describe which changes to combine or adjust.';
    const instruction = field('instruction');
    instruction.value = instruction.value.trim() ? instruction.value + '\n\n' + reference : reference;
    composerBaseVersion = turn.base_version_id; composerProposalTurnId = turn.id;
    start = end = null; renderRange(); field('sources').value = '';
    field('source-player').pause(); field('source-player').hidden = true;
    updateCapabilities();
    instruction.focus();
    status('Describe your custom option, then send. This request will use the proposal’s original base version.');
  }
  function renderHistory(data) {
    const signature = JSON.stringify([data.messages, data.turns, data.jobs]);
    if (historySignature === signature) return;
    historySignature = signature;
    const history = field('history'); history.replaceChildren();
    if (historyQuery) node('p', 'Showing an earlier page of this conversation.', history);
    else if (data.history_truncated) node('p', 'Showing recent history. Use Earlier history to open older requests and versions.', history);
    const messages = new Map(data.messages.map(message => [message.id, message]));
    data.turns.forEach(turn => {
      const article = node('article', undefined, history); article.style.cssText = 'padding:12px 0;border-bottom:1px solid var(--color-border,#ddd)';
      const original = messages.get(turn.operator_message_id);
      node('strong', 'You', article); node('p', original ? original.content : 'Edit request', article);
      node('small', 'Based on version ' + ((versionCapabilities.get(turn.base_version_id) || {}).sequence || 'from earlier history'), article);
      const reply = messages.get(turn.assistant_message_id);
      if (reply) { node('strong', ' Assistant', article); node('p', reply.content, article); }
      else node('p', 'Request: ' + turn.status.replaceAll('_', ' '), article);
      if (turn.analysis) {
        const analysis = turn.analysis;
        const names = {planning:'Planning the next check', inspect_frames:'Inspecting video frames',
          transcribe_window:'Analyzing speech', search_sources:'Finding source material',
          render_candidate:'Rendering a trial edit', list_projects:'Checking available projects',
          list_runs:'Checking project runs', list_versions:'Checking video versions',
          list_assets:'Checking available media', detect_captions:'Checking for burned captions',
          check_alignment:'Checking picture alignment', compare_frames:'Comparing frames with the source',
          script_captions:'Building captions from the script',
          final:'Preparing findings'};
        const details = node('details', undefined, article);
        node('summary', 'Video analysis · ' + (analysis.steps || []).length + ' steps', details);
        if (['queued','running'].includes(turn.status)) node('p', names[analysis.phase] || 'Analyzing media', article);
        (analysis.steps || []).forEach(step => {
          const observation = step.observation || {};
          let text = String(step.step_index + 1) + '. ' + (names[step.action] || 'Planning') + ' — ' + step.status;
          if (observation.frames && observation.frames.length) text += '. Frames at ' + observation.frames.map(frame => frame.timestamp_sec.toFixed(2)).join(', ') + ' s';
          const qa = observation.qa;
          if (qa && qa.tool === 'detect_captions') text += '. ' + qa.verdict.replaceAll('_',' ') + ' (' + qa.hits + '/' + qa.frames + ' frames)';
          if (qa && qa.tool === 'check_alignment') text += qa.mode === 'timeline'
            ? '. ' + (qa.in_sync ? 'in sync' : 'out of sync') + ' (' + qa.points + ' points, ' + qa.outliers_total + ' outliers)'
            : '. median drift ' + (qa.median_drift_sec === null ? 'n/a' : qa.median_drift_sec.toFixed(2) + ' s');
          if (qa && qa.tool === 'script_captions') text += '. ' + qa.lines.length + ' lines from ' + qa.pieces.length + ' pieces';
          if (qa && qa.tool === 'compare_frames') {
            const measured = qa.samples.filter(sample => sample.diff !== null && sample.diff !== undefined);
            text += measured.length ? '. ' + measured.length + ' of ' + qa.samples.length + ' pairs measured, max difference '
              + Math.max(...measured.map(sample => sample.diff)).toFixed(1) : '. No frame pairs could be measured';
          }
          if (observation.error) text += '. Analysis unavailable: ' + observation.error.replaceAll('_',' ');
          node('p', text, details);
        });
        node('p', 'Frames cover sampled moments. Speech recognition provides estimated text and timing; pronunciation and naturalness still need listening review.', details);
        if (analysis.candidate_url) {
          node('p', 'Trial edit for the complete proposal. Review before applying; choosing only one change produces a different result.', article);
          const player = node('video', undefined, article);
          player.controls = true; player.preload = 'none'; player.style.cssText = 'width:100%;max-height:320px';
          player.src = analysis.candidate_url;
        }
      }
      if ((turn.speech_evidence || []).length || (turn.evidence_limitations || []).length) {
        const evidence = node('details', undefined, article);
        node('summary', 'Speech search coverage and uncertainty', evidence);
        (turn.speech_evidence || []).forEach(item => {
          const uncertain = item.words.filter(word => word.probability < 0.6).length;
          node('p', 'Source ' + item.version_id.slice(0, 8) + ': searched ' + item.source_start_sec.toFixed(2) + '–' + item.source_end_sec.toFixed(2) + ' seconds of ' + item.source_duration_sec.toFixed(2) + ' seconds. Estimated language: ' + item.language.toUpperCase() + '. ' + uncertain + ' words have low confidence. Timing, pronunciation, and speaker still need review.', evidence);
        });
        (turn.evidence_limitations || []).forEach(text => node('p', text, evidence));
      }
      const jobs = (data.jobs || []).filter(job => job.turn_id === turn.id || job.id === turn.edit_job_id);
      jobs.forEach(job => {
        const phase = {resolve:'checking sources', render:'rendering', inspect:'checking result', publish:'saving version'}[job.phase];
        node('p', 'Edit: ' + job.status.replaceAll('_', ' ') + (phase && ['queued','running'].includes(job.status) ? ' · ' + phase : ''), article);
        if (job.explanation) node('p', job.explanation, article);
      });
      if (Array.isArray(turn.accepted_operation_indexes) && turn.accepted_operation_indexes.length) {
        node('p', 'Selected changes: ' + turn.accepted_operation_indexes.map(index => index + 1).join(', '), article);
      }
      const currentJob = jobs.find(job => job.id === turn.edit_job_id);
      if (turn.status === 'succeeded' && turn.reply && turn.reply.kind === 'propose' && !turn.edit_job_id) {
        const operations = turn.reply.plan ? turn.reply.plan.operations : [];
        if (operations.length) {
          const list = node('ol', undefined, article);
          operations.forEach(operation => node('li', describeOperation(operation), list));
          operations.filter(operation => operation.kind === 'extend_ending').forEach(operation => {
            const original = data.versions.find(version => version.id === turn.base_version_id);
            const added = original && Number.isFinite(original.duration_sec)
              ? ' Adds about ' + (operation.end_sec - original.duration_sec).toFixed(3) + ' seconds.' : '';
            node('p', 'Final duration: ' + operation.end_sec + ' seconds.' + added, article);
            const frozen = operation.visual && operation.visual.kind === 'freeze_selected';
            node('p', frozen
              ? 'Hold the selected video frame at ' + operation.visual.timestamp_sec + ' seconds. Any text burned into that frame stays frozen; the picture does not continue speaking.'
              : 'Use a black picture for the replaced ending, including the added time. Burned-in text in that ending is hidden with the picture.', article);
            if (operation.fade_out_sec > 0) node('p', 'Fade the held picture to black over the last ' + operation.fade_out_sec + ' seconds.', article);
            node('p', 'Use source audio from ' + operation.donor_start_sec + ' to '
              + (operation.donor_start_sec + operation.end_sec - operation.start_sec).toFixed(3)
              + ' seconds at normal speed. Speech, music and background sound in the entire replaced ending change.', article);
            node('p', 'Existing editable captions in the replaced ending are removed; new captions are not added. Caption timing may end less than 0.01 seconds early at the boundary. Picture and audio are re-encoded. Review the extended ending and listen to the transition before applying.', article);
          });
          if (operations.some(operation => operation.kind === 'image_fade')) {
            node('p', 'Picture outside each fade interval and audio are preserved.', article);
          }
        }
        operations.filter(operation => operation.source).forEach(operation => {
          const details = node('details', undefined, article);
          node('summary', 'Review the proposed source passage', details);
          node('p', 'Check pronunciation, speaker, delivery, timing, and background sound before rendering. A transcript match does not verify these.', details);
          const player = node('video', undefined, details);
          player.controls = true; player.preload = 'metadata'; player.style.cssText = 'width:100%;max-height:240px';
          const source = operation.source;
          player.src = base + '/assets/' + encodeURIComponent(source.asset_id) + '/media?base_version_id=' + encodeURIComponent(turn.base_version_id) + '&version_id=' + encodeURIComponent(source.version_id)
            + '#t=' + String(operation.donor_start_sec || 0);
        });
        action(article, operations.length > 1 ? 'Apply all changes' : 'Render this edit',
          () => { if (!versionReady(turn.base_version_id)) throw new Error('Prepare an editable copy before rendering edits.'); return api('/turns/' + turn.id + '/accept', {}); }, turn.id + ':accept', turn.base_version_id);
        if (operations.length > 1) operations.forEach((operation, index) => {
          action(article, 'Apply only change ' + (index + 1),
            () => { if (!versionReady(turn.base_version_id)) throw new Error('Prepare an editable copy before rendering edits.'); return api('/turns/' + turn.id + '/accept', {operation_indexes:[index]}); }, turn.id + ':accept:' + index, turn.base_version_id);
        });
        action(article, 'Custom option', () => customizeProposal(turn), turn.id + ':custom');
      }
      if (turn.status === 'succeeded' && turn.reply && turn.reply.kind === 'caption_lines') {
        const script = ((turn.analysis || {}).steps || []).map(step => (step.observation || {}).qa)
          .filter(qa => qa && qa.tool === 'script_captions' && qa.lines_sha256 === turn.reply.lines_sha256).pop();
        if (script) {
          node('p', 'Proposed lines for caption ' + turn.reply.caption_id.slice(0, 8) + ' (' + script.lines.length
            + ' lines built from the segment prompts; not verified speech):', article);
          const list = node('ol', undefined, article);
          script.lines.forEach(line => node('li', line.start.toFixed(2) + '-' + line.end.toFixed(2) + ' s: ' + line.text, list));
          const library = script.pieces.filter(piece => piece.kind === 'library');
          if (library.length) node('p', 'The library clip is taken to be run segment '
            + library.map(piece => piece.run_segment_id.slice(0, 8) + ' of run ' + piece.run_id.slice(0, 8)).join(', ')
            + (script.warnings.includes('library_clip_missing') ? ' (the library clip itself was deleted, so this cannot be checked)' : '')
            + '. Applying confirms that.', article);
          node('p', "Applying replaces this caption job's lines and burns it again: a new captioned file is uploaded to Drive, "
            + 'tail edits already made from this caption keep their old files, and later captions of the same audio may reuse these lines.', article);
          action(article, 'Apply captions', () => api('/turns/' + turn.id + '/apply-captions',
            {attest_library_donor: library.length > 0}), turn.id + ':apply-captions');
        } else node('p', 'The evidence for this caption proposal is unavailable.', article);
      }
      if (['queued', 'running', 'cancel_requested'].includes(turn.status) || (currentJob && ['queued', 'running', 'cancel_requested'].includes(currentJob.status))) {
        action(article, 'Cancel', () => api('/turns/' + turn.id + '/cancel', {}), turn.id + ':cancel');
      }
      if (['failed', 'cancelled'].includes(turn.status)) {
        action(article, 'Retry request', () => { if (!versionReady(turn.base_version_id)) throw new Error('Prepare an editable copy before retrying edits.'); return api('/turns/' + turn.id + '/retry', {idempotency_key: 'retry:' + turn.id}); }, turn.id + ':retry', turn.base_version_id);
      }
      if (currentJob && ['failed', 'cancelled'].includes(currentJob.status)) {
        action(article, 'Retry edit', () => { if (!versionReady(turn.base_version_id)) throw new Error('Prepare an editable copy before retrying edits.'); return api('/turns/' + turn.id + '/retry-edit', {idempotency_key: 'retry-edit:' + currentJob.id}); }, turn.id + ':retry-edit:' + currentJob.id, turn.base_version_id);
      }
      if (turn.error_code || (currentJob && currentJob.error_code)) node('p', failure(turn.error_code || currentJob.error_code), article);
    });
  }
  async function refresh() {
    if (!opened) return;
    const generation = ++refreshGeneration;
    const query = new URLSearchParams(historyQuery);
    query.delete('bound_version_ids');
    const boundIds = [pendingRequest && pendingRequest.body.base_version_id, pendingResult && pendingResult.version_id,
      composerBaseVersion, pendingPreparation && pendingPreparation.version_id].filter(Boolean);
    [...new Set(boundIds)].slice(0, 8).forEach(versionId => query.append('bound_version_ids', versionId));
    const queryString = query.toString();
    let data;
    try { data = await api('/context' + (queryString ? '?' + queryString : '')); }
    catch (error) { if (generation !== refreshGeneration) return; throw error; }
    if (generation !== refreshGeneration) return;
    data.page_versions = data.versions;
    const versions = new Map(data.versions.map(version => [version.id, version]));
    [data.selected_version, data.latest_version].filter(Boolean).forEach(version => versions.set(version.id, version));
    data.versions = [...versions.values()].sort((a,b) => a.sequence - b.sequence);
    versionCapabilities = new Map(data.versions.map(version => [version.id, version]));
    (data.version_capabilities || []).forEach(version => versionCapabilities.set(version.id, version));
    snapshot = data;
    data.versions.forEach(version => { if (version.prepared_version_id) preparedCopies.set(version.id, version.prepared_version_id); });
    data.versions.forEach(version => { if (version.preparation_source_version_id) preparedCopies.set(version.preparation_source_version_id, version.id); });
    (data.preparation_receipts || []).forEach(receipt => preparedCopies.set(receipt.source_version_id, receipt.version_id));
    if (pendingPreparation && (data.preparation_receipts || []).some(receipt => receipt.idempotency_key === pendingPreparation.body.idempotency_key && receipt.source_version_id === pendingPreparation.version_id)) {
      savePendingPreparation(null); preparationFeedback = 'Editable copy is available. The original and main Run result are preserved.';
    }
    preparationAwaitingContext = false;
    if (data.run_result && resultAwaitingContext) {
      resultAwaitingContext = false;
      resultFeedback = 'Run result saved. Previous results remain available in Run result history.';
    }
    const resultChanged = reconcileRunResult(data);
    const cursors = data.history_cursors || {};
    root.querySelector('[data-action="older-history"]').hidden = !cursors.before_version_sequence && !cursors.before_turn_id;
    root.querySelector('[data-action="latest-history"]').hidden = !historyQuery;
    renderVersions(data); renderHistory(data); updateCapabilities(); updateActionFeedback();
    if (resultChanged) {
      try { await refreshMainResult(); }
      catch (_) { resultFeedback = 'Run result saved. Reload this page to refresh the main player and download.'; renderRunResult(); }
    }
  }
  async function poll() {
    if (opened && !busy && !document.hidden) {
      try { await refresh(); } catch (_) { status('Connection interrupted. Reconnecting; your saved history remains available.'); }
    }
    timer = setTimeout(poll, 3000);
  }
  function renderRange() {
    field('range').textContent = start === null && end === null ? '' : 'Section: ' + (start === null ? '?' : start.toFixed(3)) + '–' + (end === null ? '?' : end.toFixed(3)) + ' seconds';
  }
  root.querySelector('[data-action="open"]').addEventListener('click', () => perform(async () => {
    status('Opening saved conversation…'); await api('/import', {}); opened = true;
    field('panel').hidden = false; root.querySelector('[data-action="open"]').hidden = true;
    if (pendingRequest) {
      field('instruction').value = pendingRequest.draft;
      savePending(pendingRequest);
      status('An earlier request may already be saved. Retry saved request keeps its original version and request identity.');
    } else {
      if (pendingPreparation && typeof pendingPreparation.draft === 'string') field('instruction').value = pendingPreparation.draft;
      status('Requests use the selected version. Review proposals and preview each result.');
    }
    if (!timer) timer = setTimeout(poll, 3000);
  }));
  field('versions').addEventListener('change', () => perform(async () => {
    composerBaseVersion = null; composerProposalTurnId = null;
    await api('/select', {version_id: field('versions').value, expected_epoch: snapshot.selection_epoch});
  }));
  root.querySelector('[data-action="use-selected-base"]').addEventListener('click', () => {
    if (busy || preparationAwaitingContext || !snapshot || !versionReady(snapshot.selected_version_id)) return;
    if (!versionNeedsPreparation(composerBase()) || composerBase() === snapshot.selected_version_id) return;
    savePending(null); composerBaseVersion = null; composerProposalTurnId = null;
    start = end = null; renderRange(); field('sources').value = '';
    updateCapabilities();
    status('This draft now uses the selected version. Sending it starts a separate request; earlier requests and proposals are unchanged.');
  });
  root.querySelector('[data-action="prepare-editable"]').addEventListener('click', async () => {
    if (busy || preparationBusy || preparationAwaitingContext || !snapshot) return;
    if (!pendingPreparation) {
      const selected = snapshot.versions.find(version => version.id === snapshot.selected_version_id);
      if (!selected || !selected.needs_editing_preparation || !selected.editing_preparation || selected.prepared_version_id || preparedCopies.has(selected.id)) return;
      savePendingPreparation({version_id:selected.id, preparation:selected.editing_preparation, draft:field('instruction').value,
        body:{expected_selection_epoch:snapshot.selection_epoch, idempotency_key:key()}});
    }
    busy = true; preparationBusy = true; preparationFeedback = ''; updateCapabilities();
    try {
      const sourceId = pendingPreparation.version_id;
      const version = await api('/versions/' + encodeURIComponent(sourceId) + '/prepare-editable', pendingPreparation.body);
      preparedCopies.set(sourceId, version.id);
      savePendingPreparation(null); preparationAwaitingContext = true; refreshGeneration += 1;
      preparationFeedback = 'Editable copy prepared. Review it before editing or using it as the Run result.';
      status(preparationFeedback);
      try { await refresh(); }
      catch (_) { preparationFeedback = 'Editable copy prepared. Reconnecting to confirm the selected version before continuing.'; }
    } catch (error) {
      preparationFeedback = error.message + ' Your original and draft are preserved.';
      status(preparationFeedback);
      if (error.status >= 400 && error.status < 500) {
        savePendingPreparation(null);
        try { await refresh(); } catch (_) { /* Leave the original selected until context is available. */ }
      }
    } finally {
      busy = false; preparationBusy = false; updateCapabilities();
    }
  });
  root.querySelector('[data-action="select-prepared"]').addEventListener('click', () => perform(async () => {
    if (preparationAwaitingContext || !snapshot) return;
    const selected = snapshot.versions.find(version => version.id === snapshot.selected_version_id);
    const preparedId = selected && (selected.prepared_version_id || preparedCopies.get(selected.id));
    if (!preparedId) return;
    await api('/select', {version_id:preparedId, expected_epoch:snapshot.selection_epoch});
  }));
  root.querySelector('[data-action="use-run-result"]').addEventListener('click', async () => {
    if (busy || resultBusy || resultAwaitingContext || !snapshot || !snapshot.run_result) return;
    if (preparationAwaitingContext || !versionReady(pendingResult ? pendingResult.version_id : snapshot.selected_version_id)) return;
    if (!pendingResult) {
      const selected = snapshot.versions.find(version => version.id === snapshot.selected_version_id && version.status === 'succeeded');
      if (!selected || selected.id === snapshot.run_result.version_id) return;
      savePendingResult({version_id:selected.id, expected_selection_epoch:snapshot.selection_epoch,
        expected_result_revision:snapshot.run_result.revision, idempotency_key:key()});
    }
    busy = true; resultBusy = true; resultFeedback = ''; field('send').disabled = true; renderRunResult();
    let saved = false;
    try {
      await api('/result', pendingResult);
      saved = true; resultAwaitingContext = true; refreshGeneration += 1; savePendingResult(null);
      if (window.reskinRunStatus) window.reskinRunStatus.invalidate();
      resultFeedback = 'Run result saved. Previous results remain available in Run result history.';
    } catch (error) {
      resultFeedback = error.message + ' Previous results and assistant versions are preserved.';
      // A definite rejection releases the identity only when the server has
      // no durable pending intent. Ambiguous responses keep an explicit retry.
      if (error.status >= 400 && error.status < 500) {
        try {
          await refresh();
          if (!(snapshot.run_result && snapshot.run_result.pending)) savePendingResult(null);
        } catch (_) { /* Keep the identity until its outcome can be checked. */ }
      }
    }
    if (saved) {
      try { await refresh(); }
      catch (_) { resultFeedback = 'Run result saved. Reconnecting to confirm the current Run result before another update.'; }
    }
    resultBusy = false; busy = false; updateCapabilities();
  });
  field('compare-versions').addEventListener('change', renderComparison);
  root.querySelector('[data-action="older-history"]').addEventListener('click', () => perform(async () => {
    const previous = new URLSearchParams(historyQuery);
    const cursors = snapshot.history_cursors || {};
    const query = new URLSearchParams();
    query.set('before_version_sequence', cursors.before_version_sequence || (snapshot.page_versions[0] || {}).sequence || previous.get('before_version_sequence') || 1);
    const turnId = cursors.before_turn_id || (snapshot.turns[0] || {}).id || previous.get('before_turn_id');
    if (turnId) query.set('before_turn_id', turnId);
    historyQuery = '?' + query.toString(); historySignature = null;
  }));
  root.querySelector('[data-action="latest-history"]').addEventListener('click', () => perform(async () => { historyQuery = ''; historySignature = null; }));
  field('sources').addEventListener('change', () => {
    if (composerUsesDifferentBase() || !versionReady(snapshot && snapshot.selected_version_id)) return;
    const source = field('sources').value === '' ? null : catalog[Number(field('sources').value)];
    field('source-player').hidden = !source || !source.media_url;
    if (source && source.media_url) setVideo(field('source-player'), source.media_url);
  });
  field('instruction').addEventListener('input', () => {
    if (pendingPreparation) { pendingPreparation.draft = field('instruction').value; savePendingPreparation(pendingPreparation); }
  });
  const importCaption = root.querySelector('[data-action="import-caption"]');
  if (importCaption) importCaption.addEventListener('click', () => perform(async () => {
    const captionId = field('caption-source').value;
    if (!captionId) throw new Error('No completed captioned result is available.');
    status('Verifying the captioned result against its clean source…');
    await api('/captions/' + encodeURIComponent(captionId) + '/import', {});
    status('Captioned version imported. Review it in the player before requesting changes.');
  }));
  ['start', 'end'].forEach(boundary => root.querySelector('[data-action="' + boundary + '"]').addEventListener('click', () => {
    if (composerUsesDifferentBase() || !versionReady(snapshot && snapshot.selected_version_id)) return;
    const selected = snapshot && snapshot.versions.find(v => v.id === snapshot.selected_version_id);
    const parts = selected && typeof selected.fps === 'string' ? selected.fps.split('/').map(Number) : [];
    const fps = parts.length === 2 ? parts[0] / parts[1] : (parts.length === 1 ? parts[0] : NaN);
    if (!(fps > 0 && Number.isFinite(fps))) { status('Frame timing is unavailable for this version. Describe the section in your request.'); return; }
    const value = Math.round(field('player').currentTime * fps) / fps;
    if (!Number.isFinite(value)) return;
    if (boundary === 'start') start = value; else end = value;
    renderRange();
  }));
  root.querySelector('[data-action="clear-range"]').addEventListener('click', () => { start = end = null; renderRange(); });
  root.querySelector('[data-action="sources"]').addEventListener('click', () => perform(async () => {
    if (composerUsesDifferentBase() || !versionReady(snapshot && snapshot.selected_version_id)) return;
    status('Finding completed source results in approved projects…');
    const imported = await api('/sources/import', {limit:5});
    const page = await api('/catalog?base_version_id=' + encodeURIComponent(snapshot.selected_version_id));
    catalog = page.artifacts.filter(asset => asset.media_kind === 'video' || asset.media_kind === 'audio');
    field('sources').replaceChildren(new Option('Let the assistant clarify', ''));
    catalog.forEach((asset, index) => field('sources').append(new Option(asset.media_kind + ' · run ' + asset.run_id.slice(0, 8) + ' · ' + (asset.duration_sec || '?') + 's', String(index))));
    field('source-note').textContent = page.next_after_asset_id ? 'Showing the first available sources. Open a source run’s edit chat to pin its result.' : 'Sources are pinned results from approved projects. Open another run’s edit chat to make its result available.';
    status(imported.budget_exhausted || imported.next_after_run_id ? 'Loaded a bounded set of sources. More results can be opened from their run pages.' : 'Available source results loaded.');
  }));
  field('form').addEventListener('submit', event => {
    event.preventDefault(); perform(async () => {
      if (preparationAwaitingContext || !versionReady(composerBase())) throw new Error('Prepare and select an editable copy before sending this request. Your draft is preserved.');
      if (pendingRequest) {
        await confirmPending();
        status('Saved request confirmed. It retains its original base version.');
        return;
      }
      let instruction = field('instruction').value.trim();
      if (!instruction) throw new Error('Describe the requested edit.');
      if (!composerUsesDifferentBase() && (start !== null || end !== null)) {
        if (start === null || end === null || end <= start) throw new Error('Mark a start and a later end, or clear the section.');
        instruction += '\nSelected passage in the base version: ' + String(start) + ' to ' + String(end) + ' seconds (snapped to source frames).';
      }
      const source = composerUsesDifferentBase() || field('sources').value === '' ? null : catalog[Number(field('sources').value)];
      const body = {base_version_id: composerBaseVersion || snapshot.selected_version_id, instruction, sources: source ? [{asset_id: source.asset_id, version_id: source.version_id}] : []};
      if (composerProposalTurnId) body.proposal_turn_id = composerProposalTurnId;
      if (!composerUsesDifferentBase() && start !== null && end !== null) body.selected_passage = {start_sec:start, end_sec:end};
      body.idempotency_key = key();
      savePending({body, draft:field('instruction').value});
      await confirmPending();
      status('Request saved. You can leave and reopen this chat while it runs.');
    });
  });
})();
