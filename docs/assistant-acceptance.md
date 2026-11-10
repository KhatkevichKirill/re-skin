# Assistant acceptance and release evidence

Source: operator-provided ReSkin AI Assistant requirements, 2026-10-23.
Tracking: AICP-261.
This is the verification contract, not a claim that these tests have passed.
Active execution state belongs in Linear.

## Requirements traceability

| Requirement | Implementation responsibility | Required evidence |
| --- | --- | --- |
| Run-scoped chat with settings, assets and previous edits | AICP-264, AICP-267, AICP-271 | Context snapshot identifies the run, exact base, settings, authorized assets and history; browser reopen restores it |
| Latest successful result by default; select and branch from an earlier result | AICP-262, AICP-264, AICP-271 | Concurrent completions, explicit selection and branch tests; failed/cancelled edits never change selection |
| Cross-run/project sources within existing permissions | AICP-265, AICP-269 | Allowed and denied catalog/search/media/worker tests; no cache leakage; actual launch identity documented |
| Optional manual timestamps and uncertainty handling | AICP-267, AICP-271 | Ambiguous source/section produces a candidate or player selection request without speculative execution |
| New preview/full result and brief explanation for every completed edit | AICP-264, AICP-268–271 | Both playable assets belong to the same immutable version; summary describes actual effects |
| Preserve original/prior outputs and unrelated content | AICP-264, AICP-268–270 | Original artifact hashes, Run pointers and prior Drive content unchanged; measured output properties and disclosed differences |
| Audio word/phrase/window/full-track correction | AICP-269 | Annotated alignment fixtures and listening reviews; suitable donor selection, phrase fallback and no-suitable-donor behavior |
| Pronunciation evidence beyond transcript match | AICP-269 | Identical transcript with wrong pronunciation cannot be marked verified; speaker, delivery, background and timing assessed |
| Preserve soundtrack only when supported by assets | AICP-269 | Separate-stem and mixed-track cases; mixed background change is disclosed before use |
| Trim versus image-only fade; CTA/outro and subtitle effects | AICP-268 | Duration, audio packet/timestamp and ending-boundary checks; subtitles and CTA consequences visible |
| Remove artifact by cut, available footage or fade | AICP-268, AICP-267 | Localization uncertainty, speech crossing, duration change and exact subtitle timeline mapping covered |
| Subtitle timing, duplicate/text removal and re-rendering | AICP-270, existing AICP-247 | Cue/style snapshots, clean-source lineage, quick dialogue and multilingual render checks |
| Burned-caption limitations | AICP-268, AICP-270 | Unsupported preservation/replacement returns actionable limitation, not double-burn or claimed reconstruction |
| VPS execution and qualified subscription runtime | AICP-263, AICP-267 | Intended service identity/image, real model availability, auth renewal, isolation and quota behavior proved |
| Restricted tools/assets; external-service budget and permissions | AICP-265–267 | Invalid tool/ID/path/instruction and budget tests; no silent paid fallback; estimate/known charge/unknown charge distinguishable |
| Durable progress, input, completion, failure and cancellation | AICP-266, AICP-271 | Browser disconnect/reopen and worker restart at each lifecycle phase |
| Retry without duplicate paid calls; preserve work at limits | AICP-266 | Lost provider response, Redis loss, resource exhaustion and retry fault injection; ambiguous submissions held for reconciliation |
| Launch verification | AICP-272, AICP-273 | Exact revision, independent Sol-6.1 review, migration/restore rehearsal, pilot results, health checks and rollback evidence |

## Representative fixture corpus

Use authorized synthetic or approved media; record asset hashes and provenance.
No production creative or external paid operation is required merely to build the
test harness. Label expected spans, cues and supported capabilities before running
the test, rather than treating the implementation's output as its own oracle.

1. One correctly pronounced donor and one incorrectly pronounced donor with the
   same transcript; include a wrong speaker and an unsuitable delivery.
2. Single-word replacement where coarticulation makes a phrase replacement more
   natural; repeated matching words to test ambiguity.
3. Early/late timing drift, clipped syllables, silence and a full-track donor that
   starts aligned but drifts at the ending/CTA.
4. Speech mixed with music, independent speech/music stems, and no usable donor.
5. Ending with avatar reappearance, speech crossing a proposed trim, included CTA
   and a separately planned subsequent outro.
6. Image-only fade with audio; clean image plus editable subtitles; burned-only
   subtitles with no clean source.
7. A clearly located artifact and an ambiguous repeated artifact; cuts crossing a
   spoken phrase and cues; duration-preserving replacement and fade.
8. Duplicate caption artifact versus intentionally repeated speech; quick
   consecutive short cues (including the AICP-247 regression), multilingual text,
   historical style, clip-end cues and a captioned result edited again.
9. Unsupported media properties, missing audio, changed/deleted donor, guessed
   asset IDs, revoked access and adversarial instructions inside metadata.
10. Two concurrent edits, a selected earlier version, duplicate request delivery,
    worker interruption, cancellation at publication, Redis loss, quota failure,
    full disk and provider response loss after acceptance.

## Quality criteria

These are initial engineering targets to validate against the corpus. Any change
must be explicit in the release decision and reflected in user-visible limitations.

- Cuts/trims and output duration: within one source video frame of the accepted
  edit plan, with exact timestamp remapping recorded. Variable frame rate requires
  an explicit supported policy, not an assumed constant-frame tolerance.
- Audio-only edits: copied video packets remain identical. Image-only fade and
  subtitle-only render: copied audio packets/timestamps remain identical when the
  accepted container/codec path supports stream copy. Encoding changes must be
  disclosed; source hash preservation alone does not establish perceptual quality.
- Audio splice outside the requested window: initial decoded level tolerance
  0.5 dB, plus timing/signal inspection and listening review. A passing loudness
  check does not prove natural pronunciation, voice identity or an inaudible splice.
- Word alignment: proposed p95 boundary error at most 80 ms on annotated launch
  fixtures. Low confidence still requires review; this proposed threshold is not a
  verified current capability or a guarantee for arbitrary languages/media.
- Subtitles: no unintended cue loss, duplication or changed text/style. Cues
  intersecting cuts have explicit split/remove/review outcomes. No single global
  timing tolerance substitutes for alignment review at the affected passage.
- Original/prior artifacts: exact byte hashes unchanged; earlier Drive identities
  and content preserved. New edits create distinct deliverables.

## Evidence required for a release decision

### Application integration checkpoint — 2026-10-23

The run chat, durable planning turns and SQL supervisor, immutable media worker,
audio comparison previews, subtitle tracks and strict legacy caption import,
and isolated runtime broker are implemented for review. The private beta stays
disabled until the remaining qualification and deployment checks pass.

- Production Python 3.12.15 / Pydantic 2.14.1 / pytest 9.1.1 / FFmpeg 7.1.5:
  **638 tests passed**, including the assistant suites and existing model,
  state-machine and cross-site-request regressions.
- Disposable PostgreSQL 16: **26 tests passed**, including planning claim and
  cancellation races and the complete c4/d5/e6/f7 migration upgrade/downgrade.
- Chromium and synthetic legacy media: **7 tests passed**, covering branching,
  reopen, proposal review, progress, comparisons, retry response loss, exact
  frame markers, stale legacy duration and caption provenance rejection.
- Isolated copies of both approved-project run results passed immutable import
  with actual video duration 29.7 seconds. The existing caption proof exceeded
  its 90-second bound under two CPUs; that caption is **not yet qualified**.
- Separate minimal Claude runner images built; empty-auth network tests deny
  direct public/private/metadata routes and forbidden CONNECT destinations.
  Native authenticated broker qualification remains pending.

The subsequent integration adds offline word evidence, bounded automatic donor
imports, durable broker cleanup, scoped cached-history redaction, and paginated
history. The seven Sol-6.1 findings on b6e7c0d..bcfeaa7 have dedicated fixes:
stale recovery cancellation, missing ACK reconciliation, oversized history,
populated SQLite migration failure, lost-submit base changes, integer FPS
markers, and the missing discovery clock import. The follow-up review remains
required before rollout.

Additional qualification evidence:

- The approved caption copy reproduced exactly in 143.31 seconds under two CPUs;
  its import bound is now 240 seconds. A mismatch still refuses editable import.
- Native Claude through the isolated broker/gateway returned a strict valid
  synthetic reply from claude-sonnet-5-5 in 4.081 seconds. Client UID checks,
  read-only socket mounts, restart interruption without replay, and private
  credential/process canaries passed. This was one included-subscription call.
- Offline ASR passed actual Unix RPC on synthetic speech and three authorized
  media copies within 2 GiB/2 CPU limits. Coverage/confidence and measured resource
  evidence are in `deploy/assistant/ASR-README.md`; no transcript is a verified
  pronunciation or speaker assessment.
- Actual audio replacement preserved all 891 original video packets and produced
  a complete 59.4-second before/after proxy preview. Captioned audio edits also
  preserve full picture packets and carry updated audio into later subtitle
  edits. Bounded 0.8–1.25 tempo fitting rejects audible residual clipping/padding;
  real FFmpeg 7 checks validated both quiet and rejected audible endings.
- Post-review browser/legacy/populated-SQLite checks: **10 passed**. Complete
  PostgreSQL migration/race suite: **26 passed**; separate enriched-history and
  turn races: **4 passed**. Targeted integration checks passed before the final
  production-lock rerun. Final review/deploy evidence is still pending.

Synthetic success and unit tests do not establish listening quality, annotated
word-boundary accuracy, or a completed production rollout. The UI requires review
of candidate audio, exposes uncertain speech coverage, and offers player selection
when visual localization cannot be inferred from supplied context.

Record tested revision and environment; fixture hashes; plan/base/donor IDs;
expected versus actual stream properties; cue diffs; before/after previews;
listening and visual judgments with reviewer; test commands/results; independent
Sol-6.1 findings and resolutions; resource/latency measurements; known provider
charges; migration, backup and rollback evidence. Record uncertainty separately
from verified success. An implementation unit test or successful FFmpeg exit is
not an end-to-end launch verdict.

The first release may reject unsupported media or unavailable assets explicitly.
It may not silently drop one of the four required workflows and still claim the
master feature is launched. A narrower pilot must identify its exact supported
scope and remain distinct from completion of AICP-261.
