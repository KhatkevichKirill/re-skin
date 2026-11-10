# Preparing an editable copy without losing an audio tail

AICP-309. A delivered Run can contain AAC audio beyond the declared picture
end. Run `da70b937` also exposes a terminal timestamp defect: 892 packets at
30 fps occupy slots 0–890 and 892, with slot 891 missing. The declared stream end
is 29.733333 s, equal to the final frame's PTS; that packet itself still has a
normal 1/30 s duration. Audio continues to 29.820998 s. Truncating audio could remove
speech, while treating this picture timeline as ordinary CFR is incorrect.

## Original and working version

Keep the normal strict editing profile. A separate, bounded import inspection
may classify either of two otherwise supported H.264/yuv420p MP4 profiles with
one zero-start AAC track and a positive audio overhang of at most 100 ms:

- An ordinary proved zero-start CFR picture timeline.
- One exact terminal grid defect: for N packets at native rate F, their unique
  rational presentation slots are 0,1,…,N−2,N, and the declared end is exactly N/F.
  There is one missing slot immediately before the final source frame.

Other holes, duplicates, off-grid/VFR timing, inconsistent counts/end, unsupported
properties, malformed timestamps, timeout and confinement failure do not qualify.

Import the exact original bytes as Version 1 and expose playable history with
an explicit preparation requirement. It is readable media, not executable
editing input. Block planning, rendering and publication of the unprepared
version, including attempts to use it through donor/audio-only paths.

The operator can choose **Prepare editable copy**. Before that action, disclose
gap filling and final-frame holds separately, picture re-encoding, and complete
audio preservation. All original source frames retain their presentation times.
The fixed operation copies AAC unchanged and produces an ordinary strict CFR
timeline. For the reported Run the mapping is:

| Output slots | Picture |
|---|---|
|0–890|Corresponding original source frame|
|891 (29.700000s)|Repeat source frame 890|
|892 (29.733333s)|Original final source frame|
|893–894|Repeat the final source frame|

There are 895 output frames and 29.833333 s of picture: one gap fill and two final
holds. A normal 892-frame CFR source with the same audio needs three final holds
instead. Do not describe both profiles as three appended final-frame holds.
The original remains Version 1; the prepared result is a separate owned child.
The main Run result and Drive file do not change.

For the terminal-defect profile, default MOV edit-list decoding can discard the
encoded final frame. Use a separately qualified video-only decode path, proving
the same packet count and a uniform rational PTS shift when ignoring edit lists;
normalize that constant video offset and verify decoded frame/PTS agreement.
Read the AAC track through its original edit-list handling in a separate input.
Never ignore audio edit lists or silently normalize arbitrary video timelines.
Guarded padding/FPS conversion and video-only trimming must satisfy the exact
mapping above; a matching output count alone does not establish preservation.

Preparation does not establish clean-picture or editable-subtitle provenance.
Burned text stays part of the picture. Do not manufacture clean/caption roles.
At low frame rates, rounding up may still fail the strict audio/picture timing
profile; refuse safely rather than widening that profile.

## Execution boundary and recovery

Preparation is an authenticated operator action, not a Claude tool. The backend
derives all media parameters from verified immutable input. Requests carry only
the Run/version binding, expected selection epoch and retry identity; arbitrary
paths, filters, commands or provider options are unavailable.

Use bounded confined local processing, with no model call or external cost.
The HTTP operation has a four-minute processing budget. Admit one preparation
at a time in the current single API process; a busy or uncertain response uses
503 and retains the exact retry request. Multiple API processes would require
shared admission before increasing the deployment's concurrency. Preparation
has no separate Cancel control; closing chat does not cancel server processing.
Normal assistant edit jobs retain their existing cancellation controls.

Idempotency and finalization must fence ownership, selected version ID and
epoch, concurrent publication, source integrity and duplicate preparation.
Replays return the completed child without selecting it again. Reopening chat
or returning to Version 1 must preserve the operator's selection and draft.
Timeout or failed validation cannot publish a successful child or replace the
main result. An uncertain HTTP outcome must be retried with the same request.

Validate prepared output with the unchanged strict probe and verify its exact
frame count, geometry, frame rate and complete audio packet timing/content.
Native tests also compare decoded audio, including an audible sentinel beyond
the original picture end. A successful encoder exit alone is insufficient.

## Publication and retained originals

**Use as Run result** remains a separate explicit operator action on a strict,
successful selected version. Publication may archive an old main that has the
bounded audio-tail profile; this exception applies only to the retained before
snapshot, with exact owned-original integrity binding. It does not admit an
unsupported new main or weaken publication revision and recovery checks.

The access switch and closed tools are described in
[assistant-access-scope.md](assistant-access-scope.md). Publication history and
Drive recovery are described in [assistant-run-result.md](assistant-run-result.md).

## Deployment qualification

Keep tracked public source readable by deployment UID 10001 (normally 0644).
A private backup/log umask must not accidentally make a Git checkout's Python
files root-only. Build and qualify the actual application image before activation;
compare used source and native hashes across API, supervisor and offline ASR.
Private runtime configuration and data keep their existing restricted modes.
