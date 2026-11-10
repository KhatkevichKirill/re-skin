"""The VOID fill's caption box (pipeline_segment_fill.caption_box / measure_box).

Run d46bb1b8 seg 6: the detector's per-frame box is ONE word ("KNOWING" of
"KNOWING NOBODY."), so the measured box missed "NOBODY." and the fill left "Y."
visible. The box is now the union of the caption LINES, mirrored about the
frame's centre when the captions are centred.
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from app import pipeline_segment_fill as psf
from app import qa_tools as q
from app import subtitle_loop as sl
from app.autopilot.flows import subtitle_removal_plan as P
from app.media import FFMPEG
from app.qa_tools import CaptionFrame

sys.path.insert(0, os.path.dirname(__file__))
from qa_caption_reference import legacy_caption_score  # noqa: E402


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")

W, H = 540, 960


def _words_frame(words, *, y_frac=0.68, scale=1.4):
    """White outlined *words* = [(x_frac, text), ...] on mid grey."""
    img = np.full((H, W, 3), 128, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for x_frac, text in words:
        cv2.putText(mask, text, (int(W * x_frac), int(H * y_frac)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 3, cv2.LINE_8)
    outline = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    img[outline > 0] = 0
    img[mask > 0] = 255
    return img


def _encode(frames, path, fps=10):
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
         str(path)],
        input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return str(path)


def _f(t, line, box=None):
    return CaptionFrame(t=t, score=200, box=tuple(box or line), line=tuple(line))


# ---------------------------------------------------------------------------
# qa_tools: the caption line around the best cluster
# ---------------------------------------------------------------------------

def test_line_spans_the_words_of_the_best_clusters_row():
    # A big word, a smaller one a word gap to its right: the box is the big
    # one (the detector's best cluster), the line both.
    img = _words_frame([(0.22, "WWWW"), (0.48, "ii.")])
    score, box, line = q.caption_score_line(img)
    assert (score, box) == q.caption_score(img)
    assert score > q.CAPTION_HIT_THRESHOLD
    assert box[2] < 0.48 < 0.52 < line[2]
    # The line holds the box (its rows: the union of the words' rows).
    assert line[0] == pytest.approx(box[0])
    assert line[1] <= box[1] < box[3] <= line[3] and line[3] - line[1] < 0.05


def test_line_takes_a_narrow_one_letter_word():
    # "I" alone is not text-shaped (narrower than min_aspect): never the
    # score, but part of the line it ends.
    img = _words_frame([(0.30, "WWWW"), (0.56, "I")])
    _s, box, line = q.caption_score_line(img)
    assert q.caption_score(img) == legacy_caption_score(img)
    assert box[2] < 0.56 < 0.58 < line[2]


def test_line_does_not_jump_a_wide_gap_or_another_row():
    far = _words_frame([(0.20, "WWW"), (0.50, "ii.")])         # gap > a line height
    assert len(q._text_clusters(q._edge_mask(far, q.caption_config(None)),
                                q.caption_config(None))) == 2
    _s, box, line = q.caption_score_line(far)
    assert line == box
    rows = _words_frame([(0.22, "WWWW")], y_frac=0.58)
    lower = _words_frame([(0.48, "ii.")], y_frac=0.68)
    rows[lower != 128] = lower[lower != 128]                    # next to it, a row below
    _s, box, line = q.caption_score_line(rows)
    assert box[3] < 0.62 and line == box


def test_caption_score_unchanged_against_the_reference():
    for img in (_words_frame([(0.22, "WWWW"), (0.48, "ii.")]), _words_frame([]),
                _words_frame([(0.30, "AB")], y_frac=0.25)):
        assert q.caption_score(img) == legacy_caption_score(img)


# ---------------------------------------------------------------------------
# centred / mirrored box
# ---------------------------------------------------------------------------

def test_centred_captions_get_a_box_mirrored_about_the_centre():
    # Word boxes left of centre, lines centred: d46bb1b8 seg 6 in miniature.
    frames = [_f(i / 10, (0.30, 0.67, 0.70, 0.70), box=(0.30, 0.67, 0.48, 0.70))
              for i in range(5)]
    frames += [_f(0.5 + i / 10, (0.20, 0.67, 0.79, 0.70), box=(0.20, 0.67, 0.51, 0.70))
               for i in range(5)]
    assert psf.captions_centred(frames)
    # line union 0.20-0.79 (+0.03) mirrored: 0.17-0.83
    assert psf.caption_box(frames) == [0.17, 0.65, 0.83, 0.72]


def test_left_aligned_captions_are_not_widened():
    frames = [_f(i / 10, (0.10, 0.67, 0.10 + w, 0.70))
              for i, w in enumerate((0.2, 0.35, 0.5, 0.3, 0.6, 0.25, 0.4, 0.45))]
    assert not psf.captions_centred(frames)
    assert psf.caption_box(frames) == [0.07, 0.65, 0.73, 0.72]


def test_side_caption_and_too_little_evidence_are_not_centred():
    side = [_f(i / 10, (0.55, 0.67, 0.85, 0.70)) for i in range(10)]
    assert not psf.captions_centred(side)
    assert psf.caption_box(side) == [0.52, 0.65, 0.88, 0.72]
    few = [_f(i / 10, (0.40, 0.67, 0.58, 0.70)) for i in range(psf.CENTRED_MIN_LINES - 1)]
    assert not psf.captions_centred(few)
    assert psf.caption_box(few) == [0.37, 0.65, 0.61, 0.72]
    assert psf.caption_box([CaptionFrame(t=0.0, score=0)]) is None


def test_mostly_centred_with_an_off_centre_minority_still_mirrors():
    frames = [_f(i / 10, (0.30, 0.67, 0.70, 0.70)) for i in range(8)]
    frames += [_f(0.8 + i / 10, (0.60, 0.67, 0.85, 0.70)) for i in range(2)]  # 20 %
    assert psf.captions_centred(frames)
    assert psf.caption_box(frames) == [0.12, 0.65, 0.88, 0.72]
    frames += [_f(1.0, (0.60, 0.67, 0.85, 0.70))]                             # > 20 %
    assert not psf.captions_centred(frames)


def test_frames_without_a_line_fall_back_to_their_box():
    frames = [CaptionFrame(t=i / 10, score=200, box=(0.4, 0.6, 0.6, 0.64)) for i in range(6)]
    assert psf.captions_centred(frames)
    assert psf.caption_box(frames) == [0.37, 0.58, 0.63, 0.66]


def test_mirror_never_narrows_and_clamps():
    assert psf.mirror_x([0.05, 0.1, 0.5, 0.2], True) == [0.05, 0.1, 0.95, 0.2]
    assert psf.mirror_x([0.05, 0.1, 0.5, 0.2], False) == [0.05, 0.1, 0.5, 0.2]
    frames = [_f(i / 10, (0.01, 0.67, 0.99, 0.70)) for i in range(6)]
    assert psf.caption_box(frames) == [0.0, 0.65, 1.0, 0.72]


@needs_ffmpeg
def test_measure_box_covers_the_last_word_of_a_centred_line(tmp_path):
    # Centred two-word line; the right word is narrower, so the detector's
    # per-frame box never reaches it -- the old union stopped short of it.
    frames = []
    for i in range(20):
        frames.append(_words_frame([(0.34, "WWWW"), (0.60, "ii.")]) if i < 10
                      else _words_frame([(0.4075, "WWW")]))
    clip = _encode(frames, tmp_path / "line.mp4")
    scan = q.scan_captions(clip, fps=10)
    hits = [f for f in scan.scores if f.box]
    assert len(hits) == 20
    old_x1 = max(f.box[2] for f in hits) + psf.BOX_PAD_X
    right_word_end = max(f.line[2] for f in hits)
    assert old_x1 < right_word_end                  # the bug: box misses the word
    box = psf.measure_box(clip, 10.0, None)
    assert box[2] >= right_word_end + psf.BOX_PAD_X - 1e-3
    assert box[0] == pytest.approx(1.0 - box[2], abs=2e-4)


# ---------------------------------------------------------------------------
# widened / grown boxes follow the same rule
# ---------------------------------------------------------------------------

def test_widen_and_grown_boxes_mirror_only_when_centred():
    seg = sl.Seg(index=0, start=0.0, end=4.0, action="swap")
    fill_box = [0.38, 0.59, 0.62, 0.65]
    left = [{"start_sec": 1.0, "end_sec": 1.3, "box": [0.30, 0.60, 0.40, 0.64]}]
    assert sl._widen_box(fill_box, left, seg) == [0.27, 0.57, 0.65, 0.67]
    assert sl._widen_box(fill_box, left, seg, centred=True) == [0.27, 0.57, 0.73, 0.67]
    assert P.grown_box(fill_box, left, 0.0, 4.0) == [0.23, 0.54, 0.69, 0.70]
    assert P.grown_box(fill_box, left, 0.0, 4.0, centred=True) == [0.23, 0.54, 0.77, 0.70]


def test_qa_row_reports_centred_source_captions():
    frames = [_f(i / 10, (0.3, 0.67, 0.7, 0.70), box=(0.3, 0.67, 0.5, 0.70)) for i in range(10)]
    assert sl.mask_box_from(frames) == [0.27, 0.65, 0.73, 0.72]
    assert sl._centred(frames)
    assert sl.mask_box_from([]) is None
