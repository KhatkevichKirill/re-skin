"""
Tests for app.captions — the pure ASS-subtitle builder for burned captions
(design contract: docs/burned-captions.md §3.1).

Pure module, no fixtures needed beyond pytest's own `caplog`: no DB, no
ffmpeg, no network. Several tests pin exact ASS-format details (colour byte
order, the Bold sentinel, timestamp padding) that are called out explicitly
in the design doc as easy to get backwards and invisible-until-rendered if
so — see the module docstring's "ASS gotchas" section.
"""

from __future__ import annotations

import logging

import pytest

from app.captions import (
    ALIGNMENT_BOTTOM_CENTER,
    BORDER_STYLE_BOX,
    BOX_ALPHA,
    CAPTION_STYLE_VERSION,
    CJK_FONT,
    DEFAULT_FONT,
    FONT_SIZE_RATIO,
    MARGIN_H_RATIO,
    MARGIN_V_RATIO,
    MAX_ROWS,
    MIN_LINE_DURATION_SEC,
    NOMINAL_HEIGHT,
    NOMINAL_WIDTH,
    _ass_colour,
    build_ass,
    escape_text,
    font_for_language,
    font_size_px,
    format_timestamp,
    layout_text,
    measure_text,
    prepare_lines,
    split_line,
    usable_width_px,
    wrap_rows,
    wrap_text,
)

LOGGER_NAME = "app.captions"


def _row_widths(rows, *, font=DEFAULT_FONT, height=NOMINAL_HEIGHT):
    """Measure *rows* the same way the wrapper did.

    Asserting against :func:`measure_text` rather than against a hardcoded
    pixel number keeps these tests true on a box with Pillow and the fonts
    (where widths are real metrics) and on one without either (where they
    are the em estimate) — the property under test is "the row fits the
    width this module believes it has", which holds under both.
    """
    return [measure_text(row, font=font, size_px=font_size_px(height)) for row in rows]


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "0:00:00.00"

    def test_sub_second_centiseconds(self):
        assert format_timestamp(1.23) == "0:00:01.23"

    def test_minutes_and_seconds_are_zero_padded(self):
        assert format_timestamp(65.5) == "0:01:05.50"

    def test_over_one_hour_uses_a_single_hour_digit(self):
        # One hour, two minutes, three seconds, forty centiseconds.
        assert format_timestamp(3723.4) == "1:02:03.40"

    def test_ten_plus_hours_does_not_truncate_the_hour_field(self):
        # 10h exactly: hours are NOT zero-padded to two digits, but a value
        # >= 10 must still print in full, not wrap or truncate to one digit.
        assert format_timestamp(10 * 3600) == "10:00:00.00"

    def test_rounding_carries_across_the_minute_boundary(self):
        # 59.996s rounds to 60.00s at centisecond resolution, which must
        # roll into the next minute, not print an impossible ":60.00".
        assert format_timestamp(59.996) == "0:01:00.00"

    def test_rounding_carries_across_the_hour_boundary(self):
        # 3599.996s -> 3600.00s at centisecond resolution -> must roll the
        # hour over cleanly (see the function's rounding-on-total-centis note).
        assert format_timestamp(3599.996) == "1:00:00.00"

    def test_negative_clamps_to_zero(self):
        assert format_timestamp(-5.0) == "0:00:00.00"

    def test_never_raises_on_garbage(self):
        assert format_timestamp(float("nan")) == "0:00:00.00"
        assert format_timestamp(float("inf")) == "0:00:00.00"
        assert format_timestamp(None) == "0:00:00.00"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# escape_text
# ---------------------------------------------------------------------------


class TestEscapeText:
    def test_open_brace_is_neutralised(self):
        result = escape_text("hi {\\an8}there")
        assert "{" not in result

    def test_close_brace_is_neutralised(self):
        result = escape_text("a}b")
        assert "}" not in result

    def test_backslash_is_neutralised(self):
        result = escape_text("C:\\Users\\ops")
        # The only backslashes escape_text itself ever introduces form the
        # "\N" row-break code; this input has no real newline, so the
        # output must contain zero ASCII backslash characters.
        assert "\\" not in result

    def test_real_newline_becomes_ass_row_break(self):
        assert escape_text("line one\nline two") == "line one\\Nline two"

    def test_crlf_and_cr_also_become_the_row_break(self):
        assert escape_text("a\r\nb") == "a\\Nb"
        assert escape_text("a\rb") == "a\\Nb"

    def test_backslash_neutralised_before_newline_injection(self):
        # A literal "\n" two-character sequence in the source text (as
        # opposed to an actual newline character) must not be able to masquerade
        # as this function's own row-break code: the backslash gets
        # substituted, so the trailing "n" stays a plain letter.
        result = escape_text("a\\nb")
        assert result != "a\\Nb"
        assert "\\" not in result

    def test_plain_text_is_untouched(self):
        assert escape_text("hello world") == "hello world"

    def test_never_raises_on_non_string_ish_input(self):
        assert escape_text(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _ass_colour — the &HAABBGGRR byte-order gotcha
# ---------------------------------------------------------------------------


class TestAssColour:
    def test_channel_order_is_alpha_blue_green_red(self):
        # Distinct, non-symmetric channel values so a swapped implementation
        # (e.g. accidentally emitting AARRGGBB) cannot pass by coincidence.
        assert _ass_colour(red=0x11, green=0x22, blue=0x33, alpha=0x44) == "&H44332211"

    def test_default_alpha_is_zero_i_e_opaque(self):
        assert _ass_colour(red=0xAA, green=0xBB, blue=0xCC) == "&H00CCBBAA"

    def test_house_white_is_opaque(self):
        assert _ass_colour(0xFF, 0xFF, 0xFF) == "&H00FFFFFF"

    def test_house_black_is_opaque(self):
        assert _ass_colour(0x00, 0x00, 0x00) == "&H00000000"


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------


class TestMeasureText:
    def test_empty_text_is_zero_wide(self):
        assert measure_text("", font=DEFAULT_FONT, size_px=69) == 0.0

    def test_width_grows_with_the_text(self):
        short = measure_text("hello", font=DEFAULT_FONT, size_px=69)
        longer = measure_text("hello there", font=DEFAULT_FONT, size_px=69)
        assert 0 < short < longer

    def test_width_grows_with_the_font_size(self):
        small = measure_text("hello there", font=DEFAULT_FONT, size_px=30)
        large = measure_text("hello there", font=DEFAULT_FONT, size_px=90)
        assert large > small

    def test_falls_back_to_an_em_estimate_without_a_usable_font(self, monkeypatch):
        # The estimate path is what runs on a box with no Pillow or no font
        # files, and it must stay proportional in both text length and size
        # or the wrapper silently starts wrapping at the wrong place.
        monkeypatch.setattr("app.captions._pil_font", lambda family, size_px: None)
        one = measure_text("x", font=DEFAULT_FONT, size_px=100)
        ten = measure_text("x" * 10, font=DEFAULT_FONT, size_px=100)
        assert ten == pytest.approx(one * 10)
        assert measure_text("x", font=DEFAULT_FONT, size_px=200) == pytest.approx(one * 2)

    def test_cjk_is_estimated_wider_per_character_than_latin(self, monkeypatch):
        monkeypatch.setattr("app.captions._pil_font", lambda family, size_px: None)
        assert (
            measure_text("あいうえお", font=CJK_FONT, size_px=69)
            > measure_text("abcde", font=DEFAULT_FONT, size_px=69)
        )

    def test_unknown_font_family_has_no_file_and_does_not_raise(self):
        from app.captions import _font_file

        assert _font_file("No Such Font") is None
        assert measure_text("hello", font="No Such Font", size_px=69) > 0


class TestWrapRows:
    def test_short_text_is_not_wrapped(self):
        assert wrap_rows("hello there", cjk=False) == ["hello there"]

    def test_every_row_fits_the_usable_width(self):
        # The v1 bug, as a test: at the v1 wrap width these sentences
        # produced rows ~40% wider than the frame they were burned into.
        text = (
            "and then I put it on like two and a half, 3x speed and it'll "
            "read it to me so fast, anywhere I want"
        )
        rows = wrap_rows(text, cjk=False)
        limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
        assert len(rows) > 1
        for width in _row_widths(rows):
            assert width <= limit

    def test_wrapping_loses_no_words(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        assert " ".join(wrap_rows(text, cjk=False)).split() == text.split()

    def test_rows_are_balanced_not_greedily_packed(self):
        # Greedy fill leaves "text" alone on a row of its own; balancing is
        # the difference between a caption that looks set and one that looks
        # like it ran out of room.
        rows = wrap_rows("So I just take pictures of text", cjk=False)
        assert len(rows) == 2
        widths = _row_widths(rows)
        assert min(widths) > max(widths) * 0.5

    def test_balancing_never_adds_a_row(self):
        for text in (
            "and then it reads it to you at any speed.",
            "And I can even switch the voices. So watch, this is Gwyneth Paltrow.",
            "What's your favorite voice?",
        ):
            limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
            balanced = wrap_rows(text, cjk=False)
            # A row count below the greedy minimum is impossible; above it
            # would mean bisection overshot and made the caption taller.
            greedy_rows = 0
            current = ""
            for word in text.split():
                candidate = f"{current} {word}" if current else word
                if measure_text(candidate, font=DEFAULT_FONT,
                                size_px=font_size_px(NOMINAL_HEIGHT)) <= limit:
                    current = candidate
                else:
                    greedy_rows += 1
                    current = word
            greedy_rows += 1 if current else 0
            assert len(balanced) == greedy_rows

    def test_cjk_text_is_chunked_by_character_not_by_word(self):
        # Japanese has no spaces to wrap on; without character chunking this
        # would come back as one unbroken row running off both edges.
        text = "あ" * 40
        rows = wrap_rows(text, cjk=True)
        assert len(rows) > 1
        assert "".join(rows) == text
        limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
        for width in _row_widths(rows, font=CJK_FONT):
            assert width <= limit

    def test_cjk_auto_detected_from_text_when_cjk_not_given(self):
        text = "こんにちは" * 8
        assert wrap_rows(text) == wrap_rows(text, cjk=True, font=CJK_FONT)

    def test_latin_auto_detected_when_cjk_not_given(self):
        text = "plain english sentence with no special characters at all here"
        assert wrap_rows(text) == wrap_rows(text, cjk=False)

    def test_single_overlong_word_is_broken_not_left_overflowing(self):
        long_word = "x" * 200
        rows = wrap_rows(long_word, cjk=False)
        assert len(rows) > 1
        assert "".join(rows) == long_word
        limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
        for width in _row_widths(rows):
            assert width <= limit

    def test_overlong_word_amid_other_words_does_not_swallow_them(self):
        text = f"short {'y' * 120} tail"
        rows = wrap_rows(text, cjk=False)
        assert rows[0] == "short"
        assert rows[-1].endswith("tail")
        assert "y" * 120 in "".join(rows).replace(" ", "")

    def test_no_row_cap_here_long_text_keeps_every_row(self):
        # The v1 behaviour this replaces: rows beyond MAX_ROWS were dropped.
        # Capping is split_line's job now, and it splits rather than drops.
        rows = wrap_rows(" ".join(["word"] * 40), cjk=False)
        assert len(rows) > MAX_ROWS
        assert " ".join(rows).split() == ["word"] * 40

    def test_a_narrower_frame_needs_more_rows(self):
        text = "and then it reads it to you at any speed."
        assert len(wrap_rows(text, cjk=False, width=540, height=960)) >= len(
            wrap_rows(text, cjk=False, width=1080, height=1920)
        )

    def test_empty_text_returns_no_rows(self):
        assert wrap_rows("") == []
        assert wrap_rows(None) == []  # type: ignore[arg-type]
        assert wrap_text("") == ""


class TestSplitLine:
    def test_rows_are_grouped_in_max_rows_sized_captions(self):
        rows = ["a", "b", "c", "d", "e"]
        assert split_line(rows, max_rows=2) == [["a", "b"], ["c", "d"], ["e"]]

    def test_short_line_is_a_single_caption(self):
        assert split_line(["a", "b"], max_rows=2) == [["a", "b"]]

    def test_no_rows_is_no_captions(self):
        assert split_line([], max_rows=2) == []

    def test_max_rows_is_floored_at_one(self):
        assert split_line(["a", "b"], max_rows=0) == [["a"], ["b"]]


class TestLayoutText:
    LONG = (
        "and then I put it on like two and a half, 3x speed and it'll read "
        "it to me so fast, anywhere I want to"
    )

    def test_long_text_becomes_several_captions_instead_of_losing_rows(self):
        captions = layout_text(self.LONG, cjk=False)
        assert len(captions) > 1
        for caption in captions:
            assert 0 < len(caption) <= MAX_ROWS

    def test_nothing_is_dropped_across_the_split(self):
        captions = layout_text(self.LONG, cjk=False)
        shown = " ".join(" ".join(caption) for caption in captions)
        assert shown.split() == self.LONG.split()

    def test_every_row_of_every_caption_fits(self):
        limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
        for caption in layout_text(self.LONG, cjk=False):
            for width in _row_widths(caption):
                assert width <= limit

    def test_short_text_is_one_caption(self):
        assert layout_text("hello there", cjk=False) == [["hello there"]]

    def test_empty_text_is_no_captions(self):
        assert layout_text("", cjk=False) == []


# ---------------------------------------------------------------------------
# font_for_language
# ---------------------------------------------------------------------------


class TestFontForLanguage:
    def test_japanese_gets_the_cjk_font(self):
        assert font_for_language("ja") == CJK_FONT

    def test_english_gets_the_default_font(self):
        assert font_for_language("en") == DEFAULT_FONT

    def test_case_and_region_variants_are_normalised(self):
        assert font_for_language("JA") == CJK_FONT
        assert font_for_language("ja-JP") == CJK_FONT
        assert font_for_language("zh-Hans") == CJK_FONT
        assert font_for_language("ko_KR") == CJK_FONT

    def test_iso_639_2_variants_are_accepted(self):
        assert font_for_language("jpn") == CJK_FONT
        assert font_for_language("kor") == CJK_FONT

    def test_unknown_code_falls_back_to_sample_text_sniff(self):
        assert font_for_language("xx", sample_text="こんにちは") == CJK_FONT
        assert font_for_language("xx", sample_text="hello") == DEFAULT_FONT

    def test_no_language_sniffs_sample_text(self):
        assert font_for_language(None, sample_text="日本語のテキスト") == CJK_FONT
        assert font_for_language(None, sample_text="plain english") == DEFAULT_FONT

    def test_no_language_and_no_sample_text_defaults(self):
        assert font_for_language(None) == DEFAULT_FONT
        assert font_for_language("") == DEFAULT_FONT

    def test_escaped_fullwidth_substitutes_are_not_mistaken_for_cjk(self):
        # escape_text's lookalikes for {, } and \ live in the Fullwidth
        # Forms block; the CJK sniff must not fire on them, or a caption
        # that only contains punctuation-turned-fullwidth would wrongly
        # switch fonts on a second pass.
        assert font_for_language(None, sample_text="｛｝＼") == DEFAULT_FONT


# ---------------------------------------------------------------------------
# prepare_lines
# ---------------------------------------------------------------------------


class TestPrepareLines:
    def test_empty_input_returns_empty_list(self):
        assert prepare_lines([]) == []
        assert prepare_lines(None) == []  # type: ignore[arg-type]

    def test_line_shape_is_exactly_start_end_text(self):
        out = prepare_lines([{"start": 0.0, "end": 2.0, "text": "hi",
                               "id": 1, "speaker": "narrator", "on_screen": True}])
        assert len(out) == 1
        assert set(out[0]) == {"start", "end", "text"}
        assert out[0]["text"] == "hi"

    def test_blank_text_is_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines([
                {"start": 0.0, "end": 1.0, "text": "   "},
                {"start": 2.0, "end": 3.0, "text": "real line"},
            ])
        assert [ln["text"] for ln in out] == ["real line"]
        assert any("blank" in msg for msg in _warnings(caplog))

    def test_missing_text_key_is_dropped(self):
        out = prepare_lines([{"start": 0.0, "end": 1.0}])
        assert out == []

    def test_non_dict_entries_are_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines(["not a dict", None, 42,
                                  {"start": 0.0, "end": 1.0, "text": "ok"}])
        assert [ln["text"] for ln in out] == ["ok"]

    def test_unparseable_start_is_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines([
                {"start": "not-a-number", "end": 1.0, "text": "bad"},
                {"start": 2.0, "end": 3.0, "text": "good"},
            ])
        assert [ln["text"] for ln in out] == ["good"]
        assert any("unparseable" in msg for msg in _warnings(caplog))

    def test_non_finite_start_is_dropped(self):
        out = prepare_lines([
            {"start": float("nan"), "end": 1.0, "text": "nan-start"},
            {"start": float("inf"), "end": 2.0, "text": "inf-start"},
        ])
        assert out == []

    def test_missing_end_defaults_to_start_then_gets_min_duration(self):
        out = prepare_lines([{"start": 5.0, "text": "no end given"}])
        assert len(out) == 1
        assert out[0]["start"] == 5.0
        assert out[0]["end"] - out[0]["start"] >= MIN_LINE_DURATION_SEC

    def test_out_of_order_input_is_sorted_by_start(self):
        out = prepare_lines([
            {"start": 10.0, "end": 11.0, "text": "second"},
            {"start": 0.0, "end": 1.0, "text": "first"},
        ])
        assert [ln["text"] for ln in out] == ["first", "second"]

    def test_short_line_is_extended_to_the_minimum_duration(self):
        out = prepare_lines([{"start": 1.0, "end": 1.01, "text": "flash"}])
        assert len(out) == 1
        assert out[0]["end"] - out[0]["start"] == pytest.approx(MIN_LINE_DURATION_SEC)

    def test_overlap_truncates_the_earlier_line_not_the_later_one(self, caplog):
        # Second line starts well before the first ends; the first's end
        # must be pulled back to the second's start, and the second line's
        # own start/end must be completely untouched.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines([
                {"start": 0.0, "end": 5.0, "text": "early, said first"},
                {"start": 3.0, "end": 6.0, "text": "said next"},
            ])
        assert len(out) == 2
        first, second = out
        assert first["text"] == "early, said first"
        assert first["start"] == 0.0
        assert first["end"] == 3.0  # truncated to the second line's start
        assert second["start"] == 3.0  # never shifted
        assert second["end"] == 6.0
        assert any("truncat" in msg for msg in _warnings(caplog))

    def test_overlap_that_would_leave_earlier_line_too_short_merges_it(self, caplog):
        # First line starts at 0.0; the second starts only 0.1s later, which
        # is well under MIN_LINE_DURATION_SEC — truncating the first line to
        # fit would leave it below the floor. It used to be DROPPED (and its
        # words with it, AICP-247); now both share one caption from the
        # first line's start to the later end.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines([
                {"start": 0.0, "end": 5.0, "text": "said first"},
                {"start": 0.1, "end": 5.0, "text": "said next"},
            ])
        assert out == [{"start": 0.0, "end": 5.0, "text": "said first said next"}]
        assert any("merging" in msg for msg in _warnings(caplog))

    def test_cascading_overlaps_resolve_correctly(self):
        # Three lines, each overlapping the next, all comfortably longer
        # than the minimum duration once truncated — every one should
        # survive, truncated pairwise, non-overlapping and in order.
        out = prepare_lines([
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 2.0, "end": 10.0, "text": "b"},
            {"start": 4.0, "end": 10.0, "text": "c"},
        ])
        assert [ln["text"] for ln in out] == ["a", "b", "c"]
        for earlier, later in zip(out, out[1:]):
            assert earlier["end"] <= later["start"]

    def test_lines_are_clamped_into_0_duration_sec(self):
        out = prepare_lines(
            [{"start": -5.0, "end": 100.0, "text": "spans the whole clip"}],
            duration_sec=10.0,
        )
        assert len(out) == 1
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 10.0

    def test_line_entirely_past_duration_sec_is_dropped(self):
        out = prepare_lines(
            [{"start": 20.0, "end": 25.0, "text": "after the video ends"}],
            duration_sec=10.0,
        )
        assert out == []

    def test_line_partially_past_duration_sec_with_no_room_for_minimum_is_dropped(self):
        # Starts just inside the clip but clamping the end leaves less than
        # the minimum duration, with duration_sec as a hard ceiling — must
        # be dropped, not kept as a too-short flash right at the cut.
        out = prepare_lines(
            [{"start": 9.9, "end": 20.0, "text": "right at the edge"}],
            duration_sec=10.0,
        )
        assert out == []

    def test_negative_or_zero_duration_sec_is_treated_as_unbounded(self):
        out = prepare_lines(
            [{"start": 0.0, "end": 5.0, "text": "fine"}], duration_sec=0.0
        )
        assert len(out) == 1
        assert out[0]["end"] == 5.0

    def test_never_raises_on_thoroughly_malformed_input(self):
        class Hostile:
            def __str__(self):
                raise RuntimeError("boom")

        garbage = [
            {"start": [1, 2], "end": "??", "text": Hostile()},
            {"text": "ok", "start": 0.0, "end": 1.0},
            "just a string",
            123,
            None,
            {},
        ]
        out = prepare_lines(garbage)  # must not raise
        assert [ln["text"] for ln in out] == ["ok"]

    def test_non_iterable_input_degrades_to_empty_list(self):
        assert prepare_lines(42) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AICP-247: quick consecutive lines are merged, never dropped
# ---------------------------------------------------------------------------


def _dialogue_rows(doc: str) -> list:
    """``(start, end, text)`` of every Dialogue row, ``\\N`` kept as-is."""
    rows = []
    for raw in doc.splitlines():
        if raw.startswith("Dialogue:"):
            fields = raw[len("Dialogue:"):].strip().split(",", 9)
            rows.append((fields[1], fields[2], fields[9]))
    return rows


def _assert_prepared_guarantees(out, raw_lines, *, duration_sec=None,
                                early_bound=True):
    """The invariants prepare_lines promises, checked on one output.

    * sorted, pairwise non-overlapping, every caption >= the floor and
      inside [0, duration_sec];
    * every caption starts exactly on one of the input starts (a caption is
      never pulled earlier than the first line it shows);
    * every input text is on screen in a caption that starts at or before the
      line does, and (``early_bound``) less than one floor before it.
    """
    eps = 1e-9
    for caption in out:
        assert caption["end"] - caption["start"] >= MIN_LINE_DURATION_SEC - eps
        assert caption["start"] >= 0.0
        if duration_sec is not None:
            assert caption["end"] <= duration_sec + eps
        assert any(abs(caption["start"] - ln["start"]) < eps for ln in raw_lines)
    for earlier, later in zip(out, out[1:]):
        assert earlier["start"] <= later["start"]
        assert earlier["end"] <= later["start"] + eps
    for line in raw_lines:
        holders = [c for c in out
                   if f" {line['text']} " in f" {c['text'].replace(chr(10), ' ')} "]
        assert holders, f"{line['text']!r} disappeared"
        caption = holders[0]
        assert caption["start"] <= line["start"] + eps
        if early_bound:
            assert line["start"] - caption["start"] < MIN_LINE_DURATION_SEC


class TestQuickLineMerge:
    # c9ba8c0a-fr, caption e31da1d0 (2026-10-20). That caption row no longer
    # exists in the database (the run was re-captioned; the current caption
    # c02aa503 is a fresh transcription with different lines), so these are
    # the timings recorded in AICP-247. Line 1's text and exact span are a
    # reconstruction: the issue records only that it precedes "Quoi ?". The
    # end of "Ok." is not recorded either — only that the floor extended it
    # to 2.4, i.e. it ended before 2.4 — so the test runs over that range.
    CAMERA = "off-screen camera person"
    WOMAN = "blonde woman with glasses in a light blue t-shirt"

    def _c9ba8c0a(self, ok_end=2.1, *, speakers=True):
        lines = [
            {"start": 0.0, "end": 1.1, "speaker": self.CAMERA,
             "text": "Elle lit mal."},
            {"start": 1.1, "end": 1.5, "speaker": self.WOMAN, "text": "Quoi ?"},
            {"start": 1.5, "end": 1.8, "speaker": self.CAMERA, "text": "Mets ça."},
            {"start": 1.9, "end": ok_end, "speaker": self.WOMAN, "text": "Ok."},
        ]
        if not speakers:
            for line in lines:
                del line["speaker"]
        return lines

    @pytest.mark.parametrize("ok_end", [1.9, 2.0, 2.1, 2.3])
    def test_c9ba8c0a_keeps_quoi_and_mets_ca(self, ok_end):
        raw = self._c9ba8c0a(ok_end)
        out = prepare_lines(raw, duration_sec=9.0)
        assert out == [
            {"start": 0.0, "end": 1.1, "text": "Elle lit mal."},
            {"start": 1.1, "end": pytest.approx(1.9), "text": "Quoi ?\nMets ça."},
            {"start": 1.9, "end": pytest.approx(2.4), "text": "Ok."},
        ]
        _assert_prepared_guarantees(out, raw, duration_sec=9.0)

    def test_c9ba8c0a_burns_all_four_texts(self):
        doc = build_ass(self._c9ba8c0a(), width=1080, height=1920,
                        language="fr", duration_sec=9.0)
        rows = _dialogue_rows(doc)
        # The two quick replies share ONE caption, each speaker on a row.
        assert rows == [
            ("0:00:00.00", "0:00:01.10", "Elle lit mal."),
            ("0:00:01.10", "0:00:01.90", "Quoi ?\\NMets ça."),
            ("0:00:01.90", "0:00:02.40", "Ok."),
        ]
        on_screen = " ".join(text.replace("\\N", " ") for _, _, text in rows)
        for text in ("Elle lit mal.", "Quoi ?", "Mets ça.", "Ok."):
            assert text in on_screen

    def test_without_speakers_the_merge_is_one_space_joined_row(self):
        out = prepare_lines(self._c9ba8c0a(speakers=False), duration_sec=9.0)
        assert [ln["text"] for ln in out] == ["Elle lit mal.", "Quoi ? Mets ça.", "Ok."]
        rows = _dialogue_rows(build_ass(self._c9ba8c0a(speakers=False),
                                        width=1080, height=1920, duration_sec=9.0))
        assert rows[1] == ("0:00:01.10", "0:00:01.90", "Quoi ? Mets ça.")

    def test_same_speaker_merge_is_space_joined(self):
        out = prepare_lines([
            {"start": 0.0, "end": 0.2, "speaker": "A", "text": "Oh."},
            {"start": 0.2, "end": 0.4, "speaker": " a ", "text": "Wow."},
        ])
        assert [ln["text"] for ln in out] == ["Oh. Wow."]

    def test_speaker_row_break_can_be_switched_off(self, monkeypatch):
        from app import captions as captions_mod
        monkeypatch.setattr(captions_mod, "SPEAKER_CHANGE_STARTS_NEW_ROW", False)
        out = prepare_lines(self._c9ba8c0a(), duration_sec=9.0)
        assert out[1]["text"] == "Quoi ? Mets ça."

    def test_chain_of_quick_lines_folds_until_one_can_keep_the_floor(self):
        # Four replies 0.2s apart: A+B+C fold into one caption (each
        # truncation would leave it under 0.5s); D starts 0.6s after the
        # merged caption began, so the caption is truncated there and D is
        # shown on time.
        raw = [
            {"start": 0.0, "end": 0.1, "text": "A1"},
            {"start": 0.2, "end": 0.3, "text": "B2"},
            {"start": 0.4, "end": 0.5, "text": "C3"},
            {"start": 0.6, "end": 0.7, "text": "D4"},
        ]
        out = prepare_lines(raw)
        assert out == [
            {"start": 0.0, "end": pytest.approx(0.6), "text": "A1 B2 C3"},
            {"start": 0.6, "end": pytest.approx(1.1), "text": "D4"},
        ]
        _assert_prepared_guarantees(out, raw)

    def test_chain_of_alternating_speakers_falls_back_to_the_normal_wrap(self):
        # Three speaker changes in one caption need three rows; the house
        # style allows two, so the speaker rows give way to the ordinary
        # balanced wrap rather than costing a second (time-split) caption.
        raw = [
            {"start": 5.0, "end": 5.1, "speaker": "A", "text": "Quoi ?"},
            {"start": 5.15, "end": 5.3, "speaker": "B", "text": "Mets ça."},
            {"start": 5.3, "end": 5.45, "speaker": "A", "text": "Ok."},
        ]
        out = prepare_lines(raw)
        assert len(out) == 1
        assert out[0]["text"] == "Quoi ?\nMets ça.\nOk."
        assert out[0]["start"] == 5.0
        rows = _dialogue_rows(build_ass(raw, width=1080, height=1920))
        assert rows == [("0:00:05.00", "0:00:05.80", "Quoi ? Mets ça. Ok.")]

    def test_long_chain_never_overlaps_or_loses_a_line(self):
        raw = [{"start": round(0.15 * i, 2), "end": round(0.15 * i + 0.1, 2),
                "speaker": "AB"[i % 2], "text": f"w{i}"} for i in range(12)]
        out = prepare_lines(raw, duration_sec=30.0)
        _assert_prepared_guarantees(out, raw, duration_sec=30.0)
        shown = " ".join(c["text"].replace("\n", " ") for c in out).split()
        assert shown == [ln["text"] for ln in raw]

    def test_merge_of_a_long_and_a_quick_line_keeps_both(self):
        # A long line answered 0.3s in (people talking over each other):
        # truncating the long line would leave 0.3s, so it is merged, not
        # lost — the long one used to vanish entirely.
        raw = [
            {"start": 2.0, "end": 6.0, "text": "a long sentence that runs on"},
            {"start": 2.3, "end": 2.6, "text": "yes"},
            {"start": 6.5, "end": 7.5, "text": "after"},
        ]
        out = prepare_lines(raw)
        assert out == [
            {"start": 2.0, "end": 6.0, "text": "a long sentence that runs on yes"},
            {"start": 6.5, "end": 7.5, "text": "after"},
        ]

    def test_quick_pair_at_the_clip_end_never_runs_past_it(self):
        # "Mets ça." starts 0.4s before the clip ends — too late for the
        # floor on its own (it used to be dropped in step 3). It joins
        # "Quoi ?" and the merged caption ends exactly at the clip end.
        raw = [
            {"start": 9.1, "end": 9.5, "speaker": "A", "text": "Quoi ?"},
            {"start": 9.6, "end": 9.8, "speaker": "B", "text": "Mets ça."},
        ]
        out = prepare_lines(raw, duration_sec=10.0)
        assert out == [{"start": 9.1, "end": 10.0, "text": "Quoi ?\nMets ça."}]
        _assert_prepared_guarantees(out, raw, duration_sec=10.0, early_bound=False)
        rows = _dialogue_rows(build_ass(raw, width=1080, height=1920,
                                        duration_sec=10.0))
        assert rows == [("0:00:09.10", "0:00:10.00", "Quoi ?\\NMets ça.")]

    def test_quick_chain_whose_merge_is_clamped_at_the_clip_end(self):
        raw = [
            {"start": 9.2, "end": 9.3, "text": "Hm ?"},
            {"start": 9.4, "end": 9.45, "text": "Oui."},
            {"start": 9.6, "end": 9.65, "text": "Bon."},
        ]
        out = prepare_lines(raw, duration_sec=10.0)
        assert out == [{"start": 9.2, "end": 10.0, "text": "Hm ? Oui. Bon."}]
        _assert_prepared_guarantees(out, raw, duration_sec=10.0, early_bound=False)

    def test_tail_line_after_a_gap_is_still_dropped(self, caplog):
        # Nothing close enough before it to join: an isolated 0.2s flash at
        # the cut stays out, exactly as before.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = prepare_lines([
                {"start": 5.0, "end": 6.0, "text": "earlier"},
                {"start": 9.8, "end": 9.9, "text": "at the cut"},
            ], duration_sec=10.0)
        assert [ln["text"] for ln in out] == ["earlier"]
        assert any("dropping" in msg for msg in _warnings(caplog))

    @pytest.mark.parametrize("start", [0.08, 0.2, 0.44, 0.45, 3.3])
    def test_short_line_is_not_lost_to_float_rounding(self, start):
        # (start + 0.5) - start rounds BELOW 0.5 for these starts; the old
        # subtraction-form check read that as "no room" and dropped the line
        # even with no clip end anywhere near (pre-existing, found here).
        out = prepare_lines([{"start": start, "end": start + 0.05, "text": "Hm."}],
                            duration_sec=30.0)
        assert [ln["text"] for ln in out] == ["Hm."]
        assert out[0]["end"] == start + MIN_LINE_DURATION_SEC

    def test_newline_inside_one_transcript_line_is_a_space(self):
        out = prepare_lines([{"start": 0.0, "end": 2.0, "text": "two\nrows\r\nhere"}])
        assert out[0]["text"] == "two rows here"

    def test_seeded_random_transcripts_keep_every_guarantee(self):
        import random
        rng = random.Random(247)
        for _ in range(300):
            duration = rng.choice([None, 3.0, 6.0])
            raw = []
            for i in range(rng.randint(1, 12)):
                start = round(rng.uniform(0.0, 6.5), 2)
                raw.append({"start": start,
                            "end": round(start + rng.uniform(0.0, 1.2), 2),
                            "speaker": rng.choice(["A", "B", None]),
                            "text": f"t{i}"})
            out = prepare_lines(raw, duration_sec=duration)
            for earlier, later in zip(out, out[1:]):
                assert earlier["end"] <= later["start"] + 1e-9
            for caption in out:
                assert caption["end"] - caption["start"] >= MIN_LINE_DURATION_SEC - 1e-9
                if duration is not None:
                    assert caption["end"] <= duration + 1e-9
            shown = set(" ".join(c["text"].replace("\n", " ") for c in out).split())
            for line in raw:
                if line["text"] in shown:
                    continue
                # The only remaining drops: a line starting at/after the clip
                # end, or an isolated tail line with no room for the floor.
                assert duration is not None
                assert line["start"] > duration - MIN_LINE_DURATION_SEC


class TestMergeKnowsTheLayout:
    """Terra gate round 1 (AICP-247): a merge or a truncation must leave a
    caption the time its RENDERED captions need, or _split_windows drops the
    tail — which is where a merged-in reply lands."""

    LONG = ("this opening line is long enough to need more than the two rows "
            "one caption allows at this frame size")

    def _on_screen(self, doc):
        return " ".join(text.replace("\\N", " ") for _, _, text in _dialogue_rows(doc))

    def test_reply_merged_into_a_long_line_is_not_dropped_by_the_split(self, caplog):
        lines = [
            {"start": 0.0, "end": 0.5, "text": self.LONG},
            {"start": 0.2, "end": 0.5, "text": "IMPORTANT REPLY"},
        ]
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            doc = build_ass(lines, width=1080, height=1920, duration_sec=20.0)
        assert "IMPORTANT REPLY" in self._on_screen(doc)
        assert not any("dropping the last" in msg for msg in _warnings(caplog))

    def test_long_line_cut_by_an_overlapping_neighbour_keeps_its_tail(self, caplog):
        # On main the long line was truncated to 0.6s and _split_windows
        # dropped its second caption; now it needs 1.0s, so the neighbour
        # is merged instead and every word is shown.
        lines = [
            {"start": 0.0, "end": 3.0, "text": self.LONG},
            {"start": 0.6, "end": 3.0, "text": "next"},
        ]
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            doc = build_ass(lines, width=1080, height=1920, duration_sec=20.0)
        shown = self._on_screen(doc)
        assert self.LONG in shown and shown.endswith("next")
        assert not any("dropping the last" in msg for msg in _warnings(caplog))

    def test_every_rendered_caption_of_a_merge_gets_the_floor(self):
        lines = [
            {"start": 0.0, "end": 1.0, "text": self.LONG},
            {"start": 0.1, "end": 1.0, "text": "Yes"},
        ]
        doc = build_ass(lines, width=1080, height=1920, duration_sec=20.0)
        rows = _dialogue_rows(doc)
        assert len(rows) >= 2

        def secs(ts):
            h, m, rest = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(rest)

        for start, end, _ in rows:
            assert secs(end) - secs(start) >= MIN_LINE_DURATION_SEC - 0.011
        assert self._on_screen(doc).endswith("Yes")

    def test_merged_end_never_passes_the_clip(self):
        lines = [
            {"start": 9.0, "end": 9.4, "text": self.LONG},
            {"start": 9.2, "end": 9.4, "text": "IMPORTANT REPLY"},
        ]
        out = prepare_lines(lines, duration_sec=10.0,
                            min_duration_for=lambda text: 5.0)
        assert out[-1]["end"] == 10.0

    def test_hostile_min_duration_for_falls_back_to_the_floor(self):
        def boom(text):
            raise RuntimeError("layout exploded")

        out = prepare_lines([
            {"start": 0.0, "end": 0.1, "text": "a"},
            {"start": 0.2, "end": 0.3, "text": "b"},
        ], min_duration_for=boom)
        assert out == [{"start": 0.0, "end": pytest.approx(0.7), "text": "a b"}]

    def test_generator_input_survives_the_two_passes(self):
        doc = build_ass((ln for ln in [{"start": 0.0, "end": 1.0, "text": "once"}]),
                        width=1080, height=1920)
        assert self._on_screen(doc) == "once"

    def test_latin_replies_in_a_cjk_job_keep_their_spaces(self):
        raw = [
            {"start": 0.0, "end": 0.1, "speaker": "A", "text": "Hello"},
            {"start": 0.1, "end": 0.2, "speaker": "B", "text": "Yes"},
            {"start": 0.2, "end": 0.3, "speaker": "A", "text": "Okay"},
        ]
        rows = _dialogue_rows(build_ass(raw, width=1080, height=1920, language="ja"))
        assert [text for _, _, text in rows] == ["Hello Yes Okay"]
        assert layout_text("何？\nこれ。\nはい。", cjk=True) == \
            layout_text("何？これ。はい。", cjk=True)


class TestTerraRound2:
    def test_mixed_script_touching_words_keep_their_space(self):
        out = prepare_lines([
            {"start": 0.0, "end": 0.1, "text": "日本hello"},
            {"start": 0.1, "end": 0.2, "text": "world日本"},
        ])
        assert out[0]["text"] == "日本hello world日本"

    def test_fullwidth_latin_words_keep_their_space(self):
        out = prepare_lines([
            {"start": 0.0, "end": 0.1, "text": "ＨＥＬＬＯ"},
            {"start": 0.1, "end": 0.2, "text": "ＷＯＲＬＤ２"},
        ])
        assert out[0]["text"] == "ＨＥＬＬＯ ＷＯＲＬＤ２"

    def test_fullwidth_digit_against_cjk_joins_without_a_space(self):
        out = prepare_lines([
            {"start": 0.0, "end": 0.1, "text": "第１"},
            {"start": 0.1, "end": 0.2, "text": "回"},
        ])
        assert out[0]["text"] == "第１回"

    def test_cjk_punctuation_joins_without_a_space(self):
        out = prepare_lines([
            {"start": 0.0, "end": 0.1, "text": "何？"},
            {"start": 0.1, "end": 0.2, "text": "これ。"},
        ])
        assert out[0]["text"] == "何？これ。"

    def test_windows_are_sized_in_the_font_that_renders_them(self):
        # No language. A long Latin line plus a quick reply is merged and
        # held to what its text needs, which only then reaches a Japanese
        # tail line at the clip end: the second pass's text is CJK, so the
        # file must render — and have been sized — in the CJK font.
        long = " ".join(["word"] * 60)
        lines = [
            {"start": 8.0, "end": 8.4, "text": long},
            {"start": 8.2, "end": 8.4, "text": "yes"},
            {"start": 9.8, "end": 9.9, "text": "はい"},
        ]
        doc = build_ass(lines, width=1080, height=1920, duration_sec=10.0)
        style = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
        assert style.split(",")[1] == CJK_FONT
        for _, end, _ in _dialogue_rows(doc):
            assert end <= "0:00:10.00"


class TestLayoutSpeakerRows:
    def test_two_speakers_that_fit_get_a_row_each(self):
        assert layout_text("Quoi ?\nMets ça.", cjk=False) == [["Quoi ?", "Mets ça."]]

    def test_more_speaker_rows_than_the_cap_fall_back_to_the_house_wrap(self):
        assert layout_text("Quoi ?\nMets ça.\nOk.", cjk=False) == \
            layout_text("Quoi ? Mets ça. Ok.", cjk=False)

    def test_placement_row_cap_allows_a_third_speaker_row(self):
        assert layout_text("Quoi ?\nMets ça.\nOk.", cjk=False, max_rows=3) == \
            [["Quoi ?", "Mets ça.", "Ok."]]

    def test_long_speaker_parts_fall_back_and_every_row_fits(self):
        long_a = "this first speaker says quite a lot of words in one go"
        long_b = "and the second one answers with just as many words back"
        captions = layout_text(f"{long_a}\n{long_b}", cjk=False)
        assert captions == layout_text(f"{long_a} {long_b}", cjk=False)
        limit = usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT)
        for caption in captions:
            assert len(caption) <= MAX_ROWS
            assert all(w <= limit for w in _row_widths(caption))

    def test_cjk_speaker_parts(self):
        assert layout_text("何？\nこれ。", cjk=True) == [["何？", "これ。"]]
        assert layout_text("何？\nこれ。\nはい。", cjk=True) == \
            layout_text("何？これ。はい。", cjk=True)


# ---------------------------------------------------------------------------
# build_ass
# ---------------------------------------------------------------------------


class TestBuildAss:
    LINES = [
        {"start": 0.0, "end": 2.0, "text": "hello there"},
        {"start": 2.5, "end": 4.0, "text": "second line"},
    ]

    def test_play_res_matches_the_passed_frame_size(self):
        doc = build_ass(self.LINES, width=1080, height=1920)
        assert "PlayResX: 1080" in doc
        assert "PlayResY: 1920" in doc

    def test_style_line_has_bold_minus_one_and_opaque_white(self):
        doc = build_ass(self.LINES, width=1080, height=1920)
        style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
        fields = style_line[len("Style:"):].strip().split(",")
        # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,
        # OutlineColour, BackColour, Bold, ...
        assert fields[3] == "&H00FFFFFF"  # PrimaryColour: opaque white
        assert fields[7] == "-1"  # Bold

    def test_style_line_declares_the_house_alignment_and_border_style(self):
        doc = build_ass(self.LINES, width=1080, height=1920)
        style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
        fields = style_line[len("Style:"):].strip().split(",")
        assert fields[15] == str(BORDER_STYLE_BOX)
        assert fields[18] == str(ALIGNMENT_BOTTOM_CENTER)

    def test_the_caption_band_is_semi_transparent_and_unshadowed(self):
        # Under BorderStyle 3 the box is filled with OutlineColour, and any
        # non-zero Shadow paints an OPAQUE BackColour box behind it that
        # fills the transparency back in — verified on a rendered frame, and
        # invisible in the ASS text itself, hence the assertion.
        doc = build_ass(self.LINES, width=1080, height=1920)
        style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
        fields = style_line[len("Style:"):].strip().split(",")
        assert fields[5] == _ass_colour(0, 0, 0, alpha=BOX_ALPHA)  # OutlineColour
        assert 0 < BOX_ALPHA < 0xFF  # neither invisible nor a solid black bar
        assert float(fields[17]) == 0  # Shadow

    def test_libass_is_allowed_to_re_wrap_as_a_backstop(self):
        # WrapStyle 2 ("never wrap") is what let v1's over-wide rows reach
        # the frame edge instead of being re-wrapped.
        assert "WrapStyle: 0" in build_ass(self.LINES, width=1080, height=1920)

    def test_sizes_scale_with_height(self):
        # Round-number heights so independent rounding of each side can't
        # introduce a +/-1px artifact into an exact-doubling assertion.
        small = build_ass(self.LINES, width=1000, height=1000)
        large = build_ass(self.LINES, width=1000, height=2000)

        def fontsize(doc):
            style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
            return int(style_line[len("Style:"):].strip().split(",")[2])

        assert fontsize(large) == fontsize(small) * 2
        assert fontsize(small) == int(round(1000 * FONT_SIZE_RATIO))

    def test_horizontal_margin_scales_with_width_not_height(self):
        wide = build_ass(self.LINES, width=2000, height=1000)
        narrow = build_ass(self.LINES, width=1000, height=1000)

        def margin_l(doc):
            style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
            fields = style_line[len("Style:"):].strip().split(",")
            return int(fields[19])  # MarginL

        assert margin_l(wide) == margin_l(narrow) * 2
        assert margin_l(wide) == int(round(2000 * MARGIN_H_RATIO))

    def test_vertical_margin_scales_with_height(self):
        doc = build_ass(self.LINES, width=1080, height=1000)
        style_line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
        fields = style_line[len("Style:"):].strip().split(",")
        assert int(fields[21]) == int(round(1000 * MARGIN_V_RATIO))  # MarginV

    def test_empty_input_yields_a_header_only_valid_file(self):
        doc = build_ass([], width=1080, height=1920)
        assert "[Script Info]" in doc
        assert "[V4+ Styles]" in doc
        assert "[Events]" in doc
        assert "PlayResX: 1080" in doc
        assert "Dialogue:" not in doc

    def test_garbage_input_does_not_raise_and_yields_no_dialogue(self):
        doc = build_ass(["nonsense", None, {}], width=1080, height=1920)
        assert "Dialogue:" not in doc

    def test_dialogue_rows_use_the_default_style_and_expected_timestamps(self):
        doc = build_ass(self.LINES, width=1080, height=1920)
        dialogue_lines = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
        assert len(dialogue_lines) == 2
        assert dialogue_lines[0].startswith("Dialogue: 0,0:00:00.00,0:00:02.00,Default,")
        assert dialogue_lines[1].startswith("Dialogue: 0,0:00:02.50,0:00:04.00,Default,")

    def test_dialogue_text_is_escaped(self):
        doc = build_ass(
            [{"start": 0.0, "end": 2.0, "text": "brace {test} and \\slash"}],
            width=1080, height=1920,
        )
        dialogue_line = next(ln for ln in doc.splitlines() if ln.startswith("Dialogue:"))
        assert "{test}" not in dialogue_line
        assert "\\slash" not in dialogue_line

    def test_japanese_language_selects_cjk_font(self):
        doc = build_ass(self.LINES, width=1080, height=1920, language="ja")
        assert f"Style: Default,{CJK_FONT}," in doc

    def test_default_language_selects_default_font(self):
        doc = build_ass(self.LINES, width=1080, height=1920, language="en")
        assert f"Style: Default,{DEFAULT_FONT}," in doc

    def test_unknown_language_sniffs_from_the_lines_own_text(self):
        japanese_lines = [{"start": 0.0, "end": 2.0, "text": "こんにちは世界"}]
        doc = build_ass(japanese_lines, width=1080, height=1920, language=None)
        assert f"Style: Default,{CJK_FONT}," in doc

    def test_no_caption_ever_exceeds_the_row_cap(self):
        long_text = " ".join(["word"] * 40)
        doc = build_ass(
            [{"start": 0.0, "end": 30.0, "text": long_text}],
            width=1080, height=1920,
        )
        # \N is the ASS row-break; at most MAX_ROWS rows means at most
        # MAX_ROWS - 1 row-break codes in one Dialogue row.
        for line in doc.splitlines():
            if line.startswith("Dialogue:"):
                assert line.count("\\N") <= MAX_ROWS - 1

    def test_a_long_line_is_split_in_time_rather_than_truncated(self):
        long_text = " ".join(["word"] * 40)
        doc = build_ass(
            [{"start": 0.0, "end": 30.0, "text": long_text}],
            width=1080, height=1920,
        )
        dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
        assert len(dialogue) > 1
        shown = " ".join(ln.split(",", 9)[9].replace("\\N", " ") for ln in dialogue)
        assert shown.split() == ["word"] * 40

    def test_split_windows_are_contiguous_and_stay_inside_the_line(self):
        doc = build_ass(
            [{"start": 4.0, "end": 10.0, "text": " ".join(["word"] * 30)}],
            width=1080, height=1920,
        )
        windows = [
            (ln.split(",")[1], ln.split(",")[2])
            for ln in doc.splitlines() if ln.startswith("Dialogue:")
        ]
        assert len(windows) > 1
        assert windows[0][0] == format_timestamp(4.0)
        assert windows[-1][1] == format_timestamp(10.0)
        for earlier, later in zip(windows, windows[1:]):
            assert earlier[1] == later[0]  # no gap, no overlap

    def test_a_window_too_short_for_every_caption_drops_the_tail(self, caplog):
        # 30 words in 0.6s: one caption fits at the MIN_LINE_DURATION_SEC
        # floor, so the rest are dropped rather than flashed for a few frames.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            doc = build_ass(
                [{"start": 0.0, "end": 0.6, "text": " ".join(["word"] * 30)}],
                width=1080, height=1920,
            )
        dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
        assert len(dialogue) == 1
        assert any("dropping the last" in msg for msg in _warnings(caplog))

    def test_every_rendered_row_fits_the_frame(self):
        # The end-to-end version of the v1 bug: whatever the transcript says,
        # nothing that reaches a Dialogue row is wider than the safe area.
        lines = [
            {"start": 0.0, "end": 4.0,
             "text": "and then it reads it to you at any speed."},
            {"start": 5.0, "end": 9.0,
             "text": "And I can even switch the voices. So watch, this is "
                     "Gwyneth Paltrow."},
            {"start": 10.0, "end": 12.0, "text": "What's your favorite voice?"},
        ]
        doc = build_ass(lines, width=1080, height=1920, language="en")
        limit = usable_width_px(1080, 1920)
        for line in doc.splitlines():
            if not line.startswith("Dialogue:"):
                continue
            for row in line.split(",", 9)[9].split("\\N"):
                assert measure_text(
                    row, font=DEFAULT_FONT, size_px=font_size_px(1920)
                ) <= limit


# ---------------------------------------------------------------------------
# Module-level constants sanity (the "reference by name" requirement)
# ---------------------------------------------------------------------------


def test_style_version_is_the_documented_string():
    assert CAPTION_STYLE_VERSION == "captions/v2"


def test_house_style_constants_match_the_design_doc():
    assert MAX_ROWS == 2
    assert FONT_SIZE_RATIO == 0.036
    assert MARGIN_H_RATIO == 0.06
    assert MARGIN_V_RATIO == 0.16


def test_nominal_frame_is_the_delivery_frame():
    assert (NOMINAL_WIDTH, NOMINAL_HEIGHT) == (1080, 1920)
    assert font_size_px(NOMINAL_HEIGHT) == int(round(NOMINAL_HEIGHT * FONT_SIZE_RATIO))
    # The usable width is the frame less both margins less the box padding,
    # i.e. strictly narrower than the naive "frame minus margins".
    assert usable_width_px(NOMINAL_WIDTH, NOMINAL_HEIGHT) < NOMINAL_WIDTH * (
        1 - 2 * MARGIN_H_RATIO
    )


# ---------------------------------------------------------------------------
# Italian (it) — the Latin-script path, accents and elisions intact
# ---------------------------------------------------------------------------

# The accents, the grave/acute pair and the elided apostrophe are the
# characters an escaping or font bug would eat.
IT_TEXT = "Perché è già così? Un po' di più."


class TestItalianCaptions:
    def test_italian_uses_the_default_latin_font(self):
        """No CJK glyphs needed — Italian renders in DejaVu Sans like en/es/pt/de."""
        assert font_for_language("it") == DEFAULT_FONT
        assert font_for_language("IT") == DEFAULT_FONT
        assert font_for_language("it-IT") == DEFAULT_FONT

    def test_italian_text_is_not_sniffed_as_cjk(self):
        """The CJK sniff must not fire on accented Latin."""
        assert font_for_language(None, sample_text=IT_TEXT) == DEFAULT_FONT
        assert font_for_language("xx", sample_text=IT_TEXT) == DEFAULT_FONT

    def test_build_ass_selects_the_default_font_for_italian(self):
        doc = build_ass(
            [{"start": 0.0, "end": 2.0, "text": IT_TEXT}],
            width=1080, height=1920, language="it",
        )
        assert f"Style: Default,{DEFAULT_FONT}," in doc

    def test_accents_and_apostrophes_survive_escaping(self):
        """escape_text only substitutes '{', '}' and '\\' — not accents."""
        assert escape_text(IT_TEXT) == IT_TEXT
        for ch in "éèàòùì'":
            assert ch in escape_text(f"x{ch}y")

    def test_accented_dialogue_reaches_the_ass_document(self):
        doc = build_ass(
            [{"start": 0.0, "end": 2.0, "text": IT_TEXT}],
            width=1080, height=1920, language="it",
        )
        dialogue = next(ln for ln in doc.splitlines() if ln.startswith("Dialogue:"))
        # Wrapping may insert \N between rows, so compare on the joined text.
        rendered = dialogue.split(",", 9)[-1].replace(r"\N", " ")
        assert "Perché" in rendered
        assert "è già così?" in rendered
        assert "po'" in rendered

    def test_italian_wraps_without_losing_characters(self):
        """A long Italian line still round-trips every word through the wrap.

        A line this long is split into SEVERAL timed caption events (and each
        into up to MAX_ROWS rows), so the comparison joins them all back up —
        the property under test is that nothing is dropped or mangled, not how
        many events it took.
        """
        long_it = (
            "Perché è già così difficile capire un po' di più "
            "quello che stiamo però cercando di dire qui."
        )
        doc = build_ass(
            [{"start": 0.0, "end": 4.0, "text": long_it}],
            width=1080, height=1920, language="it",
        )
        events = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
        assert len(events) > 1  # it really does split
        rendered = " ".join(
            ev.split(",", 9)[-1].replace(r"\N", " ") for ev in events
        )
        assert rendered.split() == long_it.split()


# ---------------------------------------------------------------------------
# Polish (pl) — Latin script with ogonek/acute/stroke/dot diacritics
# ---------------------------------------------------------------------------

# Every Polish diacritic, lower and upper case. A font missing any one of them
# burns a tofu box for it while ffmpeg still exits 0.
PL_DIACRITICS = "ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ"
PL_TEXT = "Zażółć gęślą jaźń — ŹDŹBŁO, ŁĄKA."
FR_DIACRITICS = "àâæçéèêëîïôœùûüÿ ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ « »"
FR_TEXT = "Ça a été déjà vu à Noël — « L'ŒUVRE » d'un être naïf."


def _libass_render_log(tmp_path, ass_doc: str) -> str:
    """Burn *ass_doc* onto a few blank frames with the real libass and return
    ffmpeg's verbose log.

    No video is written (``-f null``); what matters is libass's font
    selection, which it logs as ``fontselect: (<requested>, ...) -> <file>``
    and, for every codepoint the selected font lacks, ``Glyph 0x<cp> not
    found, selecting one more font``. That second line is the tofu signal.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    filters = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    if " subtitles " not in filters:
        pytest.skip("ffmpeg built without libass (no subtitles filter)")

    ass_path = tmp_path / "pl.ass"
    ass_path.write_text(ass_doc, encoding="utf-8")
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "verbose",
            "-f", "lavfi", "-i", "color=c=black:s=540x960:d=1",
            "-vf", f"subtitles={ass_path}",
            "-frames:v", "3", "-f", "null", "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stderr


def _font_selections(log: str) -> list[str]:
    return [ln for ln in log.splitlines() if "fontselect:" in ln]


def _missing_glyph_lines(log: str) -> list[str]:
    return [ln for ln in log.splitlines() if "Glyph 0x" in ln and "not found" in ln]


class TestPolishCaptions:
    def test_polish_uses_the_default_latin_font(self):
        assert font_for_language("pl") == DEFAULT_FONT
        assert font_for_language("PL") == DEFAULT_FONT
        assert font_for_language("pl-PL") == DEFAULT_FONT

    def test_polish_text_is_not_sniffed_as_cjk(self):
        assert font_for_language(None, sample_text=PL_DIACRITICS) == DEFAULT_FONT
        assert font_for_language("xx", sample_text=PL_TEXT) == DEFAULT_FONT

    def test_diacritics_survive_escaping_and_reach_the_ass_document(self):
        assert escape_text(PL_DIACRITICS) == PL_DIACRITICS
        doc = build_ass(
            [{"start": 0.0, "end": 2.0, "text": PL_DIACRITICS}],
            width=1080, height=1920, language="pl",
        )
        assert f"Style: Default,{DEFAULT_FONT}," in doc
        dialogue = next(ln for ln in doc.splitlines() if ln.startswith("Dialogue:"))
        rendered = dialogue.split(",", 9)[-1].replace(r"\N", " ")
        assert rendered.split() == PL_DIACRITICS.split()

    def test_burned_polish_caption_has_no_missing_glyphs(self, tmp_path):
        """The real libass render: DejaVu Sans is selected and covers every
        Polish diacritic, so libass never has to fall back to another font."""
        doc = build_ass(
            [{"start": 0.0, "end": 1.0, "text": PL_DIACRITICS}],
            width=540, height=960, language="pl",
        )
        log = _libass_render_log(tmp_path, doc)

        selections = _font_selections(log)
        if not selections:
            pytest.skip("this libass build does not log font selection")
        # A missing DejaVu Sans FAILS rather than skips: it is the production
        # caption font (Dockerfile, docs/burned-captions.md §5), and a box
        # without it would burn Polish with whatever fontconfig substitutes.
        assert all("DejaVuSans" in ln for ln in selections), selections
        assert _missing_glyph_lines(log) == []

    def test_the_glyph_check_does_catch_tofu(self, tmp_path):
        """Negative control: the same DejaVu Sans style with Japanese text
        MUST log missing glyphs — otherwise the test above proves nothing."""
        doc = build_ass(
            [{"start": 0.0, "end": 1.0, "text": "PLACEHOLDER"}],
            width=540, height=960, language="pl",
        )
        assert f"Style: Default,{DEFAULT_FONT}," in doc
        log = _libass_render_log(tmp_path, doc.replace("PLACEHOLDER", "こんにちは"))
        selections = _font_selections(log)
        if not selections:
            pytest.skip("this libass build does not log font selection")
        # DejaVu was the font libass picked first; the miss is ITS miss.
        assert "DejaVuSans" in selections[0], selections
        assert _missing_glyph_lines(log), "libass logged no missing glyph for CJK in DejaVu Sans"


class TestFrenchCaptions:
    def test_french_uses_the_default_latin_font(self):
        assert font_for_language("fr") == DEFAULT_FONT
        assert font_for_language("FR") == DEFAULT_FONT
        assert font_for_language("fr-FR") == DEFAULT_FONT

    def test_french_text_is_not_sniffed_as_cjk(self):
        assert font_for_language(None, sample_text=FR_DIACRITICS) == DEFAULT_FONT
        assert font_for_language("xx", sample_text=FR_TEXT) == DEFAULT_FONT

    def test_diacritics_survive_escaping_and_reach_the_ass_document(self):
        assert escape_text(FR_DIACRITICS) == FR_DIACRITICS
        doc = build_ass(
            [{"start": 0.0, "end": 2.0, "text": FR_DIACRITICS}],
            width=1080, height=1920, language="fr",
        )
        assert f"Style: Default,{DEFAULT_FONT}," in doc
        # Wider than the Polish sample: the capitals "word" is hard-broken
        # across rows and the 2-row cap spills into a second timed caption,
        # so compare glyphs, not words — every one must still be burned.
        rendered = "".join(
            ln.split(",", 9)[-1].replace(r"\N", "")
            for ln in doc.splitlines() if ln.startswith("Dialogue:")
        )
        assert rendered.replace(" ", "") == FR_DIACRITICS.replace(" ", "")

    def test_burned_french_caption_has_no_missing_glyphs(self, tmp_path):
        """The real libass render: DejaVu Sans is selected and covers every
        French diacritic, so libass never has to fall back to another font."""
        doc = build_ass(
            [{"start": 0.0, "end": 1.0, "text": FR_DIACRITICS}],
            width=540, height=960, language="fr",
        )
        log = _libass_render_log(tmp_path, doc)

        selections = _font_selections(log)
        if not selections:
            pytest.skip("this libass build does not log font selection")
        # A missing DejaVu Sans FAILS rather than skips: it is the production
        # caption font (Dockerfile, docs/burned-captions.md §5), and a box
        # without it would burn French with whatever fontconfig substitutes.
        assert all("DejaVuSans" in ln for ln in selections), selections
        assert _missing_glyph_lines(log) == []
