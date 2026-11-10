"""
Google Sheets client for re-skin.

Thin wrapper around the Sheets v4 API, used to append rows to the
"Final Creatives" tracking spreadsheet. Auth reuses the same service-account
JSON key as the Drive client (``settings.GOOGLE_SERVICE_ACCOUNT_FILE``); the
target spreadsheet must be shared with the service-account email.

Rows are written by HEADER NAME, not by position: we read the header row of the
target tab and place each value under its matching column. This keeps the export
robust if the operator reorders or inserts columns in the sheet.
"""

import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# "Idea" column
# ---------------------------------------------------------------------------
# The Idea cell is a compact human key for the creative, consumed downstream by
# ad-platform naming, so it has a hard 40-char ceiling. The budget is FIXED per
# part rather than shared: project 27 + "_" + run 7 + "_" + model <=4 = 40. A
# fixed split is what the UI hints promise the operator ("first 27 chars of the
# project name reach Idea") — a shared budget would make the visible prefix
# depend on how long the *other* name happens to be.

IDEA_MAX_LEN = 40
IDEA_PROJECT_MAX_LEN = 27
IDEA_RUN_MAX_LEN = 7

_IDEA_DISALLOWED = re.compile(r"[^A-Za-z0-9_-]+")
_BRACKET_OPEN = "([{"
_BRACKET_CLOSE = ")]}"


def _strip_bracketed(text: str) -> str:
    """Drop every bracketed group — the brackets AND their contents.

    ``"SheIsCheatingOnMe (Jp)"`` -> ``"SheIsCheatingOnMe "``. Brackets usually
    hold a qualifier (locale, variant) that is noise in a 27-char slot, so it
    is dropped whole instead of eating the budget. Nesting is handled by the
    depth counter; an UNMATCHED opener drops the rest of the string (we cannot
    tell where the qualifier was meant to end), while an unmatched closer is
    simply skipped.
    """
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch in _BRACKET_OPEN:
            depth += 1
        elif ch in _BRACKET_CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def sanitize_idea_part(text: Optional[str]) -> str:
    """Reduce a project/run name to the characters Idea allows.

    Bracketed groups go first (see :func:`_strip_bracketed`), then every
    character outside ``[A-Za-z0-9_-]`` is REMOVED rather than replaced — so
    the parts collapse instead of growing separators:
    ``"My-Idea+ES"`` -> ``"My-IdeaES"``, ``"Listenify Work & Shopify"`` ->
    ``"ListenifyWorkShopify"``.

    Note this also strips non-Latin scripts; a name written entirely in e.g.
    Cyrillic sanitizes to "" and :func:`build_idea` falls back to the id prefix.
    """
    return _IDEA_DISALLOWED.sub("", _strip_bracketed(text or ""))


def _idea_part(name: Optional[str], fallback_id: str, max_len: int) -> str:
    """Sanitize *name* (or the id prefix when it is empty) and clamp to *max_len*."""
    part = sanitize_idea_part(name) or sanitize_idea_part(fallback_id)[:8]
    return part[:max_len]


def build_idea(
    project_name: Optional[str],
    run_name: Optional[str],
    model: Optional[str],
    *,
    project_id: str = "",
    run_id: str = "",
) -> str:
    """Build the ``Idea`` cell: ``{project}_{run}_{MODEL}``, at most 40 chars.

    Each part is sanitized (:func:`sanitize_idea_part`) and clamped to its own
    slot — 27 for the project, 7 for the run, the rest for the model's
    ``short_label``. An empty name falls back to the 8-char id prefix so a row
    is never keyed by an anonymous ``__SD25``.
    """
    from . import ai_models  # local import: keeps the registry off module import

    project = _idea_part(project_name, project_id, IDEA_PROJECT_MAX_LEN)
    run = _idea_part(run_name, run_id, IDEA_RUN_MAX_LEN)
    return f"{project}_{run}_{ai_models.short_label_for(model)}"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SheetsError(Exception):
    """Base exception for Google Sheets errors."""


class SheetsAuthError(SheetsError):
    """Raised when service-account authentication fails."""


class SheetsAppendError(SheetsError):
    """Raised when reading/writing the spreadsheet fails."""


def _quote_tab(tab: str) -> str:
    """Quote a sheet/tab name for an A1 range (tab names may contain spaces)."""
    return "'" + tab.replace("'", "''") + "'"


#: Leading characters a spreadsheet treats as the start of a formula (plus tab /
#: CR, which Excel strips before deciding).  AICP-160.
_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")


def neutralise_formula(value):
    """Return *value* defused against spreadsheet formula injection.

    A string starting with ``= + - @`` (or tab/CR) is written as ``'<value>``:
    under ``USER_ENTERED`` Sheets drops the apostrophe and stores the rest as
    literal text; in a CSV opened by Excel the apostrophe is the OWASP-sanctioned
    neutraliser.  Non-strings (numbers, None) and everything else are returned
    unchanged.  Every cell the app writes is data -- the app writes no formulas
    of its own (no HYPERLINK/IMAGE) -- so this is applied to every string cell.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_LEADERS):
        return "'" + value
    return value


class SheetsClient:
    """Thin wrapper around the Sheets v4 API.

    Parameters
    ----------
    service_account_file:
        Path to the service-account JSON key. Defaults to
        ``settings.GOOGLE_SERVICE_ACCOUNT_FILE``.
    service:
        Pre-built Sheets service object (for testing). When supplied the
        ``service_account_file`` is ignored and no credentials are needed.
    """

    _SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    def __init__(
        self,
        service_account_file: Optional[str] = None,
        service=None,
    ) -> None:
        self._sa_file = service_account_file
        self._service = service  # may be None; built lazily on first use

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def service(self):
        """Return (or lazily build) the Sheets service object."""
        if self._service is not None:
            return self._service

        if self._sa_file is None:
            from app.config import settings  # local import for testability

            self._sa_file = settings.GOOGLE_SERVICE_ACCOUNT_FILE

        sa_file = self._sa_file

        if not sa_file or not os.path.isfile(sa_file):
            raise SheetsAuthError(
                f"Service-account file not found: {sa_file!r}. "
                "Set GOOGLE_SERVICE_ACCOUNT_FILE or pass service_account_file."
            )

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=self._SCOPES
            )
            self._service = build(
                "sheets", "v4", credentials=creds, cache_discovery=False
            )
        except Exception as exc:
            raise SheetsAuthError(
                f"Failed to build Sheets service from {sa_file!r}: {exc}"
            ) from exc

        return self._service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_header(self, spreadsheet_id: str, tab: str) -> list[str]:
        """Return the first (header) row of *tab* as a list of strings."""
        rng = f"{_quote_tab(tab)}!1:1"
        try:
            resp = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=rng)
                .execute()
            )
        except Exception as exc:
            raise SheetsAppendError(
                f"Failed to read header of {tab!r} in {spreadsheet_id!r}: {exc}"
            ) from exc
        values = resp.get("values", [])
        return [str(c) for c in values[0]] if values else []

    def _row_count(self, spreadsheet_id: str, tab: str) -> int:
        """Return the total row count of *tab* via a cheap metadata call.

        Uses ``spreadsheets.get`` (sheet metadata) rather than
        ``spreadsheets.values.get`` (cell data), so no column/row VALUES are
        transferred — just the grid dimensions. Used to bound a
        :meth:`column_values` read to its last N rows instead of the whole
        column.
        """
        try:
            resp = (
                self.service.spreadsheets()
                .get(
                    spreadsheetId=spreadsheet_id,
                    ranges=[_quote_tab(tab)],
                    fields="sheets.properties.gridProperties.rowCount",
                )
                .execute()
            )
        except Exception as exc:
            raise SheetsAppendError(
                f"Failed to read row count of {tab!r} in {spreadsheet_id!r}: {exc}"
            ) from exc
        sheets = resp.get("sheets", [])
        if not sheets:
            return 0
        props = sheets[0].get("properties", {})
        return props.get("gridProperties", {}).get("rowCount", 0)

    def column_values(
        self,
        spreadsheet_id: str,
        tab: str,
        column_index: int,
        *,
        last_n: int | None = None,
    ) -> list[str]:
        """Return values in a 0-based *column_index* of *tab*.

        By default (``last_n=None``) reads the WHOLE column (incl. header) —
        the original behaviour. When *last_n* is given, only the LAST
        *last_n* rows are read: a cheap row-count lookup (:meth:`_row_count`)
        first, then a bounded ``{col}{start}:{col}`` range, so the read cost
        stays flat as the sheet grows instead of scanning the entire column
        on every dedupe check. The header row is excluded in this mode (the
        start row is always >= 2) since a bounded dedupe scan only cares
        about recent data rows, not the header label.

        ``_row_count`` returns the sheet's GRID row count (Sheets allocates a
        fixed grid, e.g. 1000 rows, regardless of how many rows actually hold
        data), NOT the data extent. On a tab whose data doesn't fill the
        grid, ``start_row`` can land past the last row that actually has a
        value, and ``values.get`` (which trims trailing blank rows from its
        response) then returns an empty list — which would make the dedupe
        wrongly conclude "not present" and double-append on a retry. So: a
        bounded read that comes back NON-empty provably reached down to the
        last data row (values.get returns every row from ``start_row``
        through the last non-blank row, so if any value came back, the tail
        is included) — the retry-duplicate window is covered. A bounded read
        that comes back EMPTY means ``start_row`` is past the data entirely,
        i.e. the tab has few data rows — exactly when a full-column
        fallback read is cheap. So we retry with the unbounded range in that
        case.
        """
        col_letter = _col_letter(column_index)
        if last_n is None:
            rng = f"{_quote_tab(tab)}!{col_letter}:{col_letter}"
        else:
            row_count = self._row_count(spreadsheet_id, tab)
            start_row = max(2, row_count - last_n)
            rng = f"{_quote_tab(tab)}!{col_letter}{start_row}:{col_letter}"
        values = self._get_column_range(spreadsheet_id, tab, col_letter, rng)
        if last_n is not None and not values:
            # Bounded read landed past the data extent (grid rowCount padding)
            # — fall back to the full column, which is cheap exactly in this
            # case (few data rows).
            fallback_rng = f"{_quote_tab(tab)}!{col_letter}:{col_letter}"
            values = self._get_column_range(spreadsheet_id, tab, col_letter, fallback_rng)
        return values

    def _get_column_range(
        self, spreadsheet_id: str, tab: str, col_letter: str, rng: str
    ) -> list[str]:
        """Execute a single ``values.get`` COLUMNS-major read for *rng*."""
        try:
            resp = (
                self.service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=rng,
                    majorDimension="COLUMNS",
                )
                .execute()
            )
        except Exception as exc:
            raise SheetsAppendError(
                f"Failed to read column {col_letter} of {tab!r}: {exc}"
            ) from exc
        cols = resp.get("values", [])
        return [str(c) for c in cols[0]] if cols else []

    def read_rows(
        self, spreadsheet_id: str, tab: str, last_column: str = "Z"
    ) -> list[list[str]]:
        """Return all data rows of *tab* (header row excluded) as string lists.

        Reads ``A2:<last_column>``; trailing empty cells are absent from the
        API response, so rows may be shorter than the requested width.
        """
        rng = f"{_quote_tab(tab)}!A2:{last_column}"
        try:
            resp = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=rng)
                .execute()
            )
        except Exception as exc:
            raise SheetsAppendError(
                f"Failed to read rows of {tab!r} in {spreadsheet_id!r}: {exc}"
            ) from exc
        return [[str(c) for c in row] for row in resp.get("values", [])]

    def append_row(self, spreadsheet_id: str, tab: str, row: list) -> dict:
        """Append a single *row* (list of cell values) to the end of *tab*."""
        rng = f"{_quote_tab(tab)}!A1"
        try:
            resp = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=rng,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [[neutralise_formula(c) for c in row]]},
                )
                .execute()
            )
        except Exception as exc:
            raise SheetsAppendError(
                f"Failed to append row to {tab!r} in {spreadsheet_id!r}: {exc}"
            ) from exc
        return resp


def _col_letter(index: int) -> str:
    """Convert a 0-based column index to an A1 column letter (0->A, 26->AA)."""
    letters = ""
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def build_final_creatives_row(
    header: list[str],
    *,
    fields: dict[str, str],
) -> list:
    """Build a row aligned to *header*, placing each *fields* value under the
    column whose header matches its key (case-insensitive, trimmed).

    Columns not present in *fields* are left blank. Field keys that don't match
    any header are ignored (logged). Returns a row sized to ``len(header)``.
    """
    norm = {h.strip().lower(): i for i, h in enumerate(header)}
    row: list = [""] * len(header)
    for key, value in fields.items():
        idx = norm.get(key.strip().lower())
        if idx is None:
            log.warning("Final Creatives: no column matching field %r", key)
            continue
        row[idx] = value
    return row
