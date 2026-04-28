"""Markdown formatting utilities.

Provides functions for generating and reformatting Markdown tables with
column-aligned padding, so tables are readable in both rendered Markdown
viewers and plain-text editors.
"""

from typing import List, Optional

import re


# ---------------------------------------------------------------------------
# Core table formatter
# ---------------------------------------------------------------------------

def format_md_table(
    headers: List[str],
    rows: List[List[str]],
    alignment: Optional[List[str]] = None,
) -> str:
    """Build a column-aligned Markdown table from headers and data rows.

    Parameters
    ----------
    headers : list of str
        Column header strings.
    rows : list of list of str
        Data rows.  Each inner list must have the same length as *headers*
        (or fewer — missing columns are treated as empty strings).
    alignment : list of str, optional
        Per-column alignment indicator: ``'left'`` (default), ``'right'``,
        or ``'center'``.  Controls the separator row only (Markdown
        renderers use this to align cell text).

    Returns
    -------
    str
        Multi-line string containing the full Markdown table (header,
        separator, and data rows), with every column padded to uniform
        width.
    """
    n_cols = len(headers)
    if alignment is None:
        alignment = ["left"] * n_cols
    if len(alignment) < n_cols:
        alignment = list(alignment) + ["left"] * (n_cols - len(alignment))

    # --- Compute column widths (minimum 3 to fit the separator dashes) ---
    col_widths = [max(len(h), 3) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # --- Header ---
    header_cells = [
        h.ljust(col_widths[i]) for i, h in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header_cells) + " |"

    # --- Separator ---
    sep_parts = []
    for i, align in enumerate(alignment):
        w = col_widths[i]
        if align == "right":
            sep_parts.append("-" * (w - 1) + ":")
        elif align == "center":
            sep_parts.append(":" + "-" * (w - 2) + ":")
        else:
            sep_parts.append("-" * w)
    sep_line = "| " + " | ".join(sep_parts) + " |"

    # --- Data rows ---
    data_lines = []
    for row in rows:
        cells = []
        for i in range(n_cols):
            cell = row[i] if i < len(row) else ""
            cells.append(cell.ljust(col_widths[i]))
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, sep_line] + data_lines)


# ---------------------------------------------------------------------------
# Post-processing: find and pad all tables in a Markdown document
# ---------------------------------------------------------------------------

# A line that looks like a Markdown table row: starts/ends with `|` and has
# at least one inner `|`.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*\|\s*$")

# Separator row: only dashes, colons, pipes, and whitespace.
_SEP_ROW_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _is_table_line(line: str) -> bool:
    """Return True if *line* looks like a Markdown table row."""
    return bool(_TABLE_ROW_RE.match(line))


def _is_separator_line(line: str) -> bool:
    """Return True if *line* is a table separator row (e.g. |---|---|)."""
    if not _SEP_ROW_RE.match(line):
        return False
    # Must contain at least one dash
    return "-" in line


def _split_table_row(line: str) -> List[str]:
    """Split a ``| a | b | c |`` line into ``['a', 'b', 'c']``.

    Leading/trailing pipes are stripped; inner cell text is stripped of
    surrounding whitespace but internal content is preserved.
    """
    stripped = line.strip()
    # Remove leading and trailing pipe
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _detect_alignment(sep_cells: List[str]) -> List[str]:
    """Infer alignment from separator-row cells (e.g. ':---:', '---:', ':---')."""
    alignments = []
    for cell in sep_cells:
        cell = cell.strip()
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            alignments.append("center")
        elif right:
            alignments.append("right")
        else:
            alignments.append("left")
    return alignments


def _format_table_block(table_lines: List[str]) -> List[str]:
    """Re-pad a contiguous block of Markdown table lines.

    Parameters
    ----------
    table_lines : list of str
        Raw table lines (header, separator, data rows).  Must have at
        least 2 lines (header + separator).

    Returns
    -------
    list of str
        The same table with every column padded to uniform width.
    """
    if len(table_lines) < 2:
        return table_lines

    # Parse all rows into cell lists
    parsed_rows = [_split_table_row(line) for line in table_lines]

    # Find the separator row (first row that matches the separator pattern)
    sep_idx = None
    for i, line in enumerate(table_lines):
        if _is_separator_line(line):
            sep_idx = i
            break
    if sep_idx is None:
        # No separator found — not a valid table, return as-is
        return table_lines

    # Detect alignment from the separator row
    alignments = _detect_alignment(parsed_rows[sep_idx])

    # Determine number of columns from the header (row before separator)
    header_idx = sep_idx - 1 if sep_idx > 0 else 0
    n_cols = len(parsed_rows[header_idx])

    # Extend alignments if needed
    while len(alignments) < n_cols:
        alignments.append("left")

    # Compute max width per column across all non-separator rows
    col_widths = [3] * n_cols  # minimum width for separator dashes
    for i, cells in enumerate(parsed_rows):
        if i == sep_idx:
            continue  # skip separator row for width calculation
        for j, cell in enumerate(cells):
            if j < n_cols:
                col_widths[j] = max(col_widths[j], len(cell))

    # Rebuild each row
    result = []
    for i, cells in enumerate(parsed_rows):
        if i == sep_idx:
            # Rebuild separator with correct widths and alignment
            sep_parts = []
            for j in range(n_cols):
                w = col_widths[j]
                align = alignments[j] if j < len(alignments) else "left"
                if align == "right":
                    sep_parts.append("-" * (w - 1) + ":")
                elif align == "center":
                    sep_parts.append(":" + "-" * (w - 2) + ":")
                else:
                    sep_parts.append("-" * w)
            result.append("| " + " | ".join(sep_parts) + " |")
        else:
            # Pad data/header cells
            padded = []
            for j in range(n_cols):
                cell = cells[j] if j < len(cells) else ""
                padded.append(cell.ljust(col_widths[j]))
            result.append("| " + " | ".join(padded) + " |")

    return result


def pad_md_tables(text: str) -> str:
    """Find and pad all Markdown tables in *text*.

    Scans the text line by line, identifies contiguous blocks of table
    rows (lines matching ``| ... | ... |``), and reformats each block
    so every column is padded to uniform width.  Non-table content is
    left untouched.

    Parameters
    ----------
    text : str
        Full Markdown document content.

    Returns
    -------
    str
        The document with all tables reformatted.
    """
    lines = text.split("\n")
    result: List[str] = []
    table_buffer: List[str] = []
    in_code_block = False

    def flush_table():
        """Process any accumulated table lines and append to result."""
        if not table_buffer:
            return
        # Only reformat if the buffer contains a separator row (= valid table)
        has_separator = any(_is_separator_line(line) for line in table_buffer)
        if has_separator and len(table_buffer) >= 2:
            result.extend(_format_table_block(table_buffer))
        else:
            result.extend(table_buffer)
        table_buffer.clear()

    for line in lines:
        # Track fenced code blocks (``` or ~~~) — skip tables inside them
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            flush_table()
            result.append(line)
            continue

        if in_code_block:
            flush_table()
            result.append(line)
            continue

        if _is_table_line(line):
            table_buffer.append(line)
        else:
            flush_table()
            result.append(line)

    # Handle table at the end of file
    flush_table()

    return "\n".join(result)
