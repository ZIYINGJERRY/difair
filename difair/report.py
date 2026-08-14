"""Self-contained HTML audit reports.

Bundles the item-level, model-level and stage-attribution results into a single
standalone file with no external assets, so it can be attached to a technical
documentation pack. The section headings follow the evidence categories that
Article 15 of Regulation (EU) 2024/1689 expects for high-risk systems in
education, but the report is a documentation aid and not a conformity
assessment.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

__all__ = ["audit_report"]

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
margin:0;padding:2rem;color:#1a1a1a;background:#fafafa;line-height:1.55}
.wrap{max-width:1000px;margin:0 auto;background:#fff;padding:2.5rem;
border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{font-size:1.6rem;color:#1f3864;margin:0 0 .3rem}
h2{font-size:1.1rem;color:#1f3864;margin:2rem 0 .6rem;
border-bottom:2px solid #1f3864;padding-bottom:.3rem}
.meta{color:#666;font-size:.85rem;margin-bottom:1.5rem}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0 1rem}
th{background:#1f3864;color:#fff;text-align:left;padding:.5rem .6rem;font-weight:600}
td{padding:.45rem .6rem;border-bottom:1px solid #e4e4e4}
tr:nth-child(even) td{background:#f6f8fb}
.flag-C{color:#a03030;font-weight:600}.flag-B{color:#b8860b;font-weight:600}
.flag-A{color:#2e6b4f}
.note{background:#eef2f8;border-left:4px solid #1f3864;padding:.7rem 1rem;
margin:1rem 0;font-size:.88rem}
.empty{color:#888;font-style:italic}
"""


def _table(df, max_rows=200):
    if df is None or len(df) == 0:
        return '<p class="empty">No results in this section.</p>'
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = ""
    for _, r in d.iterrows():
        cells = ""
        for c in d.columns:
            val = html.escape(str(r[c]))
            cls = f' class="flag-{val}"' if c == "ets_class" and val in "ABC" and len(val) == 1 else ""
            cells += f"<td{cls}>{val}</td>"
        body += f"<tr>{cells}</tr>"
    more = (
        f'<p class="empty">Showing {max_rows} of {len(df)} rows.</p>'
        if len(df) > max_rows else ""
    )
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def audit_report(
    path,
    dif_result=None,
    fairness_table=None,
    attribution=None,
    title="DIFair audit report",
    context=None,
):
    """Write a standalone HTML audit report.

    Parameters
    ----------
    path : str
        Destination ``.html`` file.
    dif_result : DIFResult, optional
        Output of :func:`difair.dif.detect_dif`.
    fairness_table : DataFrame, optional
        Output of :func:`difair.fairness.fairness_report`.
    attribution : AttributionResult, optional
        Output of :func:`difair.pipeline.attribute_stages`.
    title : str
        Report heading.
    context : dict, optional
        Free-form key/value pairs describing the audited system; rendered in
        the header so the report is self-describing.

    Returns
    -------
    str
        The path written.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ctx = ""
    if context:
        ctx = "<br>".join(
            f"<strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}"
            for k, v in context.items()
        )

    parts = [f'<div class="wrap"><h1>{html.escape(title)}</h1>',
             f'<div class="meta">Generated {stamp}<br>{ctx}</div>']

    parts.append("<h2>1. Item-level differential item functioning</h2>")
    if dif_result is not None:
        n_flag = len(dif_result.flagged)
        parts.append(
            f'<div class="note">{n_flag} of {len(dif_result.table)} items reached '
            f"ETS class B or C against the focal group "
            f"({html.escape(str(dif_result.focal_label))}). "
            f'Matching score {"was" if dif_result.purified else "was not"} purified.</div>'
        )
        parts.append(_table(dif_result.table))
    else:
        parts.append('<p class="empty">Not supplied.</p>')

    parts.append("<h2>2. Model-level fairness</h2>")
    if fairness_table is not None:
        t = fairness_table.copy()
        if "detail" in t.columns:
            t = t.drop(columns=["detail"])
        parts.append(
            '<div class="note">Values are focal minus reference; negative '
            "numbers indicate the focal group is disadvantaged.</div>"
        )
        parts.append(_table(t))
    else:
        parts.append('<p class="empty">Not supplied.</p>')

    parts.append("<h2>3. Pipeline-stage attribution</h2>")
    if attribution is not None:
        parts.append(
            f'<div class="note">Disparity measured by '
            f"<code>{html.escape(attribution.metric)}</code>: baseline "
            f"{attribution.baseline_gap:.4f}, residual after all mitigations "
            f"{attribution.residual_gap:.4f}. Shapley values give each stage's "
            f"average marginal contribution across all orderings.</div>"
        )
        parts.append(_table(attribution.summary()))
    else:
        parts.append('<p class="empty">Not supplied.</p>')

    parts.append(
        '<h2>4. Interpretation notes</h2><div class="note">Statistical DIF is '
        "not by itself evidence of unfairness: an item may function "
        "differently for substantive reasons that content review can justify. "
        "These results are inputs to expert judgement, not a verdict. "
        "Attribution reflects the mitigations implemented here and the causal "
        "structure they assume.</div></div>"
    )

    doc = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(parts)}</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
