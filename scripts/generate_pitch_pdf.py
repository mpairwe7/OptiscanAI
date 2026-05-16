#!/usr/bin/env python3
"""Generate a styled PDF from the hackathon pitch guide markdown."""
from pathlib import Path

import markdown
from weasyprint import HTML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = PROJECT_ROOT / "docs" / "14-hackathon-pitch-guide.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "RetinalAI-Pitch-Guide-2026.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {
        content: "RetinalAI Clinical Screening Platform — Confidential";
        font-size: 8pt;
        color: #999;
    }
    @bottom-right {
        content: counter(page);
        font-size: 8pt;
        color: #999;
    }
}
body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
}
h1 {
    font-size: 22pt;
    color: #0d47a1;
    border-bottom: 3px solid #0d47a1;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-before: avoid;
}
h2 {
    font-size: 15pt;
    color: #1565c0;
    border-bottom: 1.5px solid #e0e0e0;
    padding-bottom: 5px;
    margin-top: 28px;
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    color: #1976d2;
    margin-top: 18px;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    color: #2196f3;
    margin-top: 14px;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th {
    background: #0d47a1;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 10px;
    border-bottom: 1px solid #e0e0e0;
}
tr:nth-child(even) td {
    background: #f5f8ff;
}
blockquote {
    border-left: 4px solid #1976d2;
    margin: 10px 0;
    padding: 8px 16px;
    background: #f0f4ff;
    font-style: italic;
    color: #333;
    page-break-inside: avoid;
}
blockquote p {
    margin: 4px 0;
}
code {
    background: #f5f5f5;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9pt;
    font-family: 'Consolas', 'Monaco', monospace;
}
pre {
    background: #263238;
    color: #e0e0e0;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
    color: inherit;
}
hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 24px 0;
}
strong {
    color: #0d47a1;
}
em {
    color: #555;
}
ul, ol {
    margin: 8px 0;
    padding-left: 22px;
}
li {
    margin-bottom: 4px;
}
p {
    margin: 8px 0;
}
/* Cover styling for the first h1 */
h1:first-of-type {
    font-size: 28pt;
    text-align: center;
    border-bottom: 4px solid #0d47a1;
    padding-bottom: 16px;
    margin-bottom: 20px;
}
"""


def main():
    md_text = INPUT_MD.read_text(encoding="utf-8")

    extensions = ["tables", "fenced_code", "codehilite", "toc", "smarty"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

    HTML(string=full_html).write_pdf(str(OUTPUT_PDF))
    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"PDF generated: {OUTPUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
