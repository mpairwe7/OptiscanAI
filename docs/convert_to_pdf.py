#!/usr/bin/env python3
"""Convert markdown documents to styled PDFs using weasyprint."""

import markdown
from weasyprint import HTML
from pathlib import Path

CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {
        content: "RetinalAI — Confidential";
        font-size: 8pt;
        color: #999;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #999;
    }
}

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a2e;
    max-width: 100%;
}

h1 {
    font-size: 22pt;
    color: #0d1b2a;
    border-bottom: 3px solid #1b4965;
    padding-bottom: 8px;
    margin-top: 30px;
    page-break-after: avoid;
}

h2 {
    font-size: 16pt;
    color: #1b4965;
    border-bottom: 1.5px solid #bee9e8;
    padding-bottom: 5px;
    margin-top: 25px;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    color: #2b6777;
    margin-top: 18px;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #52ab98;
    margin-top: 14px;
    page-break-after: avoid;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th {
    background-color: #1b4965;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 6px 10px;
    border-bottom: 1px solid #e0e0e0;
}

tr:nth-child(even) {
    background-color: #f8fafe;
}

code {
    background-color: #f0f4f8;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
    color: #c7254e;
}

pre {
    background-color: #0d1b2a;
    color: #bee9e8;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    color: #bee9e8;
    padding: 0;
}

blockquote {
    border-left: 4px solid #1b4965;
    margin: 12px 0;
    padding: 10px 18px;
    background-color: #f0f7fa;
    color: #2b6777;
    font-style: italic;
    page-break-inside: avoid;
}

strong {
    color: #0d1b2a;
}

hr {
    border: none;
    border-top: 2px solid #bee9e8;
    margin: 25px 0;
}

ul, ol {
    margin: 8px 0;
    padding-left: 22px;
}

li {
    margin-bottom: 4px;
}

a {
    color: #1b4965;
    text-decoration: none;
}

.cover-info {
    text-align: center;
    margin-top: 40px;
    font-size: 10pt;
    color: #666;
}
"""

FILES = [
    ("video-script.md", "RetinalAI-Video-Script.pdf"),
    ("13-commercialization-strategy.md", "RetinalAI-Commercialization-Strategy.pdf"),
    ("14-hackathon-pitch-guide.md", "RetinalAI-Hackathon-Pitch-Guide.pdf"),
]

docs_dir = Path(__file__).parent

for md_file, pdf_file in FILES:
    md_path = docs_dir / md_file
    pdf_path = docs_dir / pdf_file

    print(f"Converting {md_file} -> {pdf_file} ...")

    md_text = md_path.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(str(pdf_path))
    print(f"  -> {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")

print("\nAll 3 PDFs generated successfully!")
