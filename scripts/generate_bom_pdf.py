#!/usr/bin/env python3
"""Generate Bill of Materials PDF for project submission."""

from fpdf import FPDF


class BOMReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(
            0,
            8,
            "RetinalAI Clinical Screening Platform - MLOps Capstone Project",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 51, 102)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(2)

    def add_table(self, headers, data, col_widths):
        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(0)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(230, 240, 250)
            else:
                self.set_fill_color(255)
            for i, cell in enumerate(row):
                align = (
                    "R"
                    if i >= len(row) - 2 and row[i].replace(",", "").replace(" ", "").isdigit()
                    else "L"
                )
                if i == 0:
                    align = "C"
                self.cell(col_widths[i], 7, cell, border=1, fill=True, align=align)
            self.ln()
            fill = not fill


def main():
    pdf = BOMReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Project Title ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, "BILL OF MATERIALS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60)
    pdf.cell(
        0, 7, "RetinalAI Clinical Screening Platform", align="C", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(0, 7, "MLOps Capstone Project", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Group Members ──
    pdf.section_title("Group Members")
    members = [
        ["1", "Mpairwe Lauben", "0773 336896"],
        ["2", "Nankya Shadia", "0708 626678"],
        ["3", "Yapyeko Rebecca", "0786 614652"],
    ]
    pdf.add_table(
        ["#", "Name", "Contact"],
        members,
        [15, 90, 85],
    )
    pdf.ln(4)

    # ── Document Info ──
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0)
    info = [
        ("Date:", "5th May 2026"),
        ("Currency:", "Uganda Shillings (UGX)"),
    ]
    for label, value in info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(30, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── 1. Hardware ──
    pdf.section_title("1. Hardware")
    hw_headers = ["#", "Item", "Specification", "Qty", "Unit Cost (UGX)", "Total (UGX)"]
    hw_widths = [10, 25, 60, 15, 40, 40]
    hw_data = [
        ["1", "Laptop", "HP Brand (development & testing)", "1", "1,200,000", "1,200,000"],
        ["2", "RAM", "8 GB DDR4 (per team member)", "3", "90,000", "270,000"],
    ]
    pdf.add_table(hw_headers, hw_data, hw_widths)

    # Subtotal
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(150, 8, "Hardware Subtotal:", align="R")
    pdf.cell(40, 8, "1,470,000", border=1, align="R")
    pdf.ln(8)

    # ── 2. Services ──
    pdf.section_title("2. Services & Subscriptions")
    svc_headers = ["#", "Item", "Description", "Qty", "Unit Cost (UGX)", "Total (UGX)"]
    svc_widths = [10, 30, 55, 15, 40, 40]
    svc_data = [
        ["3", "Internet", "Monthly broadband per member", "3", "50,000", "150,000"],
        ["4", "GCP Hosting", "Cloud training, API serving, storage", "1", "120,000", "120,000"],
    ]
    pdf.add_table(svc_headers, svc_data, svc_widths)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(150, 8, "Services Subtotal:", align="R")
    pdf.cell(40, 8, "270,000", border=1, align="R")
    pdf.ln(8)

    # ── 3. Contingency ──
    pdf.section_title("3. Contingency")
    cont_headers = ["#", "Item", "Description", "Total (UGX)"]
    cont_widths = [10, 35, 105, 40]
    cont_data = [
        ["5", "Contingency", "Unforeseen expenses (supplies, extra compute, transport)", "60,000"],
    ]
    pdf.add_table(cont_headers, cont_data, cont_widths)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(150, 8, "Contingency Subtotal:", align="R")
    pdf.cell(40, 8, "60,000", border=1, align="R")
    pdf.ln(10)

    # ── Budget Summary ──
    pdf.section_title("Budget Summary")
    sum_headers = ["Category", "Amount (UGX)"]
    sum_widths = [120, 70]
    sum_data = [
        ["Hardware", "1,470,000"],
        ["Services & Subscriptions", "270,000"],
        ["Contingency", "60,000"],
    ]
    pdf.add_table(sum_headers, sum_data, sum_widths)

    # Grand total row
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255)
    pdf.cell(120, 9, "GRAND TOTAL", border=1, fill=True, align="C")
    pdf.cell(70, 9, "UGX 1,800,000", border=1, fill=True, align="R")
    pdf.set_text_color(0)
    pdf.ln(14)

    # ── Signatures ──
    pdf.section_title("Approvals")
    pdf.set_font("Helvetica", "", 10)

    sigs = ["Prepared by", "Supervisor"]
    for sig in sigs:
        pdf.cell(30, 7, f"{sig}:")
        pdf.cell(80, 7, "_" * 40)
        pdf.cell(15, 7, "Date:")
        pdf.cell(50, 7, "_" * 20, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # ── Output ──
    output_path = "docs/Bill-of-Materials.pdf"
    pdf.output(output_path)
    print(f"PDF saved to {output_path}")


if __name__ == "__main__":
    main()
