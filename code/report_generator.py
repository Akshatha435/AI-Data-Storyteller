# report_generator.py
"""
create_report(output_path, executive_text, images_info=None, image_paths=None)

Creates a colorful, two-visuals-per-page PDF.
- Preferred: images_info is a list of dicts:
    {"image": path, "title": str, "interpretation": str}
- Or pass a simple list of file paths via image_paths (will be converted).

Dependencies: reportlab, Pillow
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import os
from typing import List, Optional, Dict

PAGE_WIDTH, PAGE_HEIGHT = A4

# Palette (match app)
PALETTE = {
    "primary": "#283593",
    "accent": "#00897B",
    "highlight": "#FF7043",
    "text": "#37474F",
    "bg": "#F7FAFC"
}


def _scale_to_fit(max_w, max_h, img_path):
    """Return (w, h) scaled to fit max_w x max_h preserving aspect ratio."""
    try:
        with PILImage.open(img_path) as im:
            iw, ih = im.size
            # convert mm to points if needed: here we keep same units reportlab uses (points)
            ratio = min(max_w / iw, max_h / ih)
            return (iw * ratio, ih * ratio)
    except Exception:
        return (max_w, max_h)


def _header_footer(canvas, doc):
    canvas.saveState()
    # top stripe
    canvas.setFillColor(colors.HexColor(PALETTE["primary"]))
    canvas.rect(0, PAGE_HEIGHT - 32, PAGE_WIDTH, 32, stroke=0, fill=1)
    # footer stripe
    canvas.setFillColor(colors.HexColor(PALETTE["bg"]))
    canvas.rect(0, 0, PAGE_WIDTH, 20, stroke=0, fill=1)
    # page number
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(PALETTE["text"]))
    canvas.drawRightString(PAGE_WIDTH - 20, 8, f"Page {doc.page}")
    canvas.restoreState()


def _ensure_output_dir(path: str):
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def create_report(output_path: str,
                  executive_text: str,
                  images_info: Optional[List[Dict]] = None,
                  image_paths: Optional[List[str]] = None):
    """
    Create the PDF report.

    Parameters
    ----------
    output_path : str
        Path to write the PDF (e.g. 'report/executive_summary.pdf').
    executive_text : str
        The combined observations / executive summary text (can include newlines).
    images_info : list of dict, optional
        Preferred format: [{"image": path, "title": str, "interpretation": str}, ...]
    image_paths : list of str, optional
        If provided, will be converted into images_info with empty title/interpretation.
    """
    images_info = images_info or []

    # If caller passed a plain list of paths, convert to images_info structure
    if (not images_info) and image_paths:
        for p in image_paths:
            images_info.append({"image": p, "title": "", "interpretation": ""})

    # final safe list
    images_info = images_info or []

    # ensure output directory exists
    _ensure_output_dir(output_path)

    # Document setup
    doc = SimpleDocTemplate(output_path,
                            pagesize=A4,
                            leftMargin=20, rightMargin=20,
                            topMargin=40, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontSize=22,
        textColor=colors.HexColor(PALETTE["primary"])
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=11,
        textColor=colors.HexColor(PALETTE["accent"])
    )
    normal = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        alignment=TA_LEFT
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontSize=10,
        textColor=colors.HexColor(PALETTE["accent"])
    )
    interp_style = ParagraphStyle(
        "Interp",
        parent=styles["BodyText"],
        fontSize=9.5,
        textColor=colors.HexColor(PALETTE["text"])
    )

    # Title page with optional logo (look for outputs/logo.png relative to output_path)
    logo_candidate = os.path.join(os.path.dirname(output_path), "..", "outputs", "logo.png")
    story.append(Spacer(1, 10))
    if os.path.exists(logo_candidate):
        try:
            # reportlab Image expects width/height in points; we scale using pixels ratio
            w, h = _scale_to_fit(120 * mm, 40 * mm, logo_candidate)
            story.append(Image(logo_candidate, width=w, height=h))
            story.append(Spacer(1, 8))
        except Exception:
            pass

    story.append(Paragraph("Executive Report — AI Data Storyteller", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Automated EDA, Visualizations & Interpretations", subtitle_style))
    story.append(Spacer(1, 12))

    # Executive text lines
    for line in str(executive_text).split("\n"):
        story.append(Paragraph(line.strip(), normal))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # Two images per page layout
    per_page = 2
    # compute available width in points and divide columns
    max_w = (PAGE_WIDTH - 60) / 2
    max_h = 95 * mm

    for i in range(0, len(images_info), per_page):
        block = images_info[i:i + per_page]
        row_cells = []
        titles = []
        interps = []
        for info in block:
            img = info.get("image")
            if img and os.path.exists(img):
                try:
                    w, h = _scale_to_fit(max_w, max_h, img)
                    # reportlab.Image will scale image; pass computed width/height
                    row_cells.append(Image(img, width=w, height=h))
                except Exception:
                    row_cells.append(Paragraph("Image could not be loaded", normal))
            else:
                row_cells.append(Paragraph("Image not available", normal))
            titles.append(info.get("title", ""))
            interps.append(info.get("interpretation", ""))

        # make sure two columns exist
        if len(row_cells) == 1:
            row_cells.append(Paragraph("", normal))
            titles.append("")
            interps.append("")

        # place images side-by-side
        table = Table([row_cells], colWidths=[max_w, max_w], hAlign="CENTER")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(table)
        story.append(Spacer(1, 6))

        # titles under images
        cap_cells = [Paragraph(titles[0], caption_style), Paragraph(titles[1], caption_style)]
        cap_table = Table([cap_cells], colWidths=[max_w, max_w], hAlign="CENTER")
        cap_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(cap_table)
        story.append(Spacer(1, 4))

        # interpretations under each image
        interp_cells = [Paragraph(interps[0], interp_style), Paragraph(interps[1], interp_style)]
        interp_table = Table([interp_cells], colWidths=[max_w, max_w], hAlign="CENTER")
        interp_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(interp_table)

        story.append(PageBreak())

    # Final notes
    story.append(Paragraph("Notes & Next Steps", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated from the AI Data Storyteller dashboard. Use the visual interpretations as a starting point for modeling and deeper analysis.",
        normal
    ))

    # Build PDF with header/footer
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
