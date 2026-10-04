"""Portable report and notebook helpers copied into each independent repository."""

from pathlib import Path
import json
from xml.sax.saxutils import escape


def write_report(path, title, sections, figures=()):
    """Render a paginated evidence report from paragraphs and selected figures."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.units import inch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#12344a")
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].allowWidows = 0
    styles["Heading2"].textColor = colors.HexColor("#007f82")
    styles["Heading2"].keepWithNext = True
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 16)]
    for heading, paragraphs in sections:
        story.append(Paragraph(escape(str(heading)), styles["Heading2"]))
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        for paragraph in paragraphs:
            story.append(
                Paragraph(
                    escape(str(paragraph)).replace("\n", "<br/>"), styles["BodyText"]
                )
            )
            story.append(Spacer(1, 8))
    for figure in figures:
        if Path(figure).exists():
            img = Image(str(figure))
            ratio = min(6.4 * inch / img.imageWidth, 4.5 * inch / img.imageHeight)
            img.drawWidth = img.imageWidth * ratio
            img.drawHeight = img.imageHeight * ratio
            story += [Spacer(1, 12), img]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526879"))
        canvas.drawString(
            42,
            25,
            "Industrial Engineering + Operations + Data Analytics | Portfolio study",
        )
        canvas.drawRightString(570, 25, str(doc.page))

    SimpleDocTemplate(
        str(path),
        pagesize=(612, 792),
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def write_notebook(path, title, narrative, code_cells):
    """Create a small modular notebook with a relative repository-root bootstrap."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    bootstrap = "from pathlib import Path\nimport os, sys\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nos.chdir(ROOT)\nif str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
    nb.cells = [
        nbformat.v4.new_markdown_cell("# " + title + "\n\n" + narrative),
        nbformat.v4.new_code_cell(bootstrap),
    ]
    for cell in code_cells:
        nb.cells.append(nbformat.v4.new_code_cell(cell))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
