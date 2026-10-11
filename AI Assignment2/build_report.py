"""Generate the ABNN assignment report as a Word document and PDF.

Experiment values come from the persisted quick/full JSON artifacts. Figures
must match their provenance manifest before they are embedded. Microsoft Word
is used for PDF conversion because the submission requires a faithful Word/PDF
pair on Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

from code.make_figures import OUTPUT_NAMES, figures_are_current


ROOT = Path(__file__).resolve().parent
SOURCE_URL = (
    "https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2"
)
STUDENT_NAME = "HOI KAI CHENG"
STUDENT_ID = "UC325381"

REPORT_DIR = ROOT / "report"
DOCX_NAME = "AI_Assignment2_ABNN_Report.docx"
PDF_NAME = "AI_Assignment2_ABNN_Report.pdf"
QUICK_DOCX_NAME = "AI_Assignment2_ABNN_Report_QUICK.docx"
QUICK_PDF_NAME = "AI_Assignment2_ABNN_Report_QUICK.pdf"
SOURCE_EXCERPT_FILES = (ROOT / "code" / "abnn.py",)
FONT = "Aptos"
MONO_FONT = "Consolas"
ACCENT = RGBColor(31, 78, 121)
MUTED = RGBColor(82, 99, 119)
QUICK_RED = RGBColor(156, 31, 31)
METHOD_LABELS = {
    "map": "MAP",
    "abnn_single": "ABNN (single member)",
    "abnn_full": "ABNN (all available members)",
}
DATASET_LABELS = {"svhn": "SVHN", "cifar100": "CIFAR-100"}
FIGURE_CAPTIONS = (
    "ABNN architecture and inference pipeline. The deterministic ResNet-20 "
    "checkpoint is converted by replacing BatchNorm with Bayesian "
    "normalization, then stochastic predictions are averaged.",
    "Training curves loaded from the persisted MAP and ABNN history JSON "
    "artifacts.",
    "Reliability diagram for MAP, single-member ABNN, and the complete "
    "available-member path.",
    "OOD entropy distributions for CIFAR-10 (in-distribution), SVHN, and "
    "CIFAR-100.",
    "Metrics comparison for classification, calibration, and OOD detection.",
    "Uncertainty examples ordered by ABNN predictive entropy, from the least "
    "to the most uncertain saved CIFAR-10 examples.",
)
ORIGINAL_PROMPT = (
    r"d:\UM\Year4\CISC3024\AI Assignment2\AIassign2.pdf"
    "\nYour AI assignment report in Word/PDF format should contain the "
    "following details (1) how you ask AI tools to find the algorithm, "
    "(2) algorithm description, (3) how AI implements the algorithm, "
    "(4) experiment settings and results, (5) what you have learnt from "
    "this AI assignment, and (6) a webpage link of your source codes."
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required report artifact not found: {path}. Run "
            f"'python -m code.run_all "
            f"{'--quick ' if '_quick' in path.name else ''}--skip-report' "
            "before building the report."
        )
    try:
        with path.open(encoding="utf-8") as artifact:
            value = json.load(artifact)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid report JSON {path}: line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid report JSON {path}: expected an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(quick: bool) -> tuple[Path, Path, Path]:
    suffix = "_quick" if quick else ""
    return (
        ROOT / "results" / f"evaluation{suffix}.json",
        ROOT / "results" / f"map_history{suffix}.json",
        ROOT / "results" / f"abnn_history{suffix}.json",
    )


def _report_source_paths(quick: bool) -> tuple[Path, ...]:
    return (
        *_artifact_paths(quick),
        ROOT / "figures" / "manifest.json",
        *(ROOT / "figures" / name for name in OUTPUT_NAMES),
        *SOURCE_EXCERPT_FILES,
    )


def _source_snapshot(paths) -> dict[Path, str]:
    snapshot = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Required report source not found: {path}")
        snapshot[path] = _sha256(path)
    return snapshot


def _assert_source_snapshot(
    snapshot: dict[Path, str], phase: str
) -> None:
    try:
        current = {path: _sha256(path) for path in snapshot}
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(
            f"Report source artifacts changed {phase}; no report was published."
        ) from error
    if current != snapshot:
        changed = [
            path.name
            for path, digest in snapshot.items()
            if current.get(path) != digest
        ]
        raise RuntimeError(
            f"Report source artifacts changed {phase}: "
            f"{', '.join(changed)}. No report was published."
        )


def _load_report_artifacts(quick: bool) -> tuple[dict, dict, dict, dict]:
    if not isinstance(quick, bool):
        raise TypeError("quick must be a bool")
    evaluation_path, map_path, abnn_path = _artifact_paths(quick)
    evaluation = _read_json(evaluation_path)
    map_history = _read_json(map_path)
    abnn_history = _read_json(abnn_path)
    expected_mode = "quick" if quick else "full"
    for path, payload in (
        (evaluation_path, evaluation),
        (map_path, map_history),
        (abnn_path, abnn_history),
    ):
        run = payload.get("run")
        if (
            not isinstance(run, dict)
            or run.get("mode") != expected_mode
            or run.get("quick") is not quick
        ):
            raise ValueError(
                f"{path.name} does not contain {expected_mode}-mode metadata"
            )
    if not figures_are_current(
        ROOT / "results", ROOT / "figures", quick=quick
    ):
        raise RuntimeError(
            "Report figures are missing, invalid, stale, or from the wrong "
            f"mode. Run 'python -m code.run_all "
            f"{'--quick ' if quick else ''}--skip-report' to regenerate the "
            "matching JSON and all six figures."
        )
    manifest = _read_json(ROOT / "figures" / "manifest.json")
    return evaluation, map_history, abnn_history, manifest


def _source_excerpt(
    relative_path: str,
    markers: tuple[str, ...],
    *,
    context: int = 1,
) -> str:
    """Read selected, exact source lines and mark omitted spans."""
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Source excerpt file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    selected: set[int] = set()
    for marker in markers:
        matches = [index for index, line in enumerate(lines) if marker in line]
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected one source line containing {marker!r}, "
                f"found {len(matches)}"
            )
        index = matches[0]
        selected.update(
            range(max(0, index - context), min(len(lines), index + context + 1))
        )

    output: list[str] = []
    previous = None
    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            output.append("    # ...")
        output.append(lines[index])
        previous = index
    return "\n".join(output)


def _set_run_font(run, *, name: str = FONT, size: float = 10.5) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def _add_text(
    paragraph,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor | None = None,
    size: float = 10.5,
    font: str = FONT,
):
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    _set_run_font(run, name=font, size=size)
    if color is not None:
        run.font.color.rgb = color
    return run


def _add_paragraph(
    document: Document,
    text: str = "",
    *,
    style: str | None = None,
    bold_prefix: str | None = None,
    align=None,
    before: float = 0,
    after: float = 6,
    keep_with_next: bool = False,
):
    paragraph = document.add_paragraph(style=style)
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.12
    paragraph.paragraph_format.keep_with_next = keep_with_next
    if bold_prefix and text.startswith(bold_prefix):
        _add_text(paragraph, bold_prefix, bold=True)
        _add_text(paragraph, text[len(bold_prefix) :])
    else:
        _add_text(paragraph, text)
    return paragraph


def _add_bullet(document: Document, text: str) -> None:
    paragraph = _add_paragraph(
        document, text, style="List Bullet", after=3
    )
    paragraph.paragraph_format.left_indent = Cm(0.65)
    paragraph.paragraph_format.first_line_indent = Cm(-0.35)


def _add_numbered(document: Document, text: str) -> None:
    paragraph = _add_paragraph(
        document, text, style="List Number", after=3
    )
    paragraph.paragraph_format.left_indent = Cm(0.75)
    paragraph.paragraph_format.first_line_indent = Cm(-0.4)


def _shade_cell(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    properties.append(
        parse_xml(
            f'<w:shd {nsdecls("w")} w:fill="{fill}" w:val="clear"/>'
        )
    )


def _repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def _add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    _add_text(paragraph, text, italic=True, size=9.5, color=MUTED)


def _add_table(
    document: Document,
    caption: str,
    headers: tuple[str, ...],
    rows,
    *,
    widths: tuple[float, ...] | None = None,
):
    _add_caption(document, caption)
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    _repeat_table_header(header)
    for index, label in enumerate(headers):
        cell = header.cells[index]
        cell.text = ""
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _shade_cell(cell, "D9EAF7")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_text(paragraph, label, bold=True, size=9)

    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for column_index, value in enumerate(values):
            cell = cells[column_index]
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2:
                _shade_cell(cell, "F4F7FA")
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            _add_text(paragraph, str(value), size=8.7)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    document.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def _add_code_block(
    document: Document, source_name: str, code: str
) -> None:
    label = _add_paragraph(
        document,
        f"Excerpt read at build time from {source_name}:",
        after=2,
        keep_with_next=True,
    )
    label.runs[0].italic = True
    for index, line in enumerate(code.splitlines()):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.5)
        paragraph.paragraph_format.right_indent = Cm(0.3)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(
            5 if index == len(code.splitlines()) - 1 else 0
        )
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.keep_together = True
        shade = parse_xml(
            f'<w:shd {nsdecls("w")} w:fill="F2F4F7" w:val="clear"/>'
        )
        paragraph._p.get_or_add_pPr().append(shade)
        _add_text(
            paragraph,
            line or " ",
            size=8.2,
            font=MONO_FONT,
        )


def _add_equation(
    document: Document, expression: str, number: int
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.keep_together = True
    _add_text(
        paragraph,
        expression,
        italic=True,
        size=11,
        font="Cambria Math",
    )
    _add_text(paragraph, f"     ({number})", size=9.5)


def _add_hyperlink(paragraph, url: str, text: str | None = None) -> None:
    relationship_id = paragraph.part.relate_to(
        url,
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/hyperlink"
        ),
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT)
    fonts.set(qn("w:hAnsi"), FONT)
    properties.append(fonts)
    run.append(properties)
    text_node = OxmlElement("w:t")
    text_node.text = text or url
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _add_page_number(paragraph) -> None:
    _add_text(paragraph, "Page ", size=9)
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, end))


def _configure_document(document: Document, quick: bool) -> None:
    section = document.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.1)
    section.right_margin = Cm(2.1)

    normal = document.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    for level, size in ((1, 15), (2, 12), (3, 11)):
        style = document.styles[f"Heading {level}"]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = ACCENT
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        style.paragraph_format.space_after = Pt(5)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_page_number(footer)
    if quick:
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_text(
            header,
            "SMOKE-TEST DATA — NOT THE FINAL SUBMISSION",
            bold=True,
            color=QUICK_RED,
            size=9,
        )


def _heading(document: Document, text: str, level: int = 1) -> None:
    document.add_heading(text, level=level)


def _page_break(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def _add_figure(
    document: Document,
    number: int,
    name: str,
    caption: str,
    *,
    quick: bool,
) -> None:
    path = ROOT / "figures" / name
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(2)
    width = 6.25 if name != "uncertainty_examples.png" else 6.35
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    mode_note = " (quick smoke-test artifact)" if quick else ""
    _add_caption(
        document,
        f"Figure {number}. {caption}{mode_note} "
        f"[artifact: {name}]",
    )


def _fmt(value) -> str:
    return f"{float(value):.4f}"


def _elapsed(seconds) -> str:
    seconds = float(seconds)
    return f"{seconds:.2f} s"


def _title_page(document: Document, quick: bool) -> None:
    for _ in range(3):
        document.add_paragraph()
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_text(
        paragraph,
        "CISC3024 Pattern Recognition",
        bold=True,
        color=ACCENT,
        size=16,
    )
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_text(
        paragraph,
        "AI Assignment #2",
        bold=True,
        size=22,
    )
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_text(
        paragraph,
        "Adaptable Bayesian Neural Networks for\n"
        "Classification and Uncertainty Estimation",
        bold=True,
        color=ACCENT,
        size=17,
    )
    _add_paragraph(
        document,
        "A CPU-scale, data-driven study of “Make Me a BNN” (CVPR 2024)",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        before=6,
        after=20,
    )
    if quick:
        banner = _add_paragraph(
            document,
            "SMOKE-TEST DATA — NOT THE FINAL SUBMISSION",
            align=WD_ALIGN_PARAGRAPH.CENTER,
            before=5,
            after=18,
        )
        banner.runs[0].bold = True
        banner.runs[0].font.color.rgb = QUICK_RED
        banner.runs[0].font.size = Pt(13)
        _add_paragraph(
            document,
            "This document validates the reporting pipeline using the reduced "
            "quick workflow. Task 11 will regenerate the same report from the "
            "full experiment artifacts.",
            align=WD_ALIGN_PARAGRAPH.CENTER,
            after=20,
        )
    for label, value in (
        ("Student", STUDENT_NAME),
        ("Student ID", STUDENT_ID),
        ("Report mode", "Quick smoke test" if quick else "Full experiment"),
    ):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_text(paragraph, f"{label}: ", bold=True, size=11)
        _add_text(paragraph, value, size=11)
    _add_paragraph(
        document,
        "All implementation, experiments, figures, and report composition "
        "were performed by the AI agent under the student’s prompts and "
        "choices, as required by the assignment.",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        before=24,
        after=4,
    )
    _page_break(document)


def _section_one(document: Document) -> None:
    _heading(document, "1. How I Asked AI Tools to Find the Algorithm")
    _add_paragraph(
        document,
        "I began by giving Cursor’s AI agent the assignment PDF and the "
        "following prompt. This established the six required report sections "
        "and left the literature search, implementation, experiments, and "
        "writing to the agent.",
    )
    _add_code_block(document, "the original user message", ORIGINAL_PROMPT)
    _add_paragraph(
        document,
        "The agent read the one-page brief, extracted the requirement for a "
        "recent probability- or Bayes-based recognition algorithm, and used "
        "web searches to check publication venue, recency, computer-vision "
        "relevance, code availability, and CPU feasibility. It narrowed the "
        "search to three peer-reviewed candidates instead of selecting the "
        "first Bayesian paper it found.",
    )
    _add_table(
        document,
        "Table 1. Candidate papers researched by the AI agent.",
        ("Candidate", "Venue", "Why it was considered"),
        (
            (
                "ABNN — “Make Me a BNN: A Simple Strategy for Estimating "
                "Bayesian Uncertainty from Pre-trained Models”",
                "CVPR 2024",
                "Directly Bayesian, newest candidate, post-hoc conversion of "
                "normalization layers, and uncertainty/OOD evaluation.",
            ),
            (
                "DDU — “Deep Deterministic Uncertainty: A New Simple "
                "Baseline”",
                "CVPR 2023",
                "Fast single-pass alternative using regularized features and "
                "post-hoc Gaussian discriminant analysis.",
            ),
            (
                "I-EDL — “Uncertainty Estimation by Fisher "
                "Information-based Evidential Deep Learning”",
                "ICML 2023",
                "Dirichlet evidential uncertainty with a Fisher-information "
                "objective, but a less direct BNN fit.",
            ),
        ),
        widths=(7.3, 2.3, 6.8),
    )
    _add_paragraph(
        document,
        "The agent presented these as a multiple-choice decision and "
        "recommended ABNN because it is explicitly Bayesian, was published at "
        "CVPR 2024, and permits a small direct implementation that exposes the "
        "mathematics clearly. I replied: “ABNN (CVPR 2024) and keep doing it,” "
        "supplied my name and student ID, and requested publication in my "
        "CISC3024 GitHub repository.",
    )
    _add_paragraph(
        document,
        "The implementation choice was then refined for the available CPU: a "
        "CIFAR ResNet-20 rather than the paper’s larger ResNet-50 and "
        "WideResNet-28×10 models, CIFAR-10 as in-distribution data, and SVHN "
        "plus CIFAR-100 as practical OOD tests. The agent selected a direct "
        "PyTorch implementation instead of hiding the mechanism behind a "
        "third-party uncertainty package.",
    )


def _plain_number(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _counted(value, singular: str, plural: str | None = None) -> str:
    count = int(value)
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _section_two(
    document: Document, evaluation: dict, quick: bool
) -> None:
    config = evaluation["config"]
    _heading(document, "2. Algorithm Description")
    _heading(document, "2.1 From MAP to Bayesian prediction", 2)
    _add_paragraph(
        document,
        "A deterministic classifier normally keeps one parameter vector. "
        "Maximum a posteriori (MAP) training combines the data likelihood with "
        "a prior over weights. With a Gaussian prior, the prior term is the "
        "familiar L2 weight penalty:",
    )
    _add_equation(
        document,
        "ω_MAP = arg min_ω [−Σᵢ log p(yᵢ | xᵢ, ω) − log p(ω)]",
        1,
    )
    _add_paragraph(
        document,
        "A Bayesian neural network instead predicts by marginalizing over "
        "the posterior distribution of plausible weights:",
    )
    _add_equation(
        document,
        "p(y | x, D) = ∫ p(y | x, ω) p(ω | D) dω",
        2,
    )

    _heading(document, "2.2 Bayesian Normalization Layer (BNL)", 2)
    _add_paragraph(
        document,
        "ABNN starts from a trained deterministic network and replaces its "
        "normalization layers with Bayesian Normalization Layers. For a "
        "normalized activation hⱼ, the paper perturbs the learned scale γⱼ "
        "with independent unit Gaussian noise εⱼ:",
    )
    _add_equation(
        document,
        "BNL(hⱼ) = ((hⱼ − μ̂ⱼ) / σ̂ⱼ) γⱼ(1 + εⱼ) + βⱼ,  "
        "εⱼ ~ N(0, I)",
        3,
    )
    _add_paragraph(
        document,
        "The running statistics, affine scale, bias, and batch counter are "
        "copied from the original BatchNorm layer. The convolutional backbone "
        "and classifier are frozen; only γ and β are adapted. A new Gaussian "
        "draw on every forward pass creates different predictions without "
        "learning a full distribution for every network weight.",
    )

    _heading(document, "2.3 Multiple modes and finite inference", 2)
    _add_paragraph(
        document,
        "One local posterior mode is not enough to represent a multimodal "
        "deep-network posterior. The paper therefore starts several ABNN "
        "members from the same MAP checkpoint and changes class weights during "
        "adaptation so that members specialize differently. Our member m "
        "minimizes weighted cross-entropy with "
        f"{config['emphasized_classes']} seeded emphasized classes at weight "
        f"{_plain_number(config['emphasis_factor'])}. At inference, M members "
        "and L Gaussian draws per member approximate the Bayesian integral:",
    )
    _add_equation(
        document,
        "p(y | x, D) ≈ (1 / ML) Σₘ₌₁ᴹ Σₗ₌₁ᴸ "
        "p(y | x, ωₘ, εₗ)",
        4,
    )
    _add_paragraph(
        document,
        "Predictive entropy H[p] = −Σ꜀ p꜀ log p꜀ is used as the OOD score: "
        "larger entropy means a less certain prediction and therefore a more "
        "OOD-like sample. Calibration is measured by expected calibration "
        "error (ECE); classification also reports accuracy, negative "
        "log-likelihood (NLL), and Brier score.",
    )

    _heading(document, "2.4 What the paper reports", 2)
    _add_paragraph(
        document,
        "CVPR 2024 paper findings. Franchi et al. evaluate ABNN with larger "
        "pre-trained architectures across image classification and semantic "
        "segmentation. They report that post-hoc BNL adaptation preserves the "
        "predictive properties of the deterministic network while improving "
        "uncertainty quantification with relatively small training overhead. "
        "Their analysis attributes most gains to epistemic uncertainty, as "
        "measured by OOD detection, and studies multiple posterior modes and "
        "multiple stochastic samples.",
        bold_prefix="CVPR 2024 paper findings.",
    )
    _add_paragraph(
        document,
        "Those published conclusions are background evidence, not results "
        "measured by this assignment. Section 4 separately labels and reports "
        "our ResNet-20 values from persisted JSON; the reduced CPU study is "
        "not claimed to reproduce the paper’s absolute numbers.",
    )
    _add_figure(
        document,
        1,
        OUTPUT_NAMES[0],
        FIGURE_CAPTIONS[0],
        quick=quick,
    )


def _section_three(document: Document) -> None:
    _heading(document, "3. How AI Implemented the Algorithm")
    _heading(document, "3.1 Implementation sequence", 2)
    for text in (
        "The agent encoded all approved quick/full settings and deterministic "
        "seeds, then built reproducible CIFAR-10, SVHN, and CIFAR-100 loaders.",
        "It implemented a CIFAR ResNet-20 and trained the deterministic MAP "
        "checkpoint with SGD, warm-up, cosine decay, and resumable state.",
        "It recursively converted every BatchNorm2d layer into "
        "BayesianBatchNorm2d while preserving state, then froze all parameters "
        "except normalization scale and bias.",
        "It adapted seeded ABNN members with class-weighted cross-entropy and "
        "persisted checkpoint provenance so members cannot silently resume "
        "from a different MAP model.",
        "It evaluated MAP, one-member ABNN, and all-member ABNN predictions, "
        "checking loader order and probability validity before writing JSON.",
        "It generated six figures exclusively from the saved JSON and recorded "
        "source/output SHA-256 digests in a manifest.",
        "It composed this DOCX from those artifacts and asked Microsoft Word "
        "to export the matching PDF.",
    ):
        _add_numbered(document, text)

    _heading(document, "3.2 Source layout and tests", 2)
    _add_table(
        document,
        "Table 2. Main AI-generated source modules.",
        ("File", "Responsibility"),
        (
            ("code/config.py", "Quick/full policy and reproducible seeds"),
            ("code/model.py", "CIFAR ResNet-20 backbone"),
            ("code/abnn.py", "BNL conversion, freezing, and MC prediction"),
            ("code/train_map.py", "MAP training and JSON history"),
            ("code/train_abnn.py", "Class-diverse ABNN adaptation"),
            ("code/evaluate.py", "Classification, calibration, and OOD output"),
            ("code/make_figures.py", "Six provenance-tracked figures"),
            ("code/run_all.py", "Resumable end-to-end orchestration"),
            ("build_report.py", "Data-driven Word/PDF report"),
        ),
        widths=(5.0, 11.4),
    )
    _add_paragraph(
        document,
        "The agent followed test-driven development: each module’s public "
        "behaviour was first expressed as a failing pytest, then implemented "
        "and checked against the full suite. Tests cover BNL state conversion "
        "and stochasticity, parameter freezing, finite normalized Monte Carlo "
        "probabilities, metric calculations, deterministic evaluation order, "
        "checkpoint provenance, atomic figure publication, orchestration, and "
        "report content/Word cleanup.",
    )

    _heading(document, "3.3 Real source excerpts", 2)
    _add_paragraph(
        document,
        "The report generator reads these excerpts directly from the current "
        "source files. This prevents a copied code listing from drifting away "
        "from the implementation.",
    )
    bnl_excerpt = _source_excerpt(
        "code/abnn.py",
        (
            "def forward(self, input: torch.Tensor)",
            "scale = self.weight",
            "scale = scale * (1.0 + torch.randn_like(scale))",
            "return F.batch_norm(",
            "self.eps,",
        ),
    )
    _add_code_block(document, "code/abnn.py (BNL forward path)", bnl_excerpt)
    freeze_excerpt = _source_excerpt(
        "code/abnn.py",
        (
            "def freeze_except_bayesian_norm",
            "parameter.requires_grad = False",
            "module.weight.requires_grad = True",
            "module.bias.requires_grad = True",
        ),
        context=1,
    )
    _add_code_block(
        document,
        "code/abnn.py (normalization-only adaptation)",
        freeze_excerpt,
    )
    mc_excerpt = _source_excerpt(
        "code/abnn.py",
        (
            "def monte_carlo_probabilities",
            "draws = []",
            "for model in models:",
            "for _ in range(samples):",
            "sample_probs = torch.stack(draws).numpy()",
            "probabilities = sample_probs.mean(axis=0)",
            "return probabilities, saved_targets.numpy(), sample_probs",
        ),
        context=1,
    )
    _add_code_block(
        document,
        "code/abnn.py (finite Monte Carlo averaging)",
        mc_excerpt,
    )


def _settings_rows(evaluation: dict) -> tuple[tuple[str, str], ...]:
    config = evaluation["config"]
    effective = evaluation["run"]["effective"]
    limits = effective["dataset_limits"]
    limit_text = ", ".join(
        f"{name}={value if value is not None else 'all'}"
        for name, value in limits.items()
    )
    return (
        ("Mode", evaluation["run"]["mode"]),
        ("Global seed", str(config["seed"])),
        ("Dataset limits", limit_text),
        ("Batch size", str(config["batch_size"])),
        ("MAP epochs / learning rate", f"{effective['map_epochs']} / {config['map_lr']}"),
        (
            "ABNN epochs / learning rate",
            f"{effective['abnn_epochs']} / {config['abnn_lr']}",
        ),
        (
            "SGD momentum / MAP weight decay",
            f"{config['momentum']} / {config['weight_decay']}",
        ),
        (
            "ABNN members / member seeds",
            f"{effective['member_count']} / {effective['member_seeds']}",
        ),
        ("Monte Carlo draws per member", str(effective["mc_samples"])),
        (
            "Emphasized classes / weight factor",
            f"{config['emphasized_classes']} / {config['emphasis_factor']}",
        ),
    )


def _section_four(
    document: Document,
    evaluation: dict,
    map_history: dict,
    abnn_history: dict,
    manifest: dict,
    quick: bool,
) -> None:
    config = evaluation["config"]
    effective = evaluation["run"]["effective"]
    limits = effective["dataset_limits"]
    _heading(document, "4. Experiment Settings and Results")
    if quick:
        warning = _add_paragraph(
            document,
            "SMOKE-TEST DATA — NOT THE FINAL SUBMISSION. The values below "
            f"come from {limits['train']} training images, "
            f"{limits['cifar10_test']} CIFAR-10 test examples, "
            f"{limits['svhn_test']} SVHN test examples, and "
            f"{limits['cifar100_test']} CIFAR-100 test examples; "
            f"{_counted(effective['map_epochs'], 'MAP epoch')}, "
            f"{_counted(effective['abnn_epochs'], 'ABNN adaptation epoch')}, "
            f"{_counted(effective['member_count'], 'member')}, and "
            f"{_counted(effective['mc_samples'], 'Monte Carlo draw')}. "
            "They validate execution and document generation only; they are "
            "not evidence of final model quality.",
            before=4,
            after=8,
        )
        warning.runs[0].bold = True
        warning.runs[0].font.color.rgb = QUICK_RED

    _heading(document, "4.1 Settings loaded from evaluation JSON", 2)
    _add_paragraph(
        document,
        "The backbone is the source-defined CIFAR ResNet-20. All values in "
        "Table 3 are read from the evaluation artifact’s config and effective "
        "run metadata; quick-mode overrides are shown rather than the unused "
        "full defaults.",
    )
    _add_table(
        document,
        "Table 3. Effective experimental settings loaded from evaluation JSON.",
        ("Setting", "Persisted value"),
        _settings_rows(evaluation),
        widths=(7.0, 9.4),
    )

    _heading(document, "4.2 Training history and provenance", 2)
    map_rows = [
        (
            row["epoch"],
            _fmt(row["loss"]),
            _fmt(row["accuracy"]),
            _fmt(row["nll"]),
            _fmt(row["brier"]),
            _fmt(row["ece"]),
        )
        for row in map_history["history"]
    ]
    _add_table(
        document,
        "Table 4. MAP history loaded from map_history JSON.",
        ("Epoch", "Loss", "Accuracy", "NLL", "Brier", "ECE"),
        map_rows,
        widths=(1.8, 2.7, 2.7, 2.7, 2.7, 2.7),
    )
    _add_paragraph(
        document,
        f"Persisted MAP wall-clock time: "
        f"{_elapsed(map_history['elapsed_seconds'])}.",
    )
    member_rows = []
    for index, rows in enumerate(abnn_history["members"], start=1):
        for row in rows:
            member_rows.append(
                (
                    index,
                    row["epoch"],
                    _fmt(row["loss"]),
                    ", ".join(map(str, row["emphasized_classes"])),
                )
            )
    _add_table(
        document,
        "Table 5. ABNN adaptation history loaded from abnn_history JSON.",
        ("Member", "Epoch", "Weighted loss", "Emphasized classes"),
        member_rows,
        widths=(2.5, 2.5, 4.2, 7.2),
    )
    provenance = evaluation["provenance"]
    provenance_rows = [
        (
            "MAP checkpoint",
            provenance["map_checkpoint"]["path"],
            provenance["map_checkpoint"]["sha256"],
        )
    ]
    provenance_rows.extend(
        (
            f"ABNN member {index}",
            record["path"],
            record["sha256"],
        )
        for index, record in enumerate(
            provenance["member_checkpoints"], start=1
        )
    )
    _add_table(
        document,
        "Table 6. Checkpoint provenance persisted with evaluation results.",
        ("Artifact", "File", "SHA-256"),
        provenance_rows,
        widths=(3.5, 5.2, 7.7),
    )
    _add_paragraph(
        document,
        "The six figures were accepted only after their manifest matched the "
        f"{manifest['mode']} source JSON digests and every PNG’s SHA-256. "
        "This prevents a report from silently combining quick and full "
        "artifacts.",
    )
    _add_figure(
        document,
        2,
        OUTPUT_NAMES[1],
        FIGURE_CAPTIONS[1],
        quick=quick,
    )

    _heading(document, "4.3 Classification and calibration", 2)
    classification_rows = []
    for method_name, label in METHOD_LABELS.items():
        metrics = evaluation["methods"][method_name]["classification"]
        classification_rows.append(
            (
                label,
                _fmt(metrics["accuracy"]),
                _fmt(metrics["nll"]),
                _fmt(metrics["brier"]),
                _fmt(metrics["ece"]),
            )
        )
    _add_table(
        document,
        "Table 7. Measured CIFAR-10 metrics loaded from evaluation JSON "
        "(accuracy higher is better; NLL, Brier, and ECE lower are better).",
        ("Method", "Accuracy", "NLL", "Brier", "ECE"),
        classification_rows,
        widths=(5.2, 2.8, 2.8, 2.8, 2.8),
    )
    _add_figure(
        document,
        3,
        OUTPUT_NAMES[2],
        FIGURE_CAPTIONS[2],
        quick=quick,
    )

    _heading(document, "4.4 OOD detection", 2)
    ood_rows = []
    for method_name, label in METHOD_LABELS.items():
        for dataset_name, dataset_label in DATASET_LABELS.items():
            metrics = evaluation["methods"][method_name]["ood"][dataset_name]
            ood_rows.append(
                (
                    label,
                    dataset_label,
                    _fmt(metrics["auroc"]),
                    _fmt(metrics["aupr"]),
                    _fmt(metrics["fpr95"]),
                )
            )
    _add_table(
        document,
        "Table 8. Predictive-entropy OOD metrics loaded from evaluation JSON "
        "(AUROC/AUPR higher is better; FPR95 lower is better).",
        ("Method", "OOD set", "AUROC", "AUPR", "FPR95"),
        ood_rows,
        widths=(5.1, 3.1, 2.7, 2.7, 2.7),
    )
    _add_figure(
        document,
        4,
        OUTPUT_NAMES[3],
        FIGURE_CAPTIONS[3],
        quick=quick,
    )
    _add_figure(
        document,
        5,
        OUTPUT_NAMES[4],
        FIGURE_CAPTIONS[4],
        quick=quick,
    )
    _add_figure(
        document,
        6,
        OUTPUT_NAMES[5],
        FIGURE_CAPTIONS[5],
        quick=quick,
    )

    _heading(document, "4.5 Interpretation of measured assignment results", 2)
    methods = evaluation["methods"]
    map_metrics = methods["map"]["classification"]
    full_metrics = methods["abnn_full"]["classification"]
    svhn_map = methods["map"]["ood"]["svhn"]
    svhn_full = methods["abnn_full"]["ood"]["svhn"]
    c100_map = methods["map"]["ood"]["cifar100"]
    c100_full = methods["abnn_full"]["ood"]["cifar100"]
    _add_paragraph(
        document,
        "Measured assignment results. "
        f"MAP accuracy is {_fmt(map_metrics['accuracy'])}, while the "
        f"all-available-member ABNN path records "
        f"{_fmt(full_metrics['accuracy'])}. Its ECE is "
        f"{_fmt(full_metrics['ece'])}, compared with "
        f"{_fmt(map_metrics['ece'])} for MAP. On SVHN, entropy AUROC changes "
        f"from {_fmt(svhn_map['auroc'])} (MAP) to "
        f"{_fmt(svhn_full['auroc'])} (ABNN); on CIFAR-100 it changes from "
        f"{_fmt(c100_map['auroc'])} to {_fmt(c100_full['auroc'])}. These are "
        "our persisted measurements and are not values quoted from the paper.",
        bold_prefix="Measured assignment results.",
    )
    if quick:
        if effective["member_count"] == 1:
            quick_comparison = (
                "In quick mode, “ABNN (single member)” and “ABNN (all "
                "available members)” both use the same one adapted member. "
                "Their different numbers arise from separate stochastic "
                "Monte Carlo passes, not from a multimodal ensemble. "
            )
        else:
            quick_comparison = (
                "In this quick mode, the all-available path uses "
                f"{_counted(effective['member_count'], 'adapted member')} "
                "while the single-member path uses one. "
            )
        _add_paragraph(
            document,
            quick_comparison
            + f"The {_counted(effective['map_epochs'], 'MAP epoch')} model "
            "is not a basis for generalizing apparent OOD improvements or "
            "calibration regressions. The configured full run will use "
            f"{_counted(len(config['member_seeds']), 'member')} and "
            f"{_counted(config['mc_samples'], 'Monte Carlo draw')} per "
            "member, where the single-versus-ensemble comparison has its "
            "intended meaning.",
        )
    else:
        _add_paragraph(
            document,
            "The full report uses "
            f"{_counted(effective['member_count'], 'adapted member')} and "
            f"{_counted(effective['mc_samples'], 'Monte Carlo draw')} per "
            "member. Even then, differences remain descriptive because the "
            f"study uses configured MAP seed {config['seed']} and a much "
            "shorter schedule than the paper.",
        )

    _heading(document, "4.6 Limitations", 2)
    for limitation in (
        "CPU model scale: CIFAR ResNet-20 is much smaller than the "
        "ResNet-50 and WideResNet-28×10 models studied in the CVPR paper, so "
        "absolute paper results are not directly comparable.",
        "Short schedule: the configured full run uses "
        f"{_counted(config['map_epochs'], 'MAP epoch')} and "
        f"{_counted(config['abnn_epochs'], 'ABNN adaptation epoch')}; this "
        f"report uses {_counted(effective['map_epochs'], 'MAP epoch')} and "
        f"{_counted(effective['abnn_epochs'], 'ABNN adaptation epoch')} and "
        "is deliberately short relative to the paper.",
        f"One seed: the assignment uses global MAP seed {config['seed']}. "
        "Without repeated training seeds, small metric differences have no "
        "confidence interval and should not be presented as statistically "
        "conclusive.",
        "OOD dataset choice: SVHN supplies a strong semantic/domain shift, "
        "while CIFAR-100 is visually closer to CIFAR-10. These two datasets "
        "do not represent all real deployment shifts and entropy is only one "
        "OOD score.",
        "Method adaptation: class-weighted ABNN members and a small "
        "normalization-based ResNet preserve the central mechanism, but this "
        "is a CPU-scale reproduction rather than the authors’ exact training "
        "recipe.",
    ):
        _add_bullet(document, limitation)


def _section_five(
    document: Document, evaluation: dict, quick: bool
) -> None:
    config = evaluation["config"]
    _heading(document, "5. What I Have Learnt")
    _add_paragraph(
        document,
        "I learnt that a Bayesian prediction is not just a softmax confidence. "
        "The important operation is averaging predictions over plausible "
        "models or stochastic parameter samples. ABNN makes this idea "
        "practical by placing uncertainty in normalization scales rather than "
        "learning a distribution for every convolutional weight.",
    )
    _add_paragraph(
        document,
        "I also learnt why MAP, posterior modes, and stochastic draws describe "
        "different parts of the process. The MAP checkpoint supplies a strong "
        "deterministic starting point; class-weighted adaptation encourages "
        "several nearby modes; repeated Gaussian perturbations sample within "
        "each mode. The final average is therefore two-dimensional: across "
        "members and across draws.",
    )
    _add_paragraph(
        document,
        "Working with an AI agent showed me that the highest-value human input "
        "was setting constraints and making explicit choices. I selected ABNN "
        "from researched alternatives and specified the repository and "
        "identity, while the agent converted those choices into tests, code, "
        "experiments, plots, and this report. The process still required "
        "verification: probability normalization, deterministic data order, "
        "checkpoint hashes, and source-linked figures were all checked rather "
        "than trusted because they were AI-generated.",
    )
    _add_paragraph(
        document,
        "Finally, I learnt that reproducibility and honest limitations matter "
        "more than a flattering score. Every measured value in this report is "
        "loaded from JSON, code excerpts are read from source, and figures are "
        "checked against a digest manifest. "
        + (
            "The current low-accuracy numbers are visibly labeled as a smoke "
            "test; presenting them as final evidence would be misleading."
            if quick
            else "The final values remain a short-schedule CPU study using "
            f"configured MAP seed {config['seed']} and should be interpreted "
            "at that scale."
        ),
    )


def _section_six(document: Document) -> None:
    _heading(document, "6. Webpage Link of the Source Codes")
    _add_paragraph(
        document,
        "The direct GitHub folder containing the source code, tests, results, "
        "figures, and report is:",
    )
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_hyperlink(paragraph, SOURCE_URL)
    paragraph.paragraph_format.space_after = Pt(8)
    _add_paragraph(document, SOURCE_URL)
    _add_paragraph(
        document,
        "The encoded space in AI%20Assignment2 is intentional: this URL opens "
        "the assignment folder directly instead of linking only to the "
        "repository root.",
    )


def _references(document: Document) -> None:
    _heading(document, "References", 1)
    references = (
        (
            "G. Franchi, O. Laurent, M. Leguéry, A. Bursuc, A. Pilzer, and "
            "A. Yao, “Make Me a BNN: A Simple Strategy for Estimating "
            "Bayesian Uncertainty from Pre-trained Models,” Proceedings of "
            "the IEEE/CVF Conference on Computer Vision and Pattern "
            "Recognition (CVPR), 2024, pp. 12194–12203.",
            "https://openaccess.thecvf.com/content/CVPR2024/html/"
            "Franchi_Make_Me_a_BNN_A_Simple_Strategy_for_Estimating_"
            "Bayesian_CVPR_2024_paper.html",
        ),
        (
            "J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. S. Torr, and "
            "Y. Gal, “Deep Deterministic Uncertainty: A New Simple Baseline,” "
            "CVPR, 2023, pp. 24384–24394.",
            "https://openaccess.thecvf.com/content/CVPR2023/html/"
            "Mukhoti_Deep_Deterministic_Uncertainty_A_New_Simple_"
            "Baseline_CVPR_2023_paper.html",
        ),
        (
            "D. Deng, G. Chen, Y. Yu, F. Liu, and P.-A. Heng, “Uncertainty "
            "Estimation by Fisher Information-based Evidential Deep "
            "Learning,” Proceedings of the 40th International Conference on "
            "Machine Learning (ICML), PMLR 202, 2023, pp. 7596–7616.",
            "https://proceedings.mlr.press/v202/deng23b.html",
        ),
        (
            "K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for "
            "Image Recognition,” CVPR, 2016, pp. 770–778.",
            "https://openaccess.thecvf.com/content_cvpr_2016/html/"
            "He_Deep_Residual_Learning_CVPR_2016_paper.html",
        ),
        (
            "A. Krizhevsky, “Learning Multiple Layers of Features from Tiny "
            "Images,” University of Toronto, 2009 (CIFAR-10/CIFAR-100).",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
        ),
    )
    for index, (citation, url) in enumerate(references, start=1):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.65)
        paragraph.paragraph_format.first_line_indent = Cm(-0.65)
        paragraph.paragraph_format.space_after = Pt(5)
        _add_text(paragraph, f"[{index}] {citation} ", size=9.5)
        _add_hyperlink(paragraph, url, url)


def _compose_document(
    evaluation: dict,
    map_history: dict,
    abnn_history: dict,
    manifest: dict,
    *,
    quick: bool,
) -> Document:
    document = Document()
    _configure_document(document, quick)
    _title_page(document, quick)
    _section_one(document)
    _section_two(document, evaluation, quick)
    _section_three(document)
    _section_four(
        document,
        evaluation,
        map_history,
        abnn_history,
        manifest,
        quick,
    )
    _section_five(document, evaluation, quick)
    _section_six(document)
    _references(document)
    return document


def _dispatch_word():
    try:
        import win32com.client  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PDF conversion requires pywin32 and desktop Microsoft Word. "
            "Install requirements.txt and run on Windows with Word installed."
        ) from error
    try:
        return win32com.client.DispatchEx("Word.Application")
    except Exception as error:
        raise RuntimeError(
            "Could not start Microsoft Word for PDF conversion. Verify that "
            "desktop Word is installed, licensed, and not blocked by a dialog."
        ) from error


def _convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> Path:
    docx_path = Path(docx_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    if not docx_path.is_file():
        raise FileNotFoundError(
            f"Cannot convert missing Word document: {docx_path}"
        )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.unlink(missing_ok=True)

    word = None
    opened_document = None
    failure: Exception | None = None
    try:
        word = _dispatch_word()
        word.Visible = False
        word.DisplayAlerts = 0
        opened_document = word.Documents.Open(str(docx_path))
        opened_document.Fields.Update()
        opened_document.ExportAsFixedFormat(str(pdf_path), 17)
    except Exception as error:
        failure = error
    finally:
        if opened_document is not None:
            try:
                opened_document.Close(False)
            except Exception as error:
                if failure is None:
                    failure = error
            finally:
                opened_document = None
        if word is not None:
            try:
                word.Quit()
            except Exception as error:
                if failure is None:
                    failure = error
            finally:
                word = None

    if failure is not None:
        pdf_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Microsoft Word could not convert "
            f"{docx_path.name} to PDF. Close any Word dialogs or locked report "
            f"files, verify that Word is installed and licensed, then retry. "
            f"Underlying error: {failure}"
        ) from failure
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError(
            "Microsoft Word returned without creating a non-empty PDF at "
            f"{pdf_path}. Check Word's PDF export installation and retry."
        )
    return pdf_path


def _validate_docx(path: Path) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Expected a non-empty DOCX at {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("DOCX contains a corrupt ZIP member")
            if "word/document.xml" not in archive.namelist():
                raise ValueError("DOCX is missing word/document.xml")
        Document(path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid staged DOCX {path}: {error}") from error


def _validate_pdf(path: Path) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Expected a non-empty PDF at {path}")
    with path.open("rb") as artifact:
        header = artifact.read(8)
        artifact.seek(max(0, path.stat().st_size - 1024))
        trailer = artifact.read()
    if not header.startswith(b"%PDF-") or b"%%EOF" not in trailer:
        raise ValueError(f"Invalid staged PDF structure at {path}")


def _validate_report_pair(docx_path: Path, pdf_path: Path) -> None:
    _validate_docx(docx_path)
    _validate_pdf(pdf_path)


def _replace_file(source: Path, destination: Path) -> None:
    Path(source).replace(destination)


def _publish_report_pair(
    staged_docx: Path,
    staged_pdf: Path,
    published_docx: Path,
    published_pdf: Path,
) -> None:
    staged_docx = Path(staged_docx)
    staged_pdf = Path(staged_pdf)
    published_docx = Path(published_docx)
    published_pdf = Path(published_pdf)
    try:
        _validate_report_pair(staged_docx, staged_pdf)
        published_docx.parent.mkdir(parents=True, exist_ok=True)
        if published_pdf.parent != published_docx.parent:
            raise ValueError("DOCX and PDF must publish to the same directory")

        existing = (published_docx.exists(), published_pdf.exists())
        if existing[0] != existing[1]:
            raise RuntimeError(
                "Refusing to replace an incomplete existing report pair"
            )
        previous_pair = all(existing)
        if previous_pair:
            _validate_report_pair(published_docx, published_pdf)

        with tempfile.TemporaryDirectory(
            prefix="report-backup-",
            dir=published_docx.parent,
        ) as backup_directory:
            backup_directory = Path(backup_directory)
            backups = (
                backup_directory / published_docx.name,
                backup_directory / published_pdf.name,
            )
            if previous_pair:
                shutil.copy2(published_docx, backups[0])
                shutil.copy2(published_pdf, backups[1])

            try:
                _replace_file(staged_docx, published_docx)
                _replace_file(staged_pdf, published_pdf)
                _validate_report_pair(published_docx, published_pdf)
            except Exception as error:
                rollback_errors = []
                for destination, backup in zip(
                    (published_docx, published_pdf), backups
                ):
                    try:
                        if previous_pair:
                            _replace_file(backup, destination)
                        else:
                            destination.unlink(missing_ok=True)
                    except Exception as rollback_error:
                        rollback_errors.append(str(rollback_error))
                if rollback_errors:
                    raise RuntimeError(
                        "Report publication failed and rollback also failed: "
                        + "; ".join(rollback_errors)
                    ) from error
                outcome = (
                    "restored the previous pair"
                    if previous_pair
                    else "removed the partial pair"
                )
                raise RuntimeError(
                    f"Report publication failed and {outcome}: {error}"
                ) from error
    finally:
        staged_docx.unlink(missing_ok=True)
        staged_pdf.unlink(missing_ok=True)


def _report_names(quick: bool) -> tuple[str, str]:
    if quick:
        return QUICK_DOCX_NAME, QUICK_PDF_NAME
    return DOCX_NAME, PDF_NAME


def build_report(quick: bool = False) -> tuple[Path, Path]:
    """Build the quick or full assignment report and return DOCX/PDF paths."""
    source_snapshot = _source_snapshot(_report_source_paths(quick))
    evaluation, map_history, abnn_history, manifest = _load_report_artifacts(
        quick
    )
    _assert_source_snapshot(source_snapshot, "while loading")
    document = _compose_document(
        evaluation,
        map_history,
        abnn_history,
        manifest,
        quick=quick,
    )
    _assert_source_snapshot(source_snapshot, "during composition")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docx_name, pdf_name = _report_names(quick)
    docx_path = REPORT_DIR / docx_name
    pdf_path = REPORT_DIR / pdf_name
    with tempfile.NamedTemporaryFile(
        prefix="report-stage-",
        suffix=".docx",
        dir=REPORT_DIR,
        delete=False,
    ) as temporary:
        staged_docx = Path(temporary.name)
    staged_pdf = staged_docx.with_suffix(".pdf")
    try:
        document.save(staged_docx)
        _validate_docx(staged_docx)
        _assert_source_snapshot(source_snapshot, "while staging the DOCX")
        _convert_docx_to_pdf(staged_docx, staged_pdf)
        _validate_report_pair(staged_docx, staged_pdf)
        _assert_source_snapshot(source_snapshot, "during conversion")
        _publish_report_pair(
            staged_docx,
            staged_pdf,
            docx_path,
            pdf_path,
        )
    finally:
        staged_docx.unlink(missing_ok=True)
        staged_pdf.unlink(missing_ok=True)
    return docx_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the ABNN assignment DOCX and PDF report"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use smoke-test JSON and visibly label the report as non-final",
    )
    args = parser.parse_args()
    docx_path, pdf_path = build_report(quick=args.quick)
    print(f"saved {docx_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
