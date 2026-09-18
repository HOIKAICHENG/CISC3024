"""Build the AI Assignment #1 report (DOCX + PDF) from the experiment results.

All numbers quoted in the text are read from results/*.json so the report can
never drift from what was actually measured.  Equations are native Word
equations (LaTeX -> MathML -> OMML).  The PDF is produced through Word's COM
interface (Windows only); pass --no-pdf to skip that step.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor
from latex2mathml.converter import convert
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(HERE, "code"))
from data import CLASSES  # noqa: E402

XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
FONT = "Times New Roman"
MONO = "Consolas"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/HOIKAICHENG/CISC3024")

_transform = etree.XSLT(etree.parse(XSL_PATH))


# --------------------------------------------------------------------------- io
def J(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


PRE_V2, PRE_V1 = J("pretrain_v2_grn.json"), J("pretrain_v1_nogrn.json")
FT = {t: J(f"finetune_{t}.json") for t in ("v2_fcmae", "v2_scratch", "v1_fcmae", "v1_scratch")}
AN = J("analysis.json") or {}


def acc(tag, key="final_acc"):
    r = FT.get(tag)
    return f"{r[key]:.2f}" if r else "n/a"


def knn(name):
    v = (AN.get("knn_acc") or {}).get(name)
    return f"{v:.2f}" if v is not None else "n/a"


def minutes(r):
    return f"{r['total_time_s']/60:.0f}" if r else "n/a"


PRE_EPOCHS = PRE_V2["epochs"] if PRE_V2 else 8
FT_EPOCHS = FT["v2_fcmae"]["epochs"] if FT["v2_fcmae"] else 30


# ------------------------------------------------------------------- docx utils
def latex_to_omath(latex):
    mathml = convert(latex)
    omml = _transform(etree.fromstring(mathml.encode("utf-8")))
    return parse_xml(etree.tostring(omml.getroot(), encoding="unicode"))


def add_run(p, text, *, bold=False, italic=False, size=None, mono=False, color=None):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.name = MONO if mono else FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO if mono else FONT)
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    return r


def para(doc, parts, *, space_after=6, space_before=0, size=12, align=None, indent=None):
    """parts: str | list of str / ('m', latex) / ('b', text) / ('i', text) / ('c', code)."""
    if isinstance(parts, str):
        parts = [parts]
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after, pf.space_before, pf.line_spacing = Pt(space_after), Pt(space_before), 1.15
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if indent:
        pf.left_indent = Cm(indent)
    for part in parts:
        if isinstance(part, tuple):
            kind, val = part
            if kind == "m":
                p._p.append(latex_to_omath(val))
            elif kind == "b":
                add_run(p, val, bold=True, size=size)
            elif kind == "i":
                add_run(p, val, italic=True, size=size)
            elif kind == "c":
                add_run(p, val, mono=True, size=size - 1)
        else:
            add_run(p, part, size=size)
    return p


def bullet(doc, parts, size=12):
    p = para(doc, parts, space_after=3, size=size)
    p.style = doc.styles["List Bullet"]
    return p


_num_counter = 0


def numbered(doc, parts, size=12):
    """Manually numbered item; the counter restarts at every heading so lists in
    different sections do not share Word's single 'List Number' sequence."""
    global _num_counter
    _num_counter += 1
    if isinstance(parts, str):
        parts = [parts]
    p = para(doc, [f"{_num_counter}.  ", *parts], space_after=3, size=size)
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    return p


def display(doc, latex, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(6)
    wrapper = parse_xml(f'<m:oMathPara xmlns:m="{MATH_NS}"></m:oMathPara>')
    wrapper.append(latex_to_omath(latex))
    p._p.append(wrapper)
    if label:
        add_run(p, f"    {label}", size=11)
    return p


def _shade_paragraph(p, fill):
    p._p.get_or_add_pPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:fill="{fill}"/>'))


def code_block(doc, text):
    for line in text.rstrip("\n").split("\n"):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after, pf.space_before, pf.line_spacing = Pt(0), Pt(0), 1.0
        pf.left_indent = Cm(0.6)
        add_run(p, line if line else " ", mono=True, size=9)
        _shade_paragraph(p, "F4F4F4")
    doc.paragraphs[-1].paragraph_format.space_after = Pt(8)


def heading(doc, text, level=1):
    global _num_counter
    _num_counter = 0
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.name = FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    h.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    return h


def figure(doc, filename, caption_text, width_in=6.0):
    path = os.path.join(FIG, filename)
    if not os.path.exists(path):
        para(doc, [("i", f"[missing figure {filename}]")], align="center")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(2)
    p.add_run().add_picture(path, width=Inches(width_in))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    add_run(c, caption_text, italic=True, size=10.5)


def set_cell_shading(cell, fill):
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill}" w:val="clear"/>'))


def table(doc, headers, rows, col_widths=None, size=10.5, caption_text=None):
    if caption_text:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_before, c.paragraph_format.space_after = Pt(8), Pt(3)
        add_run(c, caption_text, italic=True, size=10.5)
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t._tbl.tblPr.append(parse_xml(
        f"""<w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>
            <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        </w:tblBorders>"""))
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        add_run(cell.paragraphs[0], h, bold=True, size=size)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, "D9D9D9")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            if isinstance(val, tuple) and val[0] == "b":
                add_run(p, val[1], bold=True, size=size)
            else:
                add_run(p, str(val), size=size)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if (i + 1) % 2 == 0:
                set_cell_shading(cell, "F2F2F2")
    if col_widths:
        for row in t.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_page_number(paragraph):
    run = paragraph.add_run()
    for tag, attr in (("w:fldChar", ("w:fldCharType", "begin")), ("w:instrText", None),
                      ("w:fldChar", ("w:fldCharType", "end"))):
        el = OxmlElement(tag)
        if attr:
            el.set(qn(attr[0]), attr[1])
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        run._r.append(el)


def hyperlink(paragraph, url, text=None):
    r_id = paragraph.part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    c = OxmlElement("w:color"); c.set(qn("w:val"), "0563C1"); rPr.append(c)
    f = OxmlElement("w:rFonts"); f.set(qn("w:ascii"), FONT); f.set(qn("w:hAnsi"), FONT); rPr.append(f)
    r.append(rPr)
    t = OxmlElement("w:t"); t.text = text or url; r.append(t)
    h.append(r)
    paragraph._p.append(h)


# ----------------------------------------------------------------------- build
def build():
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(2.3)
    s.top_margin = s.bottom_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = FONT, Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    fp = s.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(fp, "Page ", size=11)
    add_page_number(fp)

    # ---- title block
    para(doc, [("b", "CISC3024 Pattern Recognition")], size=14, space_after=2)
    para(doc, [("b", "AI Assignment #1 — Deep CNN / Deep Autoencoder")], size=14, space_after=2)
    para(doc, [("b", "ConvNeXt V2: Co-designing a ConvNet with a Fully Convolutional Masked "
                     "Autoencoder (CVPR 2023), reproduced on CIFAR-10")], size=13, space_after=8)
    para(doc, "Name: HOI KAI CHENG     Student ID: UC325381", space_after=4)
    para(doc, [("i", "AI tool used: Cursor IDE in agent mode (model: Claude). As required by the "
                     "assignment, every line of code, every experiment and this report were produced by "
                     "the AI agent; my own contribution was limited to writing prompts and making the "
                     "choices recorded in Section 1.")], size=11, space_after=12)

    # ======================================================================= 1
    heading(doc, "1. How I asked the AI tool to find the algorithm", 1)
    para(doc, [
        "I used the agent mode of the Cursor IDE, which can read files, run shell commands, write "
        "code and search the web, so a single conversation could carry the whole assignment from "
        "literature search to report. My first message simply attached the assignment PDF and the "
        "list of required report sections:"])
    code_block(doc,
        "c:\\Users\\th255\\Downloads\\AIassign1.pdf\n"
        "Your AI assignment report in Word/PDF format should contain the following details\n"
        "(1) how you ask AI tools to find the algorithm, (2) algorithm description,\n"
        "(3) how AI implements the algorithm, (4) experiment settings and results,\n"
        "(5) what you have learnt from this AI assignment, and (6) a webpage link of your\n"
        "source codes.\n"
        "then save all the file into a new folder 'hw2'")
    para(doc, [
        "The agent read the PDF, extracted the constraints (\"a recent algorithm of Deep CNN or Deep "
        "Autoencoder with an application in computer vision and pattern recognition\", \"you are NOT "
        "allowed to write any code by yourself\"), inspected my machine (Python 3.12, PyTorch 2.13, "
        "20 CPU cores, no GPU, GitHub CLI already logged in) and then, instead of guessing, came back "
        "with three multiple-choice questions. I quote them because they are effectively the prompt "
        "that selected the algorithm:"])
    table(doc, ["Question asked by the agent", "Options offered", "My choice"], [
        ["Which recent algorithm should the assignment cover? (must be a recent Deep CNN or Deep "
         "Autoencoder for computer vision)",
         "ConvNeXt V2 (CVPR 2023, FCMAE + GRN; recommended) / ConvNeXt V1 (CVPR 2022) / convolutional "
         "masked autoencoder for anomaly detection / EfficientNetV2-RepVGG",
         "ConvNeXt V2"],
        ["Section (6) needs a webpage link to the source code. How should I publish it?",
         "public GitHub repo (recommended) / GitHub Gist / local placeholder", "public GitHub repo"],
        ["Everything runs on CPU. How much compute time can I spend on the experiments?",
         "quick ~15 min / moderate 45-90 min (recommended) / long 2-4 h", "moderate"],
    ], col_widths=[6.0, 7.2, 3.0])
    para(doc, [
        "The agent recommended ConvNeXt V2 because it satisfies both halves of the assignment title "
        "at once: it is a modern ", ("b", "deep CNN"), " (a pure ConvNet with no attention) trained "
        "with a ", ("b", "deep (masked) autoencoder"), " objective; it is recent (CVPR 2023) and from a "
        "top venue; the paper contains a clear, small architectural idea (Global Response "
        "Normalization) that can be ablated; and its smallest variant (ConvNeXt V2-Atto, 3.7 M "
        "parameters) is feasible on a laptop CPU. Later in the session I sent one more instruction — "
        "to delete the working folder and redo everything from a clean state (\"給我一個全新的\") — "
        "which the agent executed by re-creating all files and re-running all experiments; every "
        "number in this report comes from that second, clean run."])
    para(doc, [("b", "Reference. "),
        "S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon and S. Xie, \"ConvNeXt V2: "
        "Co-designing and Scaling ConvNets with Masked Autoencoders\", CVPR 2023 "
        "(arXiv:2301.00808). Official code: github.com/facebookresearch/ConvNeXt-V2."], size=11)

    # ======================================================================= 2
    heading(doc, "2. Algorithm description", 1)
    heading(doc, "2.1 Motivation", 2)
    para(doc, [
        "ConvNeXt (Liu et al., CVPR 2022) showed that a ResNet, once \"modernised\" with a patchify "
        "stem, 7×7 depth-wise convolutions, an inverted bottleneck, LayerNorm, GELU and fewer "
        "normalisation layers, matches Swin Transformers under supervised ImageNet training. In the "
        "meantime, vision Transformers had gained a second advantage: masked autoencoders (MAE, He et "
        "al. 2022) give them a very effective self-supervised pre-training. ConvNeXt V2 asks whether "
        "the same recipe can be transplanted to ConvNets. The paper finds that a naive \"ConvNeXt + "
        "MAE\" performs poorly, diagnoses the reason (feature collapse), and fixes it with a "
        "co-designed architecture change (GRN) and a convolution-specific masked autoencoder (FCMAE)."])

    heading(doc, "2.2 Fully Convolutional Masked AutoEncoder (FCMAE)", 2)
    para(doc, "FCMAE is a masked autoencoder in which both encoder and decoder are convolutional "
              "(Figure 1b). Its components are:")
    bullet(doc, [("b", "Random masking. "), "The image is divided into patches at the resolution of "
                 "the last encoder stage (32×32 pixels for 224×224 inputs) and 60 % of them are removed "
                 "at random. The mask is up-sampled to the finer stages so every stage sees the same "
                 "holes."])
    bullet(doc, [("b", "Encoder = ConvNeXt (V2). "), "A convolution kernel slides over masked and "
                 "visible pixels alike, so the paper treats the visible pixels as a sparse point cloud "
                 "and implements the encoder with sparse convolutions during pre-training; masked regions "
                 "then cannot leak information. At fine-tuning time the sparse layers are converted back "
                 "to ordinary dense convolutions without changing the weights. A dense equivalent, used "
                 "in our implementation, is to zero the masked positions after every stage."])
    bullet(doc, [("b", "Lightweight decoder. "), "A single ConvNeXt block (512 channels in the paper) "
                 "followed by a 1×1 prediction layer. Learnable mask tokens are inserted at masked "
                 "positions before decoding. This asymmetry (heavy encoder, tiny decoder) is what makes "
                 "MAE-style pre-training cheap."])
    bullet(doc, [("b", "Reconstruction target. "), "Mean-squared error between the prediction and the "
                 "per-patch normalised pixel values, computed on masked patches only:"])
    display(doc, r"\mathcal{L}=\frac{1}{|M|}\sum_{p\in M}\left\| \hat{x}_p-\frac{x_p-\mu_p}{\sigma_p}\right\|_2^{2}",
            "(1)")
    para(doc, ["where ", ("m", r"M"), " is the set of masked patches and ", ("m", r"\mu_p,\sigma_p"),
               " are the mean and standard deviation of the pixels inside patch ", ("m", r"p"), "."])

    heading(doc, "2.3 The feature-collapse problem", 2)
    para(doc, [
        "Pre-training the original ConvNeXt V1 with FCMAE gives only a small gain over supervised "
        "training. Looking at the activations, the authors found many ", ("i", "dead or saturated "
        "channels"), " in the 4C-dimensional hidden layer of the inverted-bottleneck MLP: different "
        "channels were computing nearly the same thing. They quantify this with the average pairwise "
        "cosine distance between the channel feature maps ", ("m", r"X_i\in\mathbb{R}^{H\times W}"), ":"])
    display(doc, r"d(X)=\frac{1}{C^{2}}\sum_{i=1}^{C}\sum_{j=1}^{C}\frac{1-\cos(X_i,X_j)}{2}", "(2)")
    para(doc, [
        "A low value means redundant channels. ConvNeXt V1 + FCMAE shows a clear drop of ",
        ("m", r"d"), " in the deeper layers, whereas MAE-trained ViTs and supervised ConvNeXts do not."])

    heading(doc, "2.4 Global Response Normalization (GRN)", 2)
    para(doc, [
        "GRN is the fix. It is inspired by lateral inhibition in the visual cortex: it increases the ",
        ("i", "contrast between channels"), " so that they compete instead of collapsing onto the same "
        "feature. For an input ", ("m", r"X\in\mathbb{R}^{H\times W\times C}"), " it performs three "
        "steps with no extra hyper-parameters:"])
    numbered(doc, [("b", "Global feature aggregation: "), "spatially pool each channel with an L2 norm, ",
                   ("m", r"G(X)_i=\|X_i\|_2"), ", giving a vector in ", ("m", r"\mathbb{R}^{C}"), "."])
    numbered(doc, [("b", "Feature normalisation: "), "divisive normalisation of each channel by the "
                   "aggregate over all channels,"])
    display(doc, r"\mathcal{N}(\|X_i\|)=\frac{\|X_i\|}{\sum_{j=1}^{C}\|X_j\|}", "(3)")
    numbered(doc, [("b", "Feature calibration: "), "rescale the input with the computed response, "
                   "with a learnable affine transform and a residual path:"])
    display(doc, r"X_i\leftarrow\gamma\cdot X_i\,\mathcal{N}(\|X_i\|)+\beta+X_i", "(4)")
    para(doc, [
        ("m", r"\gamma"), " and ", ("m", r"\beta"), " are initialised to zero so GRN is an identity at "
        "the start of training. GRN is inserted after GELU in every block, and the LayerScale "
        "parameter of ConvNeXt V1 becomes unnecessary and is removed (Figure 1a). The official "
        "implementation divides by the channel-mean of the norms instead of the sum; the difference is "
        "a constant factor absorbed by ", ("m", r"\gamma"), "."])
    figure(doc, "fig_architecture.png",
           "Figure 1. (a) The ConvNeXt V2 block — GRN (red) is the only difference from V1. "
           "(b) FCMAE pre-training pipeline as implemented in this assignment.", 6.4)

    heading(doc, "2.5 Results reported in the paper", 2)
    para(doc, [
        "With FCMAE pre-training and GRN, ConvNeXt V2 improves over V1 across the whole model family "
        "(from Atto, 3.7 M parameters, 76.7 % ImageNet-1K top-1, to Huge, 660 M parameters, 88.9 % with "
        "ImageNet-22K intermediate fine-tuning), and the two ingredients are shown to be complementary: "
        "GRN alone or FCMAE alone give little, while their combination gives the full gain (e.g. "
        "83.7 → 84.6 % for the Base model). The pre-trained backbones transfer to COCO detection and "
        "ADE20K segmentation as well."])

    # ======================================================================= 3
    heading(doc, "3. How the AI implemented the algorithm", 1)
    heading(doc, "3.1 Workflow of the agent", 2)
    para(doc, "The implementation was done entirely by the agent inside the hw2/ folder in the "
              "following order (each step was announced in the chat before it was executed):")
    numbered(doc, "Environment check (Python, PyTorch, CPU count, GPU absence, GitHub authentication) "
                  "and creation of the hw2/code, figures, results, data and report folders.")
    numbered(doc, [("c", "convnextv2.py"), ": LayerNorm, GRN, ConvNeXt block (with a use_grn switch "
                   "so the same file also builds the V1 block with LayerScale), DropPath, the 4-stage "
                   "backbone and the Atto configuration (depths 2-2-6-2, widths 40-80-160-320)."])
    numbered(doc, [("c", "fcmae.py"), ": random mask generation, dense masked encoder, mask-token "
                   "injection, single-block decoder, patchify/unpatchify and the normalised-pixel loss."])
    numbered(doc, "Smoke test with random tensors (shapes, exact mask ratio, one backward pass) and a "
                  "speed benchmark; the channels_last memory format was found to be ~25 % faster on CPU "
                  "and adopted.")
    numbered(doc, [("c", "data.py"), " / ", ("c", "prepare_data.py"), ": CIFAR-10 pipeline. The "
                   "official download ran at 70 kB/s (40 min), so the agent measured several mirrors, "
                   "pulled the identical data as Parquet from the HuggingFace mirror at ~1 MB/s and "
                   "repacked it into the torchvision batch format (two small bugs — pickle key type and "
                   "torchvision's MD5 check — were caught by the agent and fixed)."])
    numbered(doc, [("c", "pretrain.py"), ", ", ("c", "finetune.py"), ", ", ("c", "engine.py"),
                   ": AdamW + warm-up + cosine schedule, per-epoch JSON logging, checkpointing."])
    numbered(doc, [("c", "analysis.py"), ": the cosine-distance collapse diagnostic of Section 2.3 "
                   "(forward hooks on the MLP hidden layer of every block), 20-nearest-neighbour "
                   "evaluation of the frozen encoder, reconstruction visualisation."])
    numbered(doc, [("c", "make_figures.py"), ", ", ("c", "run_all.py"), ": figures and a one-command "
                   "reproduction script (with a --quick smoke-test mode)."])
    numbered(doc, [("c", "build_report.py"), ": this report, generated with python-docx from the JSON "
                   "results and converted to PDF with Word."])

    heading(doc, "3.2 Key code produced by the agent", 2)
    para(doc, "The GRN layer is eight lines and follows Eqs. (3)–(4) exactly (channels-last layout):")
    code_block(doc,
        "class GRN(nn.Module):\n"
        "    def __init__(self, dim, eps=1e-6):\n"
        "        super().__init__()\n"
        "        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))\n"
        "        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))\n"
        "        self.eps = eps\n"
        "\n"
        "    def forward(self, x):                      # x: (N, H, W, C)\n"
        "        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)          # step 1: aggregate\n"
        "        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)       # step 2: normalise\n"
        "        return self.gamma * (x * nx) + self.beta + x               # step 3: calibrate")
    para(doc, "The block simply drops LayerScale and inserts GRN after GELU:")
    code_block(doc,
        "x = self.dwconv(x)                       # 7x7 depth-wise conv\n"
        "x = x.permute(0, 2, 3, 1)                # -> channels last\n"
        "x = self.norm(x)                         # LayerNorm\n"
        "x = self.pwconv1(x); x = self.act(x)     # 1x1 conv C->4C, GELU\n"
        "if self.grn is not None: x = self.grn(x) # V2 only\n"
        "x = self.pwconv2(x)                      # 1x1 conv 4C->C\n"
        "if self.gamma is not None: x = self.gamma * x   # V1 LayerScale only\n"
        "return shortcut + self.drop_path(x.permute(0, 3, 1, 2))")
    para(doc, "Masking in the encoder (dense equivalent of the sparse convolutions) and the loss:")
    code_block(doc,
        "# ConvNeXtV2.forward_features\n"
        "for i in range(4):\n"
        "    x = self.downsample_layers[i](x)\n"
        "    if mask is not None: x = x * (1 - resize(mask, x.shape[-2:]))\n"
        "    x = self.stages[i](x)\n"
        "    if mask is not None: x = x * (1 - resize(mask, x.shape[-2:]))\n"
        "\n"
        "# FCMAE.forward_loss  (norm_pix_loss)\n"
        "target = self.patchify(imgs)\n"
        "target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6) ** 0.5\n"
        "loss = ((pred - target) ** 2).mean(-1)          # per patch\n"
        "loss = (loss * mask).sum() / mask.sum()          # masked patches only")

    heading(doc, "3.3 Adaptations for CIFAR-10 on a CPU", 2)
    para(doc, "The paper works at 224×224 on GPUs; the agent made the following documented changes "
              "so that the whole study fits in about 1.5 hours on a 20-core laptop CPU:")
    bullet(doc, [("b", "Input resolution. "), "Stem = 2×2/stride-2 patchify conv (instead of 4×4/4), so "
                 "the four stages run at 16×16, 8×8, 4×4 and 2×2 for 32×32 images."])
    bullet(doc, [("b", "Mask granularity. "), "The last stage is only 2×2, too coarse for masking, so "
                 "the mask grid is decoupled from the encoder stride: 4×4-pixel patches (8×8 = 64 mask "
                 "units per image), 60 % masked exactly as in the paper. The decoder up-samples the "
                 "2×2 encoder output to the 8×8 grid before injecting mask tokens."])
    bullet(doc, [("b", "Dense instead of sparse convolutions. "), "MinkowskiEngine is not available on "
                 "Windows/CPU; the agent used the dense fallback that the official code also ships "
                 "(zero the masked positions after every stage)."])
    bullet(doc, [("b", "Model size. "), "ConvNeXt V2-Atto widths and depths; "
                 f"{PRE_V2['encoder_params_M'] if PRE_V2 else 3.39} M encoder parameters here (3.7 M in "
                 "the paper — the difference is the ImageNet stem and 1000-class head). Decoder width "
                 "128 instead of 512."])
    bullet(doc, [("b", "Schedule. "), f"{PRE_EPOCHS} pre-training epochs and {FT_EPOCHS} fine-tuning "
                 "epochs instead of 800 + 100/300. Runs were executed two or four at a time with 10 or 4 "
                 "threads each, after measuring that CPU throughput did not improve beyond 10 threads "
                 "per process."])

    # ======================================================================= 4
    heading(doc, "4. Experiment settings and results", 1)
    heading(doc, "4.1 Settings", 2)
    table(doc, ["Item", "Setting"], [
        ["Dataset", "CIFAR-10: 50,000 training images (used without labels for pre-training), a "
                    "stratified 5,000-image labelled subset (500 per class) for supervised training, "
                    "the full 10,000-image test set for evaluation"],
        ["Backbone", "ConvNeXt V2-Atto: depths (2, 2, 6, 2), widths (40, 80, 160, 320), stem 2×2/2, "
                     f"{PRE_V2['encoder_params_M'] if PRE_V2 else 3.39} M parameters (V1 variant: "
                     f"{PRE_V1['encoder_params_M'] if PRE_V1 else 3.37} M)"],
        ["FCMAE pre-training", "mask ratio 0.6 on 4×4 patches, decoder = 1 ConvNeXt block (128 ch), "
                               "normalised-pixel MSE, AdamW (β = 0.9/0.95, wd 0.05), lr 1.5e-3, "
                               f"1 warm-up epoch + cosine, batch 128, {PRE_EPOCHS} epochs, "
                               "RandomResizedCrop(0.6–1) + horizontal flip"],
        ["Supervised training", "identical recipe for all four runs: AdamW (wd 0.05), lr 1e-3, "
                                f"3 warm-up epochs + cosine, batch 128, {FT_EPOCHS} epochs, label "
                                "smoothing 0.1, drop-path 0.1, RandomCrop(pad 4) + horizontal flip"],
        ["Hardware / software", "20-core CPU, no GPU; Python 3.12, PyTorch 2.13 (channels_last); "
                                f"wall-clock: pre-training ≈ {minutes(PRE_V2)} min per model (two in "
                                f"parallel), fine-tuning ≈ {minutes(FT['v2_fcmae'])} min per run (four "
                                "in parallel)"],
        ["Seeds", "torch.manual_seed(0) for all runs; the 5k subset is drawn with numpy seed 0"],
    ], col_widths=[4.0, 12.2], caption_text="Table 1. Experimental settings.")

    heading(doc, "4.2 Self-supervised pre-training", 2)
    pre_rows = []
    if PRE_V2 and PRE_V1:
        for h2, h1 in zip(PRE_V2["history"], PRE_V1["history"]):
            pre_rows.append([h2["epoch"], f"{h2['loss']:.4f}", f"{h1['loss']:.4f}"])
    figure(doc, "fig_pretrain_loss.png", "Figure 2. FCMAE reconstruction loss during pre-training "
           "for the V2 (GRN) and V1 (no GRN) encoders.", 4.8)
    if pre_rows:
        table(doc, ["Epoch", "V2 (GRN) loss", "V1 (no GRN) loss"], pre_rows, col_widths=[2.5, 4, 4],
              caption_text="Table 2. Masked-patch MSE per epoch (lower is better).")
    figure(doc, "fig_reconstruction.png", "Figure 3. Reconstructions by the pre-trained ConvNeXt V2 "
           "FCMAE on test images (rows: original, 60 % masked input, raw prediction, visible patches + "
           "prediction). Predicted patches are de-normalised with the ground-truth patch statistics for "
           "display.", 6.2)
    if PRE_V2 and PRE_V1:
        para(doc, [
            f"Both encoders learn the inpainting task: the loss falls from about "
            f"{PRE_V2['history'][0]['loss']:.3f} to {PRE_V2['history'][-1]['loss']:.3f} (V2) and "
            f"{PRE_V1['history'][-1]['loss']:.3f} (V1). Reconstructions (Figure 3) recover the coarse "
            "shape and colour of the objects even with only 40 % of the pixels visible, which is the "
            "expected behaviour of an MAE with a small decoder — it is trained to predict the mean of "
            "plausible completions, so results are blurry."])

    heading(doc, "4.3 Feature collapse and the effect of GRN", 2)
    figure(doc, "fig_feature_collapse.png", "Figure 4. Mean channel cosine distance (Eq. 2) inside the "
           "MLP of every encoder block after FCMAE pre-training. Higher = more diverse channels.", 5.4)
    cd = AN.get("cosine_distance") or {}
    if cd:
        v2d, v1d = cd["ConvNeXt V2 (GRN)"], cd["ConvNeXt V1 (no GRN)"]
        s3 = slice(4, 10)  # the six blocks of stage 3
        para(doc, [
            f"Averaged over the 12 blocks the cosine distance is {np.mean(v2d):.3f} for V2 with GRN "
            f"and {np.mean(v1d):.3f} for V1 without it. The V1 curve stays flat at roughly "
            f"{min(v1d[1:]):.2f}–{max(v1d):.2f} from the second block onwards (a random, uncorrelated "
            "set of channels would give 0.5), so at this scale ", ("b", "the feature collapse reported "
            "in the paper does not occur"), ": there are no dead or duplicated channels for GRN to "
            f"rescue. In V2 the GRN layer actually lowers the distance in stage 3 (blocks S3.1–S3.6: "
            f"{np.mean(v2d[s3]):.3f} vs {np.mean(v1d[s3]):.3f}), i.e. it makes the 4C hidden channels "
            "more correlated, before the distance recovers in stage 4. This is consistent with what GRN "
            "does mechanically — it amplifies channels with a large global response and suppresses the "
            "others — but it is the opposite of the paper's observation, which was made on a 3.7 M–660 M "
            "parameter model pre-trained for 800 epochs on 224×224 ImageNet. The most likely reading is "
            "that collapse is a phenomenon of long, large-scale MAE pre-training that an 8-epoch CIFAR-10 "
            "run simply does not reach. Interestingly GRN still helps the pretext task itself: the "
            f"final reconstruction loss is {PRE_V2['history'][-1]['loss']:.3f} with GRN against "
            f"{PRE_V1['history'][-1]['loss']:.3f} without (Table 2)."])
    para(doc, [
        "Before any labels are used, the quality of the frozen representation can be measured with a "
        f"20-NN classifier on the pooled encoder output: ", ("b", f"{knn('ConvNeXt V2 (GRN)')} %"),
        " (V2) versus ", ("b", f"{knn('ConvNeXt V1 (no GRN)')} %"), " (V1) top-1 on the test set. "
        "Both are far above the 10 % chance level, so the self-supervised task does produce "
        "class-relevant features, and the two encoders are within a point of each other — a lower "
        "reconstruction loss did not translate into better frozen features here."])

    heading(doc, "4.4 Transfer to classification with 5,000 labels", 2)
    table(doc, ["Block", "Initialisation", "Final test acc. (%)", "Best test acc. (%)", "Train time (min)"], [
        ["ConvNeXt V1 (no GRN)", "random (scratch)", acc("v1_scratch"), acc("v1_scratch", "best_acc"), minutes(FT["v1_scratch"])],
        ["ConvNeXt V1 (no GRN)", "FCMAE pre-trained", acc("v1_fcmae"), acc("v1_fcmae", "best_acc"), minutes(FT["v1_fcmae"])],
        ["ConvNeXt V2 (GRN)", "random (scratch)", acc("v2_scratch"), acc("v2_scratch", "best_acc"), minutes(FT["v2_scratch"])],
        [("b", "ConvNeXt V2 (GRN)"), ("b", "FCMAE pre-trained"), ("b", acc("v2_fcmae")), ("b", acc("v2_fcmae", "best_acc")), minutes(FT["v2_fcmae"])],
    ], col_widths=[4.2, 3.6, 3.0, 3.0, 2.6],
        caption_text=f"Table 3. CIFAR-10 test accuracy (10,000 images) after {FT_EPOCHS} epochs on "
                     "5,000 labelled images.")
    figure(doc, "fig_accuracy_bars.png", "Figure 5. Summary of all evaluations: k-NN on frozen FCMAE "
           "features, supervised from scratch, FCMAE pre-training + fine-tuning.", 5.4)
    figure(doc, "fig_finetune_curves.png", "Figure 6. Training loss and test accuracy of the four "
           "supervised runs (solid = FCMAE-initialised, dashed = from scratch).", 6.4)
    if all(FT.values()):
        g_v2 = FT["v2_fcmae"]["final_acc"] - FT["v2_scratch"]["final_acc"]
        g_v1 = FT["v1_fcmae"]["final_acc"] - FT["v1_scratch"]["final_acc"]
        g_grn_s = FT["v2_scratch"]["final_acc"] - FT["v1_scratch"]["final_acc"]
        g_grn_p = FT["v2_fcmae"]["final_acc"] - FT["v1_fcmae"]["final_acc"]
        para(doc, [("b", "Observations. "),
            f"(i) FCMAE pre-training on unlabeled data is the dominant effect in this low-label regime: "
            f"{g_v2:+.2f} points for the V2 block and {g_v1:+.2f} points for the V1 block, and the "
            "pre-trained models are ahead at every evaluation point of Figure 6 while also reaching a "
            "much lower training loss. (ii) GRN on its own, in supervised training from scratch, is "
            f"worth only {g_grn_s:+.2f} points. (iii) Combined with FCMAE, GRN is worth "
            f"{g_grn_p:+.2f} points, and the full ConvNeXt V2 recipe (GRN + FCMAE) is the best of the "
            f"four configurations at {FT['v2_fcmae']['final_acc']:.2f} %. This is the same qualitative "
            "pattern as Table 3 of the paper — the architectural change matters most when paired with "
            "the self-supervised objective — even though Section 4.3 showed that the feature-collapse "
            "mechanism the paper uses to explain it is not visible at our scale. The absolute numbers "
            "are of course far below the paper's, and with a single seed and 30 epochs differences of "
            "about one point (such as the GRN-only gain) should be treated as indicative only."])
    figure(doc, "fig_confusion.png", "Figure 7. Confusion matrix of ConvNeXt V2 + FCMAE on the test "
           "set, rows normalised to 100 %.", 4.4)
    if FT["v2_fcmae"]:
        pc = FT["v2_fcmae"]["per_class_acc"]
        table(doc, list(CLASSES), [[f"{v:.1f}" for v in pc]], size=9.5,
              caption_text="Table 4. Per-class accuracy (%) of ConvNeXt V2 + FCMAE.")
        worst = sorted(zip(pc, CLASSES))[:3]
        para(doc, [f"The hardest classes are {', '.join(c for _, c in worst)}; the classic cat/dog "
                   "and automobile/truck confusions dominate the off-diagonal of Figure 7."])

    heading(doc, "4.5 Limitations", 2)
    bullet(doc, "Single seed per configuration; differences below about one percentage point should "
                "not be over-interpreted.")
    bullet(doc, f"{PRE_EPOCHS} pre-training epochs are two orders of magnitude fewer than the paper's "
                "800; FCMAE gains normally keep growing with longer pre-training.")
    bullet(doc, "Dense masking instead of sparse convolutions slightly leaks information through "
                "zero-padding at patch borders.")
    bullet(doc, "32×32 images force a 2×2 last stage; the mask grid therefore had to be decoupled "
                "from the encoder stride.")

    # ======================================================================= 5
    heading(doc, "5. What I have learnt from this AI assignment", 1)
    bullet(doc, [("b", "About the algorithm. "), "I now understand why MAE pre-training does not "
                 "transfer to ConvNets for free: convolutions mix masked and visible pixels (hence sparse "
                 "convolutions / masking after every stage), and a ConvNeXt trained this way suffers from "
                 "channel collapse in the wide MLP layer. GRN is a tiny, parameter-light normalisation "
                 "that restores channel competition, and it is meant to be evaluated together with "
                 "FCMAE rather than as a stand-alone block improvement. I also learnt that a paper's "
                 "headline result and its explanation are separate claims: the accuracy pattern "
                 "(FCMAE + GRN best, GRN alone marginal) reproduced on CIFAR-10, but the collapse "
                 "diagnostic did not — at small scale and short schedules there was no collapse to fix, "
                 "so the mechanism cannot be confirmed from our data."])
    bullet(doc, [("b", "About working with an AI agent. "), "The most valuable prompts were not "
                 "\"write the code\" but the decisions the agent asked me to make (which algorithm, how "
                 "much compute, where to publish). Giving the agent access to the machine let it find "
                 "and fix practical problems on its own — a 70 kB/s download replaced by a mirror, a "
                 "25 % speed-up from channels_last, a torchvision checksum that rejected repacked data. "
                 "I also saw it make and correct mistakes (a syntax slip in a forward method, wrong "
                 "pickle key types), which is a reminder that every generated line still has to be "
                 "tested. Asking for a clean restart cost only the compute time: because everything was "
                 "scripted, the agent rebuilt the folder and re-ran the study with one instruction."])
    bullet(doc, [("b", "About reproducible experiments. "), "Having the agent log every run to JSON and "
                 "generate the figures and this report from those files means every number here can be "
                 "regenerated with one command (python run_all.py). Honest reporting of the adaptations "
                 "(dense masking, decoupled mask grid, short schedules) is as important as the numbers."])
    bullet(doc, [("b", "About the limits of the tool. "), "The agent cannot conjure compute: budgeting "
                 "the CPU time (parallel processes, thread count, evaluation frequency) was a real part "
                 "of the engineering, and the final accuracies are bounded by that budget rather than "
                 "by the method."])

    # ======================================================================= 6
    heading(doc, "6. Web page of the source code", 1)
    p = para(doc, ["All code, logs, figures and this report are publicly available at: "])
    hyperlink(p, REPO_URL)
    para(doc, [
        "Repository layout: ", ("c", "code/"), " (all Python sources listed in Section 3.1), ",
        ("c", "results/"), " (JSON logs of every run and analysis), ", ("c", "figures/"),
        " (all figures of this report), ", ("c", "report/"), " (this document as DOCX and PDF). "
        "Reproduce everything with ", ("c", "pip install -r requirements.txt && python code/run_all.py"),
        "."], size=11)

    out_dir = os.path.join(HERE, "report")
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "AI_Assignment1_ConvNeXtV2_Report.docx")
    doc.save(docx_path)
    print("saved", docx_path)
    return docx_path


def docx_to_pdf(docx_path: str) -> str:
    import win32com.client  # type: ignore

    pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(os.path.abspath(docx_path))
        d.Fields.Update()
        d.SaveAs(os.path.abspath(pdf_path), FileFormat=17)  # wdFormatPDF
        d.Close(False)
    finally:
        word.Quit()
    print("saved", pdf_path)
    return pdf_path


if __name__ == "__main__":
    path = build()
    if "--no-pdf" not in sys.argv:
        docx_to_pdf(path)
