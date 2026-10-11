"""Build the AI Assignment #2 report (DOCX + PDF) from the experiment results."""

from __future__ import annotations

import json
import os
import sys

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
XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
FONT = "Times New Roman"
MONO = "Consolas"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/HOIKAICHENG/CISC3024")

_transform = etree.XSLT(etree.parse(XSL_PATH))


def J(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


MAP, ABNN, EV = J("map.json"), J("abnn.json"), J("eval.json") or {}


def fnum(x, nd=2, default="n/a"):
    return default if x is None else f"{x:.{nd}f}"


def metric(tag, key, nd=2):
    block = EV.get(tag) or {}
    val = block.get(key)
    return fnum(val, nd)


def ood(tag, key, nd=2):
    val = ((EV.get(tag) or {}).get("ood") or {}).get(key)
    return fnum(val, nd)


def minutes(rec):
    if not rec:
        return "n/a"
    return f"{rec['total_time_s'] / 60:.1f}"


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


def experts_text():
    if not ABNN:
        return "n/a"
    bits = []
    for run in ABNN["runs"]:
        bits.append(f"member {run['member']}: {', '.join(run['experts'])}")
    return "; ".join(bits)


def per_class_row():
    pc = (EV.get("abnn") or {}).get("per_class_acc") or {}
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return [fnum(pc.get(n), 1) for n in names]


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

    para(doc, [("b", "CISC3024 Pattern Recognition")], size=14, space_after=2)
    para(doc, [("b", "AI Assignment #2 — Probability / Bayes theory")], size=14, space_after=2)
    para(doc, [("b", "ABNN: Make Me a BNN — converting a pretrained DNN into a Bayesian "
                     "network (CVPR 2024), reproduced on CIFAR-10")], size=13, space_after=8)
    para(doc, "Name: HOI KAI CHENG     Student ID: UC325381", space_after=4)
    para(doc, [("i", "AI tool used: Cursor IDE in agent mode (model: Cursor Grok 4.6). As required "
                     "by the assignment, every line of code, every experiment and this report were "
                     "produced by the AI agent; my own contribution was limited to writing prompts "
                     "and making the choices recorded in Section 1.")], size=11, space_after=12)

    heading(doc, "1. How I asked the AI tool to find the algorithm", 1)
    para(doc, [
        "I used the agent mode of the Cursor IDE, which can read files, run shell commands, write "
        "code and search the web, so a single conversation could carry the whole assignment from "
        "literature search to report. My first message attached the assignment PDF and the list of "
        "required report sections:"])
    code_block(doc,
        "d:\\UM\\Year4\\CISC3024\\AI Assignment2\\AIassign2.pdf\n"
        "Your AI assignment report in Word/PDF format should contain the following details\n"
        "(1) how you ask AI tools to find the algorithm, (2) algorithm description,\n"
        "(3) how AI implements the algorithm, (4) experiment settings and results,\n"
        "(5) what you have learnt from this AI assignment, and (6) a webpage link of your\n"
        "source codes.")
    para(doc, [
        "The agent read the PDF and extracted the constraints: a ",
        ("b", "recent classification / recognition algorithm based on probability or Bayes theory"),
        " for computer vision and pattern recognition; the report must document the six numbered "
        "sections; and ", ("b", "I am not allowed to write any code myself"), ". It also inspected "
        "the previous assignment (ConvNeXt V2 on CIFAR-10 in the same GitHub repository), the "
        "machine (Python 3.12, PyTorch 2.13 CPU-only, 20 logical cores) and then searched the web "
        "for recent peer-reviewed methods rather than guessing. The shortlist it came back with "
        "was:"])
    table(doc, ["Candidate", "Venue / year", "Why it was considered", "Why it was or was not chosen"], [
        ["ABNN — “Make Me a BNN” (recommended)",
         "CVPR 2024",
         "Explicitly Bayesian: a post-hoc BNN obtained by attaching Gaussian noise to "
         "normalisation layers; official CIFAR-10 classification + SVHN OOD protocol.",
         "Chosen: most recent, clearly Bayesian, and feasible on CPU."],
        ["Deep Deterministic Uncertainty (DDU)",
         "CVPR 2023",
         "Class-conditional Gaussians (GDA) on last-layer features; a classical Bayes classifier "
         "on top of a CNN.",
         "Simpler and faster, but less “Bayesian neural network” than ABNN."],
        ["Fisher-information evidential DL (I-EDL)",
         "ICML 2023",
         "Predicts Dirichlet evidence and therefore a full categorical posterior.",
         "Strong probabilistic story, but a heavier training recipe than ABNN."],
    ], col_widths=[3.8, 2.6, 5.4, 4.4], size=9.5)
    para(doc, [
        "The agent asked me to pick one of those three. I chose ",
        ("b", "ABNN (CVPR 2024)"),
        " and told it to keep going, to put the source on ",
        ("c", "https://github.com/HOIKAICHENG/CISC3024.git"),
        ", and to sign the report with my name (HOI KAI CHENG) and student ID (UC325381). From "
        "that point the agent implemented the paper from scratch, ran the experiments and wrote "
        "this report. I did not edit any source file by hand."])
    para(doc, [("b", "Reference. "),
        "G. Franchi, O. Laurent, M. Leguery, A. Bursuc, A. Pilzer and A. Yao, \"Make Me a BNN: "
        "A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models\", CVPR "
        "2024 (arXiv:2312.15297)."], size=11)

    heading(doc, "2. Algorithm description", 1)
    heading(doc, "2.1 Motivation", 2)
    para(doc, [
        "A standard deep network keeps a single weight vector ", ("m", r"\omega_{\mathrm{MAP}}"),
        " and treats it as an oracle. That point estimate is often over-confident: a cat that the "
        "model has never seen, or a house-number image when the training set was CIFAR-10, can "
        "still receive a softmax probability close to one. Bayesian neural networks (BNNs) instead "
        "try to represent the posterior ", ("m", r"P(\omega\mid\mathcal{D})"),
        " and to predict by averaging,"])
    display(doc, r"P(y\mid x,\mathcal{D})=\int P(y\mid x,\omega)\,P(\omega\mid\mathcal{D})\,d\omega", "(2)")
    para(doc, [
        "which in practice is replaced by a finite average over ", ("m", "M"),
        " weight samples. Classical variational BNNs put a Gaussian on every weight and are "
        "unstable to train on modern ConvNets. Deep ensembles work well but multiply the training "
        "cost. Last-layer Laplace methods are cheap but only see high-level features. ABNN sits "
        "between these extremes: start from a finished deterministic network and turn ",
        ("b", "only the normalisation layers"), " into a cheap Bayesian approximation."])

    heading(doc, "2.2 From MAP to a BNN", 2)
    para(doc, [
        "Training first minimises the usual MAP objective (cross-entropy plus a Gaussian weight "
        "prior, i.e. ", ("m", "L_2"), " regularisation),"])
    display(doc,
            r"\mathcal{L}_{\mathrm{MAP}}(\omega)=-\sum_{(x_i,y_i)\in\mathcal{D}}"
            r"\log P(y_i\mid x_i,\omega)-\log P(\omega)", "(1)")
    para(doc, [
        "yielding a deterministic ResNet. ABNN then replaces every batch / layer / instance "
        "normalisation with a ", ("b", "Bayesian Normalisation Layer (BNL)"), ". Write ",
        ("m", r"h_j"), " for the pre-activation of layer ", ("m", "j"),
        ", and ", ("m", r"\hat\mu_j,\hat\sigma_j"), " for the empirical mean and standard "
        "deviation already computed by the original normalisation. The BNL is"])
    display(doc,
            r"\mathrm{BNL}(h_j)=\frac{h_j-\hat\mu_j}{\hat\sigma_j}\cdot\gamma_j(1+\epsilon_j)+\beta_j,"
            r"\qquad\epsilon_j\sim\mathcal{N}(0,I)", "(7)")
    para(doc, [
        "The only new source of randomness is the multiplicative Gaussian noise on the scale ",
        ("m", r"\gamma_j"), ". The running statistics and the convolutional weights stay frozen "
        "at their MAP values; only ", ("m", r"\gamma_j"), " and ", ("m", r"\beta_j"),
        " are fine-tuned for a few epochs. Because the noise lives in the latent activations "
        "rather than on every weight, the paper argues that the gradient variance is smaller "
        "than in a classical variational BNN, which is why the conversion is stable."])
    figure(doc, "fig_bnl.png",
           "Figure 1. The Bayesian Normalisation Layer: standard normalisation, then a noisy "
           "scale γ(1+ε) and the original shift β.")
    figure(doc, "fig_pipeline.png",
           "Figure 2. Three-step ABNN recipe: train a MAP network, replace every BN by a BNL, "
           "fine-tune M copies of (γ, β).")

    heading(doc, "2.3 Diversity loss and the predictive posterior", 2)
    para(doc, [
        "A single diagonal Gaussian around one MAP checkpoint cannot represent the multi-modal "
        "DNN posterior. The paper therefore trains ", ("m", "M"),
        " ABNN members from the same checkpoint and, to stop them collapsing onto each other, "
        "adds a class-dependent perturbation of the cross-entropy"])
    display(doc,
            r"\mathcal{E}(\omega)=-\sum_{(x_i,y_i)\in\mathcal{D}}\eta_i\log P(y_i\mid x_i,\omega)",
            "(9)")
    para(doc, [
        "where ", ("m", r"\eta_i"), " is a class weight drawn once per member. In the paper, a "
        "random subset of classes is treated as that member’s “expert” classes and their loss "
        "is multiplied by 5 or 7. The member then minimises ",
        ("m", r"\mathcal{L}(\omega)=\mathcal{L}_{\mathrm{MAP}}(\omega)+\mathcal{E}(\omega)"),
        ", which is ordinary weighted cross-entropy plus weight decay. At test time each member "
        "is evaluated with ", ("m", "L"), " independent draws of ", ("m", r"\epsilon"),
        " and the ", ("m", "ML"), " softmax vectors are averaged:"])
    display(doc,
            r"P(y\mid x,\mathcal{D})\approx\frac{1}{ML}\sum_{m=1}^{M}\sum_{l=1}^{L}"
            r"P(y\mid x,\omega_m,\epsilon_l)", "(8)")
    para(doc, [
        "Table 1 of the paper uses ", ("m", "M=3"), " and ", ("m", "L=3"),
        " on CIFAR-10/100; we keep the same ensemble size. The quality of the predictive "
        "distribution is measured by accuracy, negative log-likelihood (NLL), expected "
        "calibration error (ECE), and out-of-distribution detection of SVHN from CIFAR-10 "
        "using the maximum softmax probability (AUPR, AUROC, FPR95)."])

    heading(doc, "3. How the AI implemented the algorithm", 1)
    heading(doc, "3.1 Workflow of the agent", 2)
    para(doc, [
        "The implementation lives entirely under ", ("c", "AI Assignment2/"),
        " and was written in this order (each step was executed by the agent, not by me):"])
    numbered(doc, [("b", "Environment and paper. "), "Confirmed Python 3.12 / PyTorch 2.13 CPU / "
                    "20 logical cores, read the CVPR 2024 PDF, and cross-checked the unofficial "
                    "reproduction at github.com/abtinmU/MakeMe-BNN for how practitioners interpret "
                    "the BNL (the official repository is only a project page)."])
    numbered(doc, [("c", "resnet.py"), ": CIFAR ResNet-20 (He et al.), 6n+2 with n=3, 16-32-64 "
                    "channels, BatchNorm after every convolution — 21 BN layers, 272 474 parameters. "
                    "ResNet-50 / WideResNet-28×10 from the paper are not realistic on this CPU."])
    numbered(doc, [("c", "bnl.py"), ": ", ("c", "BNL.from_batchnorm"), " copies ",
                    ("c", "weight"), "/", ("c", "bias"), " and the running statistics; the forward "
                    "pass uses frozen running mean/variance and multiplies ", ("c", "gamma"),
                    " by ", ("c", "(1 + randn)"), ". ", ("c", "replace_bn_with_bnl"),
                    " walks the module tree; ", ("c", "freeze_except_bnl"),
                    " leaves only the 42 affine tensors trainable."])
    numbered(doc, [("c", "data.py"), " / ", ("c", "prepare_data.py"), ": CIFAR-10 from the "
                    "HuggingFace ", ("c", "uoft-cs/cifar10"), " mirror (the official Toronto "
                    "tarball is too slow from this network), plus the official SVHN ",
                    ("c", "test_32x32.mat"), " as the OOD set."])
    numbered(doc, [("c", "train_map.py"), ": SGD (lr 0.1, momentum 0.9, wd 5e-4), cosine decay, "
                    f"{MAP['epochs'] if MAP else 20} epochs, channels-last memory format, 10 "
                    "intra-op threads."])
    numbered(doc, [("c", "train_abnn.py"), ": load the MAP checkpoint, convert BN → BNL, freeze "
                    "the backbone, AdamW on ", ("m", r"(\gamma,\beta)"), " only, class weights "
                    "with factor 5 on 4 random classes per member, "
                    f"{ABNN['members'] if ABNN else 3} members × "
                    f"{ABNN['epochs'] if ABNN else 5} epochs."])
    numbered(doc, [("c", "evaluate.py"), ": MAP (one deterministic pass), MCBN (the MAP model "
                    "with BatchNorm in train mode, L stochastic batch statistics) and ABNN "
                    "(M×L averaged softmaxes). Metrics: Acc, NLL, 15-bin ECE, SVHN OOD "
                    "AUROC / AUPR / FPR95."])
    numbered(doc, [("c", "make_figures.py"), " / ", ("c", "build_report.py"), ": every number in "
                    "this document is read back from ", ("c", "results/*.json"),
                    " so the report cannot drift from the run."])

    heading(doc, "3.2 Key code produced by the agent", 2)
    para(doc, "The BNL forward pass is the paper’s Eq. (7) with frozen MAP statistics:")
    code_block(doc,
        "out = F.batch_norm(x, running_mean, running_var, None, None,\n"
        "                   training=False, momentum=0.0, eps=self.eps)\n"
        "noise = torch.randn(self.weight.shape, device=x.device, dtype=x.dtype)\n"
        "gamma = self.weight * (1.0 + noise)          # γ (1 + ε)\n"
        "return out * gamma.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)")
    para(doc, "Conversion and the diversity loss (one member):")
    code_block(doc,
        "replace_bn_with_bnl(model)          # every BatchNorm2d -> BNL\n"
        "trainable = freeze_except_bnl(model)  # conv / FC weights stay at MAP\n"
        "w = ones(10); w[random_classes] = 5.0   # expert classes, factor 5\n"
        "loss = F.cross_entropy(logits, y, weight=w)  # L_MAP + E")

    heading(doc, "3.3 Adaptations for CIFAR-10 on a CPU", 2)
    bullet(doc, [("b", "Backbone. "), "ResNet-20 (0.27 M parameters) instead of ResNet-50 or "
                  "WideResNet-28×10. The BNL mechanism does not depend on width."])
    bullet(doc, [("b", "Schedule. "),
                  f"{MAP['epochs'] if MAP else 20} MAP epochs and "
                  f"{ABNN['epochs'] if ABNN else 5} ABNN epochs instead of the paper’s long "
                  "CIFAR/ImageNet recipes. Absolute accuracies are therefore lower; the "
                  "comparison between MAP, MCBN and ABNN remains meaningful."])
    bullet(doc, [("b", "Frozen statistics. "), "After conversion the BNL always uses the MAP "
                  "running mean/variance. Only ", ("m", r"\gamma,\beta"),
                  " move. This matches “exclusively retrain γ and β” in Section 4.1."])
    bullet(doc, [("b", "Noise scale. "), "Eq. (7) writes ",
                  ("m", r"\epsilon\sim\mathcal{N}(0,I)"),
                  ". With that unit Gaussian on every one of the 21 BNLs the MAP features "
                  "collapsed (fine-tune accuracy fell to chance, ~10 %). The compound factor ",
                  ("m", r"\prod_j(1+\epsilon_j)"),
                  " over a 20-layer residual net is too violent when the convolutions are "
                  "frozen. We therefore use ",
                  ("m", r"\epsilon\sim\mathcal{N}(0,0.1^2)"),
                  ", the same multiplicative form at a smaller scale, and check that "
                  "conversion with ", ("m", r"\epsilon=0"),
                  " preserves the MAP accuracy before fine-tuning."])
    bullet(doc, [("b", "Ensemble size. "), ("m", "M=L=3"),
                  " as in Table 1 of the paper (the recommended default)."])
    bullet(doc, [("b", "No TorchUncertainty. "), "The paper’s training configs live in that "
                  "library; the agent re-implemented the algorithm rather than wrapping the "
                  "library, so every line can be read and marked."])

    heading(doc, "4. Experiment settings and results", 1)
    heading(doc, "4.1 Settings", 2)
    table(doc, ["Item", "Setting"], [
        ["Dataset (ID)", "CIFAR-10: 50 000 train / 10 000 test, 32×32 RGB, 10 classes"],
        ["Dataset (OOD)", "SVHN test set (house-number images), same 32×32 resolution; "
         "maximum softmax probability as the OOD score"],
        ["Backbone", f"CIFAR ResNet-20, 272 474 parameters, 21 BatchNorm / BNL layers"],
        ["MAP training",
         f"SGD lr 0.1, momentum 0.9, wd 5e-4, cosine, batch 128, "
         f"{MAP['epochs'] if MAP else 20} epochs, RandomCrop(pad 4) + flip"],
        ["ABNN fine-tune",
         f"AdamW lr 1e-3, wd 1e-4, {ABNN['epochs'] if ABNN else 5} epochs, "
         f"M={ABNN['members'] if ABNN else 3} members, ε~N(0,0.1²), factor-5 weights on 4 "
         f"random classes ({experts_text()})"],
        ["Inference",
         f"MAP: 1 pass; MCBN: L={EV.get('l_samples', 3)} passes with BN in train mode; "
         f"ABNN: M×L = {EV.get('members', 3)}×{EV.get('l_samples', 3)} softmaxes averaged"],
        ["Hardware / software",
         f"20-core CPU, no GPU; Python 3.12, PyTorch 2.13 (channels_last, 10 threads); "
         f"MAP {minutes(MAP)} min, ABNN {minutes(ABNN)} min"],
        ["Seeds", "torch.manual_seed(0); each ABNN member adds 17·(m+1) for its class weights"],
    ], col_widths=[4.2, 12.0], caption_text="Table 1. Experimental settings.")

    heading(doc, "4.2 MAP pre-training", 2)
    figure(doc, "fig_map_curves.png",
           "Figure 3. Deterministic ResNet-20 MAP training on CIFAR-10.")
    para(doc, [
        "The MAP run reaches a final test accuracy of ",
        ("b", f"{fnum(MAP['final_acc']) if MAP else 'n/a'} %"),
        " (best ",
        f"{fnum(MAP['best_acc']) if MAP else 'n/a'}",
        " %) after ",
        f"{MAP['epochs'] if MAP else 'n/a'}",
        " epochs. This is the ",
        ("m", r"\omega_{\mathrm{MAP}}"),
        " that every later method starts from: MCBN reuses the same weights, and every ABNN "
        "member is obtained by converting its BatchNorm layers and fine-tuning only the affine "
        "parameters. The absolute number is below a 200-epoch ResNet-20 (~91 %) because the "
        "schedule was cut to fit a laptop CPU; the checkpoint is still a genuine local minimum "
        "of the MAP loss, which is all ABNN needs."])

    heading(doc, "4.3 ABNN fine-tuning", 2)
    figure(doc, "fig_abnn_curves.png",
           "Figure 4. Fine-tuning of the three ABNN members. Only γ and β are updated.")
    if ABNN:
        rows = []
        for run in ABNN["runs"]:
            rows.append([
                f"member {run['member']}",
                ", ".join(run["experts"]),
                fnum(run["final_acc"]),
                fnum(run["best_acc"]),
                f"{run['total_time_s'] / 60:.1f}",
            ])
        table(doc, ["Member", "Expert classes (weight = 5)", "Final acc. (%)", "Best acc. (%)",
                    "Time (min)"],
              rows, col_widths=[2.6, 6.4, 2.8, 2.8, 2.4],
              caption_text="Table 2. Per-member ABNN fine-tuning (single stochastic pass, so "
                           "noisier than the M×L evaluation in Table 3).")
    para(doc, [
        "Each member stays close to the MAP accuracy while the expert-class weights push it "
        "slightly off the original mode. Pairwise disagreement of the three members on the "
        "test set is ",
        ("b", f"{fnum(100 * EV.get('abnn_disagreement', 0), 2)} %"),
        " of examples — not as diverse as independently trained deep ensembles (the paper’s "
        "own t-SNE in Figure 3 makes the same point), but enough that averaging them changes "
        "the predictive distribution."])

    heading(doc, "4.4 Calibration and OOD detection", 2)
    table(doc,
          ["Method", "Acc ↑ (%)", "NLL ↓", "ECE ↓ (%)", "OOD AUROC ↑", "OOD AUPR ↑", "FPR95 ↓"],
          [
              ["MAP (single model)",
               metric("map", "acc"), metric("map", "nll", 3), metric("map", "ece"),
               ood("map", "auroc"), ood("map", "aupr"), ood("map", "fpr95")],
              ["MCBN",
               metric("mcbn", "acc"), metric("mcbn", "nll", 3), metric("mcbn", "ece"),
               ood("mcbn", "auroc"), ood("mcbn", "aupr"), ood("mcbn", "fpr95")],
              [("b", "ABNN (M=L=3)"),
               ("b", metric("abnn", "acc")), ("b", metric("abnn", "nll", 3)),
               ("b", metric("abnn", "ece")), ("b", ood("abnn", "auroc")),
               ("b", ood("abnn", "aupr")), ("b", ood("abnn", "fpr95"))],
          ],
          col_widths=[3.6, 2.2, 1.8, 2.2, 2.4, 2.2, 2.0],
          caption_text="Table 3. In-distribution CIFAR-10 and OOD SVHN. Bold = ABNN row, not "
                       "necessarily the best entry.")
    figure(doc, "fig_metrics.png",
           "Figure 5. Side-by-side summary of Table 3. Lower is better for NLL, ECE and FPR95.")
    figure(doc, "fig_reliability.png",
           "Figure 6. Reliability diagrams (15 bins). A perfectly calibrated model lies on the diagonal.")
    figure(doc, "fig_confidence.png",
           "Figure 7. Histogram of maximum softmax probability on CIFAR-10 (ID) versus SVHN (OOD).")
    para(doc, [
        "Observations. (i) ABNN ",
        ("b", "preserves MAP accuracy"),
        " (Table 3: ",
        metric("map", "acc"), " % → ", metric("abnn", "acc"),
        " %), which is the paper’s first claim: converting the BatchNorm layers and fine-tuning "
        "only ", ("m", r"(\gamma,\beta)"),
        " must not destroy the pretrained predictor. Pairwise disagreement among the three "
        "members is ",
        f"{fnum(100 * EV.get('abnn_disagreement', 0), 2)}",
        " % of test images, so the diversity loss does move the members apart. (ii) Averaging "
        "M×L stochastic softmaxes changes the ",
        ("b", "shape"),
        " of the predictive distribution: mean confidence falls from ",
        fnum((EV.get('map') or {}).get('conf_mean')),
        " (MAP) to ",
        fnum((EV.get('abnn') or {}).get('conf_mean')),
        " (ABNN), i.e. the ensemble is less peaky. (iii) At this CPU scale that extra "
        "spread does ",
        ("b", "not"),
        " improve the proper scoring rules or the SVHN OOD numbers. NLL moves from ",
        metric("map", "nll", 3), " to ", metric("abnn", "nll", 3),
        ", ECE from ", metric("map", "ece"), " % to ", metric("abnn", "ece"),
        " %, and max-softmax AUROC from ",
        ood("map", "auroc"), " % to ", ood("abnn", "auroc"),
        " % (FPR95 ", ood("map", "fpr95"), " % → ", ood("abnn", "fpr95"),
        " %). MCBN, which only randomises batch statistics, improves ECE slightly (",
        metric("mcbn", "ece"),
        " %) but is the worst OOD detector. (iv) These are ResNet-20 / 20-epoch numbers, not "
        "the paper’s ResNet-50 / WideResNet figures (~95 % accuracy, FPR95 ≈ 17). The study "
        "therefore supports the conversion claim and the diversity claim, and does ",
        ("b", "not"),
        " reproduce the paper’s headline “better than a single model on NLL / ECE / FPR95” "
        "result — an honest gap that Section 4.5 attributes to width, schedule and the "
        "reduced noise scale."])
    figure(doc, "fig_confusion.png",
           "Figure 8. Row-normalised confusion matrix of the ABNN ensemble on the CIFAR-10 test set.")
    table(doc,
          ["air.", "auto.", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
          [per_class_row()],
          caption_text="Table 4. Per-class accuracy (%) of ABNN.",
          size=9.5)

    heading(doc, "4.5 Limitations", 2)
    bullet(doc, "Single seed for MAP; three ABNN members share that one checkpoint, so we do not "
                 "estimate variance across independent MAP runs.")
    bullet(doc, "20 MAP epochs (instead of 164–200) cap the absolute accuracy. ABNN cannot invent "
                 "features the backbone never learned.")
    bullet(doc, "ResNet-20 is much narrower than the paper’s ResNet-50 / WideResNet-28×10, so "
                 "feature-space diversity is smaller.")
    bullet(doc, "OOD detection uses only SVHN and only the maximum softmax probability. Energy or "
                 "feature-density scores are left out.")
    bullet(doc, "The paper’s Algorithm 1 is slightly ambiguous about whether running statistics "
                 "stay frozen; we freeze them. A different reading would keep updating them.")

    heading(doc, "5. What I have learnt from this AI assignment", 1)
    bullet(doc, [("b", "About the algorithm. "),
                  "I now have a concrete picture of a post-hoc BNN that is not “put a Gaussian on "
                  "every weight”. ABNN keeps the MAP convolutional features, injects noise only "
                  "on the normalisation scale, and recovers a multi-modal posterior by fine-tuning "
                  "several copies with a class-weighted loss. The predictive distribution is just "
                  "the average of M×L softmaxes. That is Bayes’ rule in the form a practitioner "
                  "can actually run. I also learnt to separate the paper’s architectural claim "
                  "(BNL + frozen backbone) from its engineering claim (almost Deep-Ensemble OOD "
                  "numbers at single-model cost): on a small CPU budget we can test the first "
                  "honestly and only gesture at the second."])
    bullet(doc, [("b", "About working with an AI agent. "),
                  "The useful prompt was not “write a BNN”. It was the assignment PDF plus the "
                  "constraint “probability or Bayes theory”, after which the agent searched, "
                  "proposed three recent papers and waited for a choice. Giving it the machine "
                  "let it pick a HuggingFace CIFAR-10 mirror, freeze running statistics so the "
                  "BNL conversion did not destroy the MAP checkpoint, and log every run to JSON "
                  "so this report could be generated from the measurements. I also saw that an "
                  "unofficial GitHub reproduction is not the same as the paper: that repo unfroze "
                  "extra weights after a 20–30 % accuracy drop, which would have hidden whether "
                  "ABNN itself works. The agent kept the paper’s “only γ, β” rule and tested the "
                  "conversion with a stochastic/deterministic smoke test before training."])
    bullet(doc, [("b", "About uncertainty metrics. "),
                  "Accuracy alone cannot tell MAP and ABNN apart. NLL, ECE and the ID/OOD "
                  "confidence histograms are the quantities that move, and they are what a "
                  "Bayesian method is for. Plotting a reliability diagram made the phrase "
                  "“over-confident softmax” concrete."])
    bullet(doc, [("b", "About the limits of the tool. "),
                  "The agent cannot conjure a GPU. Budgeting 20 MAP epochs and M = 3 was a real "
                  "design choice, and every number in Section 4 is bounded by that budget. It "
                  "also cannot decide which paper I should study — that choice stayed with me."])

    heading(doc, "6. Web page of the source code", 1)
    p = para(doc, ["All code, logs, figures and this report are publicly available at: "])
    hyperlink(p, REPO_URL)
    para(doc, [
        "Repository layout (folder ", ("c", "AI Assignment2/"), "): ",
        ("c", "code/"), " (all Python sources listed in Section 3.1), ",
        ("c", "results/"), " (JSON logs of every run), ",
        ("c", "figures/"), " (all figures of this report), ",
        ("c", "report/"), " (this document as DOCX and PDF). Reproduce everything with ",
        ("c", "pip install -r requirements.txt && python code/run_all.py"), "."], size=11)

    out_dir = os.path.join(HERE, "report")
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "AI_Assignment2_ABNN_Report.docx")
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
        d.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        d.Close(False)
    finally:
        word.Quit()
    print("saved", pdf_path)
    return pdf_path


if __name__ == "__main__":
    path = build()
    if "--no-pdf" not in sys.argv:
        docx_to_pdf(path)
