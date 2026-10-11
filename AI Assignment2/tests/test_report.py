import json
from pathlib import Path
import shutil
import tempfile

import pytest
from docx import Document

import build_report
from build_report import SOURCE_URL, build_report as generate_report
from code.make_figures import make_all_figures


ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "architecture.png",
    "training_curves.png",
    "reliability.png",
    "ood_entropy.png",
    "metrics_comparison.png",
    "uncertainty_examples.png",
)


def _document_text(document: Document) -> str:
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    cells = [
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    ]
    return "\n".join((*paragraphs, *cells))


def _write_valid_docx(path: Path, text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    document.save(path)
    return path.read_bytes()


def _write_valid_pdf(path: Path, marker: bytes) -> bytes:
    content = b"%PDF-1.7\n" + marker + b"\n%%EOF\n"
    path.write_bytes(content)
    return content


def _stub_report_inputs(monkeypatch, source_path: Path) -> None:
    monkeypatch.setattr(
        build_report,
        "_report_source_paths",
        lambda _quick: (source_path,),
    )
    monkeypatch.setattr(
        build_report,
        "_load_report_artifacts",
        lambda _quick: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(
        build_report,
        "_compose_document",
        lambda *_args, **_kwargs: Document(),
    )


@pytest.fixture(scope="module")
def quick_report():
    with tempfile.TemporaryDirectory(
        prefix=".quick-report-",
        dir=ROOT,
    ) as directory:
        isolated_root = Path(directory)
        isolated_results = isolated_root / "results"
        isolated_code = isolated_root / "code"
        isolated_results.mkdir()
        isolated_code.mkdir()
        for name in (
            "evaluation_quick.json",
            "map_history_quick.json",
            "abnn_history_quick.json",
        ):
            shutil.copy2(ROOT / "results" / name, isolated_results / name)
        shutil.copy2(ROOT / "code" / "abnn.py", isolated_code / "abnn.py")
        make_all_figures(
            isolated_results,
            isolated_root / "figures",
            quick=True,
        )

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(build_report, "ROOT", isolated_root)
            monkeypatch.setattr(
                build_report,
                "REPORT_DIR",
                isolated_root / "report",
            )
            monkeypatch.setattr(
                build_report,
                "SOURCE_EXCERPT_FILES",
                (isolated_code / "abnn.py",),
            )
            monkeypatch.setattr(
                build_report,
                "_convert_docx_to_pdf",
                lambda _docx, pdf: _write_valid_pdf(
                    Path(pdf),
                    b"quick-report\n" * 1_000,
                ),
            )
            docx_path, pdf_path = generate_report(quick=True)
        document = Document(docx_path)
        yield docx_path, pdf_path, document, _document_text(document)


def test_quick_report_has_required_sections_and_measured_results(quick_report):
    docx_path, pdf_path, _, text = quick_report
    for heading in (
        "1. How I Asked AI Tools to Find the Algorithm",
        "2. Algorithm Description",
        "3. How AI Implemented the Algorithm",
        "4. Experiment Settings and Results",
        "5. What I Have Learnt",
        "6. Webpage Link of the Source Codes",
    ):
        assert heading in text

    assert "HOI KAI CHENG" in text
    assert "UC325381" in text
    assert "SMOKE-TEST DATA" in text
    assert "NOT THE FINAL SUBMISSION" in text
    assert "CVPR 2024 paper findings" in text
    assert "Measured assignment results" in text

    evaluation = json.loads(
        (ROOT / "results" / "evaluation_quick.json").read_text(encoding="utf-8")
    )
    for method in ("map", "abnn_single", "abnn_full"):
        classification = evaluation["methods"][method]["classification"]
        for metric in ("accuracy", "nll", "brier", "ece"):
            assert f"{classification[metric]:.4f}" in text
        for dataset in ("svhn", "cifar100"):
            for metric in ("auroc", "aupr", "fpr95"):
                value = evaluation["methods"][method]["ood"][dataset][metric]
                assert f"{value:.4f}" in text

    assert docx_path.exists() and docx_path.stat().st_size > 100_000
    assert pdf_path.exists() and pdf_path.stat().st_size > 10_000


def test_quick_report_embeds_all_figures_tables_captions_and_live_code(
    quick_report,
):
    _, _, document, text = quick_report
    assert len(document.inline_shapes) == len(FIGURE_NAMES)
    assert len(document.tables) >= 5
    for number, figure_name in enumerate(FIGURE_NAMES, start=1):
        assert f"Figure {number}." in text
        assert figure_name.removesuffix(".png").replace("_", " ") in text.lower()
    assert "Table 1." in text
    assert "References" in text
    assert "Franchi" in text

    bnl_line = "scale = scale * (1.0 + self.noise_std * torch.randn_like(scale))"
    averaging_line = "probabilities = sample_probs.mean(axis=0)"
    source = (ROOT / "code" / "abnn.py").read_text(encoding="utf-8")
    assert bnl_line in source and bnl_line in text
    assert averaging_line in source and averaging_line in text


def test_quick_report_contains_clickable_direct_source_link(quick_report):
    _, _, document, text = quick_report
    assert SOURCE_URL in text
    external_targets = {
        relationship.target_ref
        for relationship in document.part.rels.values()
        if relationship.is_external
    }
    assert SOURCE_URL in external_targets


def test_word_conversion_closes_document_and_application_on_error(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.docx"
    source.write_bytes(b"placeholder")
    target = tmp_path / "target.pdf"

    class FakeDocument:
        def __init__(self):
            self.closed = False
            self.Fields = type("Fields", (), {"Update": lambda self: None})()

        def ExportAsFixedFormat(self, *_args, **_kwargs):
            raise OSError("simulated Word export failure")

        def Close(self, _save_changes):
            self.closed = True

    class FakeWord:
        def __init__(self):
            self.document = FakeDocument()
            self.Documents = type(
                "Documents",
                (),
                {"Open": lambda _self, _path: self.document},
            )()
            self.Visible = True
            self.DisplayAlerts = 1
            self.quit_called = False

        def Quit(self):
            self.quit_called = True

    word = FakeWord()
    monkeypatch.setattr(build_report, "_dispatch_word", lambda: word)

    with pytest.raises(RuntimeError, match="Microsoft Word.*source.docx"):
        build_report._convert_docx_to_pdf(source, target)

    assert word.document.closed
    assert word.quit_called


def test_quick_build_uses_separate_names_and_preserves_final_sentinels(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("stable", encoding="utf-8")
    _stub_report_inputs(monkeypatch, source)
    monkeypatch.setattr(build_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        build_report,
        "_convert_docx_to_pdf",
        lambda _docx, pdf: _write_valid_pdf(Path(pdf), b"quick"),
    )

    final_docx = tmp_path / build_report.DOCX_NAME
    final_pdf = tmp_path / build_report.PDF_NAME
    final_docx.write_bytes(b"FINAL-DOCX-SENTINEL")
    final_pdf.write_bytes(b"FINAL-PDF-SENTINEL")

    quick_docx, quick_pdf = generate_report(quick=True)

    assert quick_docx.name == "AI_Assignment2_ABNN_Report_QUICK.docx"
    assert quick_pdf.name == "AI_Assignment2_ABNN_Report_QUICK.pdf"
    assert quick_docx.exists() and quick_pdf.exists()
    assert final_docx.read_bytes() == b"FINAL-DOCX-SENTINEL"
    assert final_pdf.read_bytes() == b"FINAL-PDF-SENTINEL"
    assert not any("report-" in path.name for path in tmp_path.iterdir())


def test_conversion_failure_preserves_existing_published_pair(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("stable", encoding="utf-8")
    _stub_report_inputs(monkeypatch, source)
    monkeypatch.setattr(build_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        build_report,
        "_convert_docx_to_pdf",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("conversion failed")),
    )

    final_docx = tmp_path / build_report.DOCX_NAME
    final_pdf = tmp_path / build_report.PDF_NAME
    old_docx = _write_valid_docx(final_docx, "old final document")
    old_pdf = _write_valid_pdf(final_pdf, b"old-final-pdf")

    with pytest.raises(RuntimeError, match="conversion failed"):
        generate_report(quick=False)

    assert final_docx.read_bytes() == old_docx
    assert final_pdf.read_bytes() == old_pdf
    assert not any("report-" in path.name for path in tmp_path.iterdir())


def test_second_file_publication_failure_restores_existing_pair(
    tmp_path, monkeypatch
):
    final_docx = tmp_path / build_report.DOCX_NAME
    final_pdf = tmp_path / build_report.PDF_NAME
    old_docx = _write_valid_docx(final_docx, "old final document")
    old_pdf = _write_valid_pdf(final_pdf, b"old-final-pdf")

    staging = tmp_path / "staging"
    staging.mkdir()
    staged_docx = staging / final_docx.name
    staged_pdf = staging / final_pdf.name
    _write_valid_docx(staged_docx, "new document")
    _write_valid_pdf(staged_pdf, b"new-pdf")

    real_replace = build_report._replace_file

    def fail_second_publish(source, destination):
        if Path(source) == staged_pdf and Path(destination) == final_pdf:
            raise OSError("simulated second-file replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(build_report, "_replace_file", fail_second_publish)

    with pytest.raises(RuntimeError, match="restored the previous pair"):
        build_report._publish_report_pair(
            staged_docx,
            staged_pdf,
            final_docx,
            final_pdf,
        )

    assert final_docx.read_bytes() == old_docx
    assert final_pdf.read_bytes() == old_pdf
    assert not staged_docx.exists()
    assert not staged_pdf.exists()
    assert not any(
        "report-backup-" in path.name for path in tmp_path.iterdir()
    )


def test_report_source_snapshot_includes_figures_and_excerpt_files():
    paths = set(build_report._report_source_paths(quick=True))
    assert {
        ROOT / "figures" / name for name in FIGURE_NAMES
    }.issubset(paths)
    assert ROOT / "code" / "abnn.py" in paths


def test_source_change_during_composition_aborts_before_conversion(
    tmp_path, monkeypatch
):
    source = tmp_path / "embedded-source.txt"
    source.write_text("before", encoding="utf-8")
    monkeypatch.setattr(build_report, "REPORT_DIR", tmp_path / "report")
    monkeypatch.setattr(
        build_report,
        "_report_source_paths",
        lambda _quick: (source,),
    )
    monkeypatch.setattr(
        build_report,
        "_load_report_artifacts",
        lambda _quick: ({}, {}, {}, {}),
    )

    def compose_and_mutate(*_args, **_kwargs):
        source.write_text("after", encoding="utf-8")
        return Document()

    monkeypatch.setattr(build_report, "_compose_document", compose_and_mutate)
    monkeypatch.setattr(
        build_report,
        "_convert_docx_to_pdf",
        lambda *_args: pytest.fail("conversion must not start"),
    )

    with pytest.raises(RuntimeError, match="changed during composition"):
        generate_report(quick=True)

    assert not (tmp_path / "report").exists()


def test_experiment_setting_prose_uses_persisted_metadata(monkeypatch):
    evaluation = json.loads(
        (ROOT / "results" / "evaluation_quick.json").read_text(encoding="utf-8")
    )
    evaluation["config"].update(
        {
            "seed": 77,
            "member_seeds": [201, 202, 203, 204],
            "map_epochs": 41,
            "abnn_epochs": 13,
            "mc_samples": 6,
            "emphasized_classes": 7,
            "emphasis_factor": 9.0,
        }
    )
    evaluation["run"]["effective"].update(
        {
            "dataset_limits": {
                "train": 123,
                "cifar10_test": 45,
                "svhn_test": 46,
                "cifar100_test": 47,
            },
            "map_epochs": 2,
            "abnn_epochs": 3,
            "member_seeds": [201, 202],
            "member_count": 2,
            "mc_samples": 4,
        }
    )
    map_history = json.loads(
        (ROOT / "results" / "map_history_quick.json").read_text(encoding="utf-8")
    )
    abnn_history = json.loads(
        (ROOT / "results" / "abnn_history_quick.json").read_text(
            encoding="utf-8"
        )
    )
    document = Document()
    monkeypatch.setattr(build_report, "_add_figure", lambda *_args, **_kwargs: None)

    build_report._section_two(document, evaluation, quick=True)
    build_report._section_four(
        document,
        evaluation,
        map_history,
        abnn_history,
        {"mode": "quick"},
        quick=True,
    )
    text = _document_text(document)

    for expected in (
        "7 seeded emphasized classes at weight 9",
        "123 training images",
        "45 CIFAR-10 test examples",
        "46 SVHN test examples",
        "47 CIFAR-100 test examples",
        "2 MAP epochs",
        "3 ABNN adaptation epochs",
        "2 members",
        "4 Monte Carlo draws",
        "41 MAP epochs",
        "13 ABNN adaptation epochs",
        "seed 77",
    ):
        assert expected in text


def test_result_interpretation_reports_abnn_underperformance(monkeypatch):
    evaluation = json.loads(
        (ROOT / "results" / "evaluation_quick.json").read_text(encoding="utf-8")
    )
    evaluation["methods"]["map"]["classification"]["accuracy"] = 0.90
    evaluation["methods"]["abnn_full"]["classification"]["accuracy"] = 0.20
    evaluation["methods"]["map"]["ood"]["svhn"]["auroc"] = 0.90
    evaluation["methods"]["abnn_full"]["ood"]["svhn"]["auroc"] = 0.30
    evaluation["methods"]["map"]["ood"]["cifar100"]["auroc"] = 0.80
    evaluation["methods"]["abnn_full"]["ood"]["cifar100"]["auroc"] = 0.50
    map_history = json.loads(
        (ROOT / "results" / "map_history_quick.json").read_text(encoding="utf-8")
    )
    abnn_history = json.loads(
        (ROOT / "results" / "abnn_history_quick.json").read_text(
            encoding="utf-8"
        )
    )
    document = Document()
    monkeypatch.setattr(build_report, "_add_figure", lambda *_args, **_kwargs: None)

    build_report._section_four(
        document,
        evaluation,
        map_history,
        abnn_history,
        {"mode": "full"},
        quick=False,
    )
    text = _document_text(document)

    assert "underperformed MAP by 70.00 percentage points" in text
    assert "no post-hoc tuning" in text


def test_word_conversion_uses_fixed_format_export_and_cleans_up(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.docx"
    source.write_bytes(b"placeholder")
    target = tmp_path / "target.pdf"

    class FakeDocument:
        def __init__(self):
            self.closed = False
            self.fields_updated = False
            self.Fields = type(
                "Fields",
                (),
                {"Update": lambda _self: setattr(self, "fields_updated", True)},
            )()

        def ExportAsFixedFormat(self, output_path, export_format):
            assert export_format == 17
            Path(output_path).write_bytes(b"%PDF-1.7\nreport")

        def SaveAs2(self, *_args, **_kwargs):
            raise AssertionError("SaveAs2 is less reliable for PDF export")

        def Close(self, _save_changes):
            self.closed = True

    class FakeWord:
        def __init__(self):
            self.document = FakeDocument()
            self.Documents = type(
                "Documents",
                (),
                {"Open": lambda _self, _path: self.document},
            )()
            self.quit_called = False

        def Quit(self):
            self.quit_called = True

    word = FakeWord()
    monkeypatch.setattr(build_report, "_dispatch_word", lambda: word)

    assert build_report._convert_docx_to_pdf(source, target) == target.resolve()
    assert word.document.fields_updated
    assert word.document.closed
    assert word.quit_called
