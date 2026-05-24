"""Document classifier — heuristic must spot contract / paper / manual
shapes from the page text alone. We don't need a real PDF here; the
classifier eats a :class:`ChunkedSources` directly."""
from __future__ import annotations

from paperreel.models import (ChunkedSources, DocKind, PdfChunk, PdfPage)
from paperreel.utils import doc_classify


def _sources(pages: list[str], *, images: int = 0,
             heading_count: int = 0) -> ChunkedSources:
    pdf_pages = [
        PdfPage(page=i + 1, text=t, cjk_char_count=len(t),
                headings=[], text_source="text")
        for i, t in enumerate(pages)
    ]
    joined = "\n".join(pages)
    chunk = PdfChunk(
        chunk_id="chunk_0001",
        start_page=1,
        end_page=len(pages),
        text=joined,
        cjk_char_count=len(joined),
        headings=[],
    )
    return ChunkedSources(
        source_pdf="memory.pdf",
        pdf_sha256="0" * 64,
        page_count=len(pages),
        cjk_char_count=sum(p.cjk_char_count for p in pdf_pages),
        image_count=images,
        heading_count=heading_count,
        estimated_density=(sum(p.cjk_char_count for p in pdf_pages)
                           / max(1, len(pdf_pages))),
        pages=pdf_pages,
        chunks=[chunk],
        images=[],
    )


def test_classifier_detects_contract() -> None:
    sources = _sources([
        "本合約由甲方與乙方雙方簽署，雙方應於 45 天內完成付款，"
        "違約金為合約總額的 30% 並須附上身分證、護照與簽名。",
        "若任一方違約，甲方有權終止本協議並請求賠償。"
        "雙方亦同意 7 天內提供必要文件。",
    ])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.contract
    assert "deadline_timeline" in profile.suggested_storyboard
    assert "penalty_table" in profile.suggested_storyboard


def test_classifier_detects_paper() -> None:
    sources = _sources([
        "Abstract: We introduce a novel method for compressing transformer "
        "models. Our experiments on standard benchmarks show significant "
        "improvement.",
        "Methods: We use mixed-precision quantization. Results: We "
        "evaluate on three datasets. Discussion: We compare with related work. "
        "References: [1] ...",
    ])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.paper


def test_classifier_detects_manual() -> None:
    sources = _sources([
        "操作說明：請依下列步驟安裝。第一步：解壓縮。第二步：執行 setup.exe。"
        "注意事項：請勿同時開啟其他應用程式。",
        "故障排除：若無法啟動，請檢查防毒軟體設定。警告：請勿在執行中拔除電源。",
    ])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.manual


def test_classifier_unknown_when_nothing_matches() -> None:
    sources = _sources(["天空很藍，今天天氣很好。", "我去公園散步看花。"])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.unknown
    assert profile.suggested_storyboard  # always a fallback skeleton
