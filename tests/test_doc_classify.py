"""Document classifier — heuristic must spot contract / paper / manual
shapes from the page text alone. We don't need a real PDF here; the
classifier eats a :class:`ChunkedSources` directly."""
from __future__ import annotations

from paperreel.models import (ChunkedSources, DocKind, PdfChunk, PdfImage,
                               PdfPage, VisualCandidate)
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
    pdf_images = [
        PdfImage(
            image_id=f"img_{i}", page=(i % max(1, len(pages))) + 1,
            path=f"/tmp/img_{i}.png", width=640, height=360,
            pixel_count=640 * 360, sha256=f"{i:064x}",
            bbox=(80.0, 120.0, 520.0, 420.0),
            caption_hint="攝影範例圖",
        )
        for i in range(images)
    ]
    visual_inventory = [
        VisualCandidate(
            candidate_id=f"vis_{i}", page=im.page, image_path=im.path,
            bbox=im.bbox, nearby_heading="攝影技巧範例",
            nearby_caption=im.caption_hint, nearby_text=pages[im.page - 1][:120],
            image_width=im.width, image_height=im.height,
            image_size=(im.width, im.height), page_area_ratio=0.28,
            visual_role="source_photo", salience_score=4.2,
            likely_useful=True, is_decorative=False, source_image_id=im.image_id,
            source_quote=pages[im.page - 1][:80],
        )
        for i, im in enumerate(pdf_images)
    ]
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
        images=pdf_images,
        visual_inventory=visual_inventory,
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


def test_classifier_detects_form() -> None:
    sources = _sources([
        "申請表：請填寫姓名、身分證字號、護照號碼、聯絡電話與電子郵件。"
        "請勾選同意個人資料蒐集，並於簽名欄簽名或蓋章。",
    ])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.form


def test_classifier_detects_policy() -> None:
    sources = _sources([
        "本辦法適用範圍為全體員工。員工應遵守資訊安全規範，"
        "違反本政策者依公司規定懲處。",
        "本規範包含個人資料保護、防疫措施及合規要求。",
    ])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.policy


def test_classifier_unknown_when_nothing_matches() -> None:
    sources = _sources(["天空很藍，今天天氣很好。", "我去公園散步看花。"])
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.unknown
    assert profile.suggested_storyboard  # always a fallback skeleton

def test_classifier_detects_visual_rich_tutorial_manual() -> None:
    sources = _sources([
        "第二單元：學習相機的各種拍攝模式。這裡用範例照片說明 Auto、P、A、S、M 模式如何選擇。",
        "曝光鐵三角包含光圈、快門與 ISO 感光度。請看圖中的景深變化與快門速度差異。",
        "直方圖與曝光補償可以幫助你判斷畫面是否過曝。以下步驟示範如何調整設定。",
        "白平衡會改變色溫。比較偏冷與偏暖的範例，可以更快理解顏色控制。",
    ], images=5, heading_count=8)
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.manual
    assert profile.document_visual_rich is True
    assert profile.visual_tutorial is True
    assert "source_visual_explainer" in profile.suggested_storyboard


def test_contract_with_decorative_images_stays_contract() -> None:
    sources = _sources([
        "本合約由甲方與乙方雙方簽署，雙方應於 45 天內完成付款，違約金為 30%。",
        "簽名蓋章欄位僅供確認。若逾期取消，訂金不予退款。",
    ], images=0, heading_count=1)
    decorative = VisualCandidate(
        candidate_id="seal_1", page=2, image_path="/tmp/seal.png",
        bbox=(420.0, 680.0, 500.0, 760.0), nearby_heading="簽名蓋章",
        nearby_caption="印章", nearby_text="簽名蓋章欄位",
        image_width=120, image_height=120, image_size=(120, 120),
        page_area_ratio=0.02, visual_role="seal", salience_score=0.1,
        is_decorative=True, likely_useful=False, repeated=False,
    )
    sources = sources.model_copy(update={
        "visual_inventory": [decorative],
        "image_count": 1,
    })
    profile = doc_classify.classify(sources)
    assert profile.doc_kind == DocKind.contract
    assert profile.document_visual_rich is False
    assert "deadline_timeline" in profile.suggested_storyboard

