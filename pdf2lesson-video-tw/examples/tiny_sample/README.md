# tiny_sample

`tests/conftest.py` 會在 `tmp_path` 動態產生一份 4 頁的 CJK PDF；
這個資料夾故意保持空白，避免把任何受版權保護的範例檔簽入 repo。

如果想手動跑一次 dry-run：

```bash
python - <<'PY'
import fitz, pathlib
p = pathlib.Path("examples/tiny_sample/tiny.pdf")
doc = fitz.open()
for i, t in enumerate([
    "第一章 緒論\n本書介紹簡短範例" * 6,
    "第二章 方法\n說明流程與重點" * 6,
    "第三章 範例\n以範例展示步驟" * 6,
    "結語\n回顧所有重點" * 6,
]):
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), t, fontsize=12, fontname="china-s")
doc.save(p)
print("wrote", p)
PY

pdf2lesson all examples/tiny_sample/tiny.pdf \
    --project ./runs/tiny --dry-run --skip-render
```

完成後檢視 `runs/tiny/intermediate/scene_graph.json` 與
`runs/tiny/assets/visuals/*.png`。
