# paperreel

把任意 PDF 轉成繁體中文教學影片 (MP4 + SRT/ASS)。

---

## 1. 安裝

```bash
pip install -e .
# 想用真實 LLM / 線上 TTS:
pip install -e ".[anthropic,edge]"
export ANTHROPIC_API_KEY=sk-ant-...     # 真實 LLM 才需要
```

系統需有 `ffmpeg` 和 `ffprobe` 在 PATH 上：

- Ubuntu : `sudo apt install -y ffmpeg fonts-noto-cjk`
- macOS  : `brew install ffmpeg`
- Windows: `choco install ffmpeg`

---

## 2. 產生影片（一條指令）

```bash
paperreel all ./your_book.pdf \
    --project ./runs/my_video \
    --target-minutes auto \
    --max-hours 10 \
    --resume
```

- `--target-minutes auto`：依 PDF 篇幅自動估算 (12–120 分鐘)；可填整數強制指定。
- `--max-hours 10`：總執行時間上限。
- `--resume`：中斷後再執行同一行即從中斷點接續。
- 第一次跑想試小成本：加 `--dry-run`（全 mock，不需 API key / 網路）。

---

## 3. 輸出位置

跑完後到 `./runs/my_video/outputs/` 找：

| 檔案 | 內容 |
|---|---|
| `final.mp4` | 1080p / 30fps / H.264+AAC 教學影片 |
| `subtitles.srt` | 整片字幕 |
| `quality_report.json` | 長度、缺漏、原文重疊比例的檢查報告 |
| `segments/<scene_id>.mp4` | 各 scene 的 MP4，可單獨重跑 |

`./runs/my_video/assets/` 內有逐段的 wav / png / srt / ass，方便事後挑單段調整。

---

## 4. 出錯怎麼辦

```bash
paperreel status        --project ./runs/my_video      # 看每個 stage 狀態
paperreel retry-failed  --project ./runs/my_video      # 重做標 failed 的 scene
paperreel all ./your_book.pdf --project ./runs/my_video --resume   # 繼續跑
```

只想重做某幾個 stage：

```bash
paperreel all ./your_book.pdf --project ./runs/my_video \
    --force-stage plan,script,scenes --resume
```

可用的 stage 名稱：`ingest, plan, script, scenes, audio, visuals, subtitles, segments, concat, quality`。
