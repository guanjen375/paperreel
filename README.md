# paperreel

把任意 PDF 轉成繁體中文教學影片 (MP4 + SRT/ASS)。

---

## 1. 安裝

```bash
pip install -e .
```

系統需有 `ffmpeg` 和 `ffprobe` 在 PATH 上：

- Ubuntu : `sudo apt install -y ffmpeg fonts-noto-cjk`
- macOS  : `brew install ffmpeg`
- Windows: `choco install ffmpeg`

---

## 2. Mock vs 真實 provider

預設 `llm.provider = mock`、`tts.provider = mock` — 不需要任何 API key、不需要連網就能跑完並產出 MP4。但要注意：

| | 預設 (mock) | 換成真實 provider |
|---|---|---|
| 腳本/講稿 | 抓 PDF 開頭幾句套教學殼，**佔位文字** | 真正的繁中摘要、章節規劃、教學講稿 |
| 配音 | 220 Hz 嗡嗡聲，長度依字數估算 | 真實人聲 |
| 用途 | 驗證 pipeline 是否跑得通 | 可實際拿來教學的影片 |
| 成本 | 0 元、離線 | LLM 通常按 token 計費；線上 TTS 依 provider 而定 |

LLM / TTS 都是**可插拔架構**，第一版內建的真實 provider 可這樣裝：

```bash
pip install -e ".[anthropic,edge]"   # 同時裝 LLM + TTS
# 或只升級其中一邊:
pip install -e ".[anthropic]"
pip install -e ".[edge]"
```

裝好之後：

1. 在自訂 config (或直接改 `configs/default.yaml`) 把 `llm.provider` / `tts.provider` 從 `mock` 改成對應名稱 (如 `anthropic` / `edge`)。
2. 把該 provider 需要的 API key 設成環境變數 (例如 `ANTHROPIC_API_KEY`)，見下方「設定 API key」。

沒裝套件或沒設 key 時會自動 fallback 回 mock，不會中斷 pipeline。

### 設定 API key

不同 provider 取 key 的方式不一樣，常見內建幾個：

- **edge-tts** — 免費，**不需要 key**，能連網即可。
- **anthropic (Claude)** — 到 `console.anthropic.com` → **API Keys** → **Create Key**，會拿到一串 `sk-ant-...`。
- 自接其他 provider — 依該服務官網申請。

拿到 key 後設成環境變數 (把 `<your-key>` 換成實際 key)：

```bash
# macOS / Linux (bash 或 zsh)
export ANTHROPIC_API_KEY=<your-key>                                        # 本次 shell 有效
echo 'export ANTHROPIC_API_KEY=<your-key>' >> ~/.zshrc                     # 永久 (bash 改 ~/.bashrc)
```

```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "<your-key>"                                      # 本次視窗有效
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","<your-key>","User")   # 永久 (重開終端機後生效)
```

驗證：

```bash
echo $ANTHROPIC_API_KEY        # macOS / Linux
$env:ANTHROPIC_API_KEY         # PowerShell
```

> **⚠️ 安全提醒**：API key 等同密碼。不要 commit 進 git、不要貼到聊天 / issue / 截圖。若不慎外洩，立刻到該 provider 後台 revoke 並重新申請。

想接其他 provider：實作 `src/paperreel/providers/llm_base.py` 或 `tts_base.py` 的介面，再到對應 `make_*_provider()` 加分支即可。

---

## 3. 產生影片（一條指令）

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
- 第一次跑想試小成本：加 `--dry-run`（強制全 mock，不需 API key / 網路）。

---

## 4. 輸出位置

跑完後到 `./runs/my_video/outputs/` 找：

| 檔案 | 內容 |
|---|---|
| `final.mp4` | 1080p / 30fps / H.264+AAC 教學影片 |
| `subtitles.srt` | 整片字幕 |
| `quality_report.json` | 長度、缺漏、原文重疊比例的檢查報告 |
| `segments/<scene_id>.mp4` | 各 scene 的 MP4，可單獨重跑 |

`./runs/my_video/assets/` 內有逐段的 wav / png / srt / ass，方便事後挑單段調整。

---

## 5. 出錯怎麼辦

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
