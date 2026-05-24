# paperreel (API edition)

把任意 PDF 轉成繁體中文教學影片 (MP4 + SRT/ASS)。

> 這是 **`paperreel_api`** 分支：腳本 / 章節走 Anthropic API、配音走 edge-tts。
> 沒有 mock fallback — 缺套件 / 缺 key / API 出錯都會明確報錯。
> 若想跑離線本地模型版本，請切到 `main` 分支。

---

## 1. 安裝

```bash
pip install -e ".[anthropic,edge]"
```

API edition **必裝** `anthropic` + `edge-tts` 兩個 extras，否則 pipeline 啟動時會直接報錯。

系統需有 `ffmpeg` 和 `ffprobe` 在 PATH 上：

- Ubuntu : `sudo apt install -y ffmpeg fonts-noto-cjk`
- macOS  : `brew install ffmpeg`
- Windows: `choco install ffmpeg`

---

## 2. 設定 API key

到 `console.anthropic.com` → **API Keys** → **Create Key**，拿到一串 `sk-ant-...`。

> **⚠️ 跟 Claude Code 訂閱的衝突**：如果你也在用 Claude Code (Pro / Max 訂閱)，**不要** 把 key 設成 `ANTHROPIC_API_KEY`。Claude Code 一旦偵測到這個變數，會自動切成「按 token 計費」走 API，繞過你的訂閱。paperreel 因此優先讀專案專屬的 `PAPERREEL_ANTHROPIC_API_KEY`，找不到才 fallback 到 `ANTHROPIC_API_KEY`。下面範例都用前者。

把 `<your-key>` 換成實際 key：

```bash
# macOS / Linux (bash 或 zsh)
export PAPERREEL_ANTHROPIC_API_KEY=<your-key>                                        # 本次 shell 有效
echo 'export PAPERREEL_ANTHROPIC_API_KEY=<your-key>' >> ~/.zshrc                     # 永久 (bash 改 ~/.bashrc)
```

```powershell
# Windows PowerShell
$env:PAPERREEL_ANTHROPIC_API_KEY = "<your-key>"                                      # 本次視窗有效
[Environment]::SetEnvironmentVariable("PAPERREEL_ANTHROPIC_API_KEY","<your-key>","User")   # 永久 (重開終端機後生效)
```

驗證：

```bash
echo $PAPERREEL_ANTHROPIC_API_KEY        # macOS / Linux
$env:PAPERREEL_ANTHROPIC_API_KEY         # PowerShell
```

> **⚠️ 安全提醒**：API key 等同密碼。不要 commit 進 git、不要貼到聊天 / issue / 截圖。若不慎外洩，立刻到該 provider 後台 revoke 並重新申請。

edge-tts 免費、不需要 key，但要能連網 (走微軟雲端)。

### 換模型 / 估算成本

- 要換 LLM 模型，改 `configs/default.yaml` 裡的 `llm.model` 即可。
- Anthropic 的可用型號、定價、推薦用途會持續更新——**請以 [Anthropic 官方文件](https://docs.anthropic.com/en/docs/about-claude/models) 為準**；本 README 刻意不寫死特定型號名稱以免過時。一般而言 Claude 家族「能力越高、單價越貴」是正比關係，預設選的是中階兼顧成本與品質的型號。
- 每跑一份 PDF 約打 `chunks + chapters + 1` 次 LLM call。1 頁的小 PDF 約 6 通；100 頁的書可能 30 通以上，實際費用看選的模型。

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

常見錯誤訊息：

- `AnthropicProviderError: no API key` → 沒設 `PAPERREEL_ANTHROPIC_API_KEY`，回到 §2。
- `AnthropicProviderError: anthropic package not installed` → 沒裝 extras，回到 §1。
- `EdgeTTSError: edge-tts package not installed` → 同上。
- `EdgeTTSError: ffmpeg not on PATH` → 裝 ffmpeg 或檢查 PATH。
