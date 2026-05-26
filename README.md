# paperreel

把 PDF 轉成繁體中文、source-grounded 的視覺導讀影片。預設輸出是白色/淡色資訊卡片：時間線、表格、清單、風險提醒、do/don't、重點回顧與來源頁註記。重要文字由程式穩定渲染，事實來自 PDF，不靠生成圖片承載內容。

## 一條指令

```bash
paperreel input.pdf --project runs/demo --target-minutes 5
```

`--target-minutes` 是一般使用者唯一需要調整的內容長度控制值。可以用 `2`、`5`、`10` 這類分鐘數；實際長度會盡量落在目標的正負 10% 內。若省略或使用 `auto`，paperreel 會依 PDF 頁數、文字量與文件類型估算合理長度。

預設不需要雲端 API、不需要 OpenAI/Anthropic/Gemini、不需要 VLM、不需要 SDXL，也不需要特殊 GPU。視覺卡片用本機 deterministic renderer 產生；GPU 或高階硬體只會改善本地 LLM/TTS/OCR 的速度、可用模型大小與批次處理能力。

## 安裝

需要 Python 3.12+、`ffmpeg`、`ffprobe`，建議安裝 Noto CJK 字型：

```bash
sudo apt install -y ffmpeg fonts-noto-cjk
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

`[all]` 會安裝本機 Ollama client、XTTS、OCR 與選用的 SDXL 依賴。預設 explainer 不會呼叫 SDXL；若只想跑 CPU/無生成圖片路徑，可先用：

```bash
pip install -e ".[ollama,xtts,ocr,test]"
```

第一次使用本機模型時可能需要下載權重。這是本機模型初始化，不是雲端推理 API。

## 預設輸出

預設模式會自動判斷文件類型並選 storyboard；使用者不需要選 mode、style、depth、renderer、VLM、SDXL 或硬體 profile：

| 文件類型 | 常見卡片 |
|---|---|
| 合約 / 表單 / 政策 | 總覽、期限時間線、費用/罰則表、應辦文件、風險/不退費提醒、do/don't、回顧清單 |
| 論文 | 問題、核心想法、方法、結果、限制、takeaways |
| 手冊 | 前置條件、步驟、警告、故障排除、檢查清單 |
| 報告 | 摘要、關鍵指標、趨勢、風險、建議、回顧 |
| 投影片 | 段落總覽、重點摘錄、回顧 |

對合約、表單、政策與商務文件，paperreel 會保留時間線、表格、風險卡、do/don't 與檢查清單這類 source-grounded 卡片。對影像豐富的教學書、操作手冊、tutorial 或 slide-like PDF，paperreel 會自動改成來源視覺 walkthrough：優先顯示 PDF 自己的照片、圖解、表格、截圖或有意義的頁面裁切，旁白解釋畫面，卡片文字只保留短 headline 與 callout。

所有 factual scene 會盡量帶 `source_pages`、`evidence_spans` 與 `facts`；visual-first scene 會額外帶 `visual_anchor` 與 `screen_plan`。若重要數字、日期、費用、百分比、期限、風險或義務無法被來源支持，預設會修復、移除該場景，或清楚失敗。

## 複查輸出

跑完後可以產生靜態 review，不需要 VLM：

```bash
paperreel review --project runs/demo
```

會輸出：

| 檔案 | 用途 |
|---|---|
| `outputs/review/contact_sheet.jpg` | 全部卡片縮圖牆 |
| `outputs/review/storyboard.html` | 每張卡片、旁白、facts、來源摘錄 |
| `outputs/review/semantic_quality.json` | 時長、evidence、生成圖片外洩、卡片密度、來源覆蓋、visual-first 覆蓋率、screen/narration overlap、文件類型高優先事實檢查 |

主要影片與字幕在：

| 檔案 | 內容 |
|---|---|
| `outputs/final.mp4` | 最終影片 |
| `outputs/subtitles.srt` / `.ass` | 字幕 |
| `outputs/quality_report.json` | 影片時長與資產缺漏檢查 |

## 本機模型

預設設定使用本機 Ollama 與 XTTS：

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
export COQUI_TOS_AGREED=1
```

Ollama 主要用於 outline 摘要；explainer 腳本與視覺卡片仍以 PDF 抽取 facts/evidence 為核心。若 Ollama 暫時不可用，explainer plan 會退回 deterministic outline。XTTS 用於語音合成；`tts.device: auto` 會自動使用 CUDA 或 CPU。預設語速略低於 XTTS 原速，避免繁中導讀聽起來太趕。

高階硬體可以改用更大的本機模型、較大的 context、較快的 TTS/OCR，或自行開啟進階 review；這些都不是正常路徑必需。`highend_sketchbook` 只是一個進階範例，可使用如 `qwen3:30b` 或 `qwen2.5:32b-instruct` 搭配較大 context。

## 改善中文旁白聲音

XTTS 內建 speaker 在中文旁白可能有外國口音。建議提供一段你本人或你有權使用的本機繁中參考聲音，paperreel 會自動檢查、轉 mono、重取樣、裁切前後靜音、正規化音量，並快取成 XTTS 的 `speaker_wav`：

```bash
paperreel input.pdf --project runs/demo --target-minutes 5 --voice-sample ./my_voice.wav
```

建議聲音樣本：

- 6-10 秒；4-15 秒可用但可能警告，少於 4 秒或超過 20 秒會失敗
- 單人說話、安靜環境、沒有背景音樂
- 沒有明顯混響或回音
- 語速自然，不要太播報腔
- WAV 最佳；16k、24k、48k 會自動轉成 mono 24k WAV
- 請只使用你擁有或取得授權的聲音；不要擅自使用第三方產品、公眾人物或他人的聲音

沒提供 `--voice-sample` 時，paperreel 會顯示 `[INFO] 未提供 voice_sample，使用 XTTS 預設聲音`，並使用 XTTS 內建 speaker 繼續產生影片。一般使用者不用改 config 或原始碼。進階 config 仍可使用：

`--voice-sample` 只控制聲音來源，不是影片內容控制參數；`--target-minutes` 仍是一般使用者唯一需要調整的內容長度控制值。
## 進階相容選項

舊版旗標仍保留給進階使用者，例如 `--style default`、`--config highend_sketchbook`、`--depth brief/deep`、`--force-stage`、`--skip-render`。正常使用不需要選 style、depth、config、renderer、VLM、SDXL 或硬體 profile。

若要看所有子指令：

```bash
paperreel --help
paperreel run --help
```

## 續跑與除錯

```bash
paperreel status --project runs/demo
paperreel retry-failed --project runs/demo
paperreel input.pdf --project runs/demo --target-minutes 5
```

同一個 `--project` 會自動續跑。可用 stage 名稱：`ingest, plan, script, scenes, match_visuals, audio, visuals, subtitles, segments, concat, quality`。

## 開發 / 測試

```bash
pip install -e ".[test]"
pytest -q
```

測試不需要 GPU、網路、雲端 API、SDXL 或 VLM；provider 會被測試替身取代。

可選的本機 regression / reference 檔案放在 `dev_samples/reference/`；舊本機開發資料夾 `dev_examples/reference/` 只作為 legacy fallback：

| 路徑 | 用途 |
|---|---|
| `dev_samples/reference/sample.pdf` | 合約/表單類 PDF smoke regression |
| `dev_samples/reference/sample_visual_tutorial.pdf` | 影像豐富教學 PDF smoke regression（若本機存在） |
| `dev_samples/reference/notebooklm_short.mp4` | 視覺節奏與資訊密度參考，不複製品牌或 UI |
| `dev_samples/reference/notebooklm_long.mp4` | 較長篇 pacing 參考，不是正常使用依賴 |
| `dev_samples/reference/frames/` | MP4 無法讀取時的備用影格參考 |

這些檔案不是一般使用者執行 `paperreel input.pdf --project runs/demo --target-minutes 5` 的必要條件。測試會優先使用 `dev_samples/reference/`；若本機仍有舊的 `dev_examples/reference/`，只作為 legacy fallback。參考影片只用來觀察「來源視覺 + 旁白解釋」的高層節奏與視覺層級，不複製品牌、UI、水印、聲音或動畫。
