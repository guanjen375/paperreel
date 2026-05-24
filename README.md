# paperreel (local-only build)

把任意 PDF 轉成繁體中文教學影片 (MP4 + SRT/ASS)，**全部在本機跑**。
沒有 API key、不打網路 (除了第一次拉模型)、不會被算 token 費。

> 如果你想要走 API 服務 (Anthropic / OpenAI / 線上 TTS) 的版本，請看另一個分支。

---

## 1. 安裝

需要 Python ≥ 3.12、`ffmpeg`、`ffprobe`：

```bash
sudo apt install -y ffmpeg fonts-noto-cjk   # Ubuntu
brew install ffmpeg                          # macOS
choco install ffmpeg                         # Windows
```

Ubuntu 24.04+ / Debian 12+ 預設 Python 走 PEP 668，請先建 venv (或 conda env)，否則 `pip install` 會被擋掉：

```bash
python3 -m venv .venv && source .venv/bin/activate
# 之後本 README 所有 pip / paperreel 指令都在這個 venv 裡跑
```

> 不想用 venv，也可以在每個 `pip install` 後面加 `--break-system-packages`，但會污染系統 Python。

clone + install：

```bash
git clone <this-repo> paperreel && cd paperreel
pip install -e ".[all]"      # 同時裝 LLM + TTS + SDXL
# 或單獨裝：
pip install -e ".[ollama]"   # 只裝 LLM
pip install -e ".[xtts]"     # 只裝 TTS
pip install -e ".[sdxl]"     # 只裝圖片生成
```

`[xtts]` 跟 `[sdxl]` 會拉 PyTorch (~2 GB)；建議事先用對應 CUDA 版本的 wheel 裝好 torch，再 `pip install -e .[…]`，避免抓到 CPU-only 版本。

> `[xtts]` extra 已經改用社群維護的 `coqui-tts` fork（同樣的 XTTS v2 模型、同樣的 `from TTS.api import TTS`），因為 PyPI 上原始的 Coqui `TTS` 套件最高只支援 Python 3.11。同時也鎖了 `transformers<5`，因為 coqui-tts 0.27 還在用 transformers 4.x 的 `isin_mps_friendly`。

---

## 2. 本地模型 (LLM / TTS / SDXL)

| 角色 | Backend | 預設模型 | 第一次下載 | 備註 |
|---|---|---|---|---|
| LLM | [Ollama](https://ollama.com) | `qwen2.5:14b-instruct` | ~8 GB | 繁中 + JSON 結構化輸出穩定 |
| TTS | [Coqui XTTS v2](https://github.com/idiap/coqui-ai-TTS) | `tts_models/multilingual/multi-dataset/xtts_v2` | ~1.8 GB | 多語、可 voice clone (用 Idiap 維護的 `coqui-tts` fork) |
| 圖片 | [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (diffusers) | `stabilityai/stable-diffusion-xl-base-1.0` | ~6.7 GB | 只有 `visual_type=generated_image` 的 scene 才會用到 |

模型完全跑不到時 (Ollama 沒開、TTS 套件沒裝、SDXL OOM) 會直接 raise；沒有 silent fallback，避免你拿到不知道是真是假的影片。

### 2.1 LLM — 裝 Ollama + 拉模型

```bash
# Ubuntu / macOS / Windows 都從官網裝 daemon:
curl -fsSL https://ollama.com/install.sh | sh    # Ubuntu / macOS
# Windows 走官網 installer

# 啟動 daemon (大多數平台是 systemd / launchd 自動跑)
ollama serve &

# 拉預設模型 (~8 GB)
ollama pull qwen2.5:14b-instruct
```

確認可用：

```bash
ollama run qwen2.5:14b-instruct "用繁體中文簡單說明牛頓第二定律。"
```

**有 GPU 的話一定要確認 ollama 真的吃到 GPU**：14B 模型跑在 CPU 上會慢到 paperreel 的 HTTP timeout（`ReadTimeout`）跳出，整條 pipeline 死在 script stage。

```bash
# 上面 ollama run 之後模型還在記憶體裡，直接查：
curl -s http://localhost:11434/api/ps | python3 -c \
  "import sys,json;m=json.load(sys.stdin)['models'][0];print(f'size_vram={m[\"size_vram\"]/1e9:.1f} GB')"
# 預期：size_vram=10+ GB (Q4_K_M 14B 大概 12-14 GB)
# 若回 0.0 GB → ollama 在 CPU 跑，重啟 daemon 讓它重新 detect GPU：
sudo systemctl restart ollama          # Linux systemd 安裝
# macOS: 從選單列 quit Ollama.app 再開
```

ollama 透過 curl 安裝腳本起的 systemd 服務常常在開機時 NVIDIA driver 還沒載入就先起來、之後永遠看不到 GPU；restart 一次就好。

要換更大模型 (e.g. RTX 5090)：改 `configs/default.yaml` 的 `llm.model` (或用 `configs/rtx5090.yaml` 預設的 `llama3.3:70b-instruct`)，記得先 `ollama pull`。

### 2.2 TTS — XTTS 語音設定

XTTS v2 第一次合成會自動下載 weights 到 `~/.local/share/tts`，並會跳出 Coqui Public Model License (CPML) 同意 prompt。要在非互動環境（CI / 背景跑）順利跑過，先 export：

```bash
export COQUI_TOS_AGREED=1
```

語音來源二擇一：

```yaml
# configs/default.yaml
tts:
  speaker_wav: /abs/path/to/reference.wav   # 6–10 秒乾淨人聲，效果最好
  speaker: "Ana Florence"                   # 沒給 speaker_wav 才會用內建 speaker
  language: "zh-cn"                          # XTTS 用 zh-cn tag，餵繁中文字仍可
```

要 GPU：`device: "cuda"`；沒 GPU 改 `cpu` 也跑得動，只是慢 (一個 30 秒旁白要 30+ 秒)。

### 2.3 SDXL — 圖片生成 (選用)

只在 LLM 決定某個 scene 用 `visual_type: generated_image` 時才會呼叫 SDXL。沒裝 / 失敗 / 沒 GPU 都會自動 fallback 到 Pillow 卡片渲染，pipeline 不會中斷。

```yaml
image:
  provider: "sdxl"
  model: "stabilityai/stable-diffusion-xl-base-1.0"
  device: "cuda"            # CPU 太慢, 不建議
  num_inference_steps: 30   # 5090 上一張約 5–10 秒
```

要關掉，就把 `image.provider` 改成 `sdxl` 但保證 `[sdxl]` 沒裝 → 所有 scene 都會自動走卡片。或更乾脆：把 LLM prompt 限制讓它不要選 `generated_image` (在 `src/paperreel/providers/llm_ollama.py` 把那個 enum value 拿掉)。

---

## 3. 產生影片 (一條指令)

```bash
paperreel all ./your_book.pdf \
    --project ./runs/my_video \
    --target-minutes auto \
    --max-hours 10 \
    --resume
```

- `--target-minutes auto`：依 PDF 篇幅自動估算 (3–120 分鐘)；可填整數強制指定。短 PDF 別預期硬撐到 12 min — 真實長度通常是 PDF 字數 / 2200 chars/min，下限 3 min。
- `--max-hours 10`：總執行時間上限 (中斷後再 `--resume`)。
- `--resume`：中斷後再執行同一行即從中斷點接續。

第一次跑會看到 ollama / TTS / (可能) SDXL 的下載 + 載入時間；之後跑都是冷快取 → 立即開始。

---

## 4. 輸出位置

跑完後到 `./runs/my_video/outputs/` 找：

| 檔案 | 內容 |
|---|---|
| `final.mp4` | 1080p / 30fps / H.264+AAC 教學影片 |
| `subtitles.srt` | 整片字幕 |
| `quality_report.json` | 長度、缺漏、原文重疊比例的檢查報告 |
| `segments/<scene_id>.mp4` | 各 scene 的 MP4，可單獨重跑 |

`./runs/my_video/assets/` 內有逐段的 wav / png / srt / ass，方便事後挑單段調整；`assets/generated/` 是 SDXL 原圖 (未套卡片框)。

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

可用的 stage 名稱：`ingest, plan, script, scenes, match_visuals, audio, visuals, subtitles, segments, concat, quality`。

> `match_visuals`：把 ingest 抓到的 PDF 圖片配對到對應的 scene，符合條件的 scene 會升級成 `pdf_image`，影片裡會看到原文件的圖表 + 你寫的字幕。要關掉就在 config 把 `visuals.prefer_pdf_figures` 設成 `false`。

要處理掃描檔 / 投影片截圖 PDF：

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-chi-sim
pip install -e ".[ocr]"
```

OCR 預設 `ocr_fallback: true` — 該頁文字少於 `ingest.ocr_min_chars` 時自動跑 Tesseract。沒裝 `[ocr]` 就會 silently degrade 成 empty page；`quality_report.json` 會把這些頁面標出來。

常見錯誤：

| 訊息 | 原因 / 解法 |
|---|---|
| `cannot reach Ollama at http://localhost:11434` | `ollama serve` 沒跑，或 daemon 在別的 host／port — 改 `llm.base_url` |
| `model not found` / 拉不到 | `ollama pull <model>` 沒做；或 `llm.model` 拼錯 |
| `OllamaUnavailable: ... ReadTimeout` | 通常是 ollama 跑在 CPU 上、大模型推理超過 paperreel 的 HTTP timeout — 照 §2.1 檢查 `size_vram > 0`，0 的話 `sudo systemctl restart ollama` |
| `Coqui TTS not installed` | `pip install -e ".[xtts]"` (會抓 `coqui-tts` fork — 原 `TTS` 套件最高到 Python 3.11)；torch 要是 CUDA 版才能用 GPU |
| TTS 第一次卡在 `... agree to the terms of the non-commercial CPML ... [y/n]` | `export COQUI_TOS_AGREED=1` 再重跑 (見 §2.2) |
| `pypinyin` 沒裝 / `ImportError('Chinese requires: pypinyin')` | `[xtts]` 已含 `pypinyin`；舊環境用 `pip install -e ".[xtts]" --upgrade` 重灌即可 |
| `no CUDA GPU detected — SDXL on CPU is impractical` | 換 GPU 機器，或在 config 把 LLM prompt 不要產 `generated_image` (見 §2.3) |

---

## 6. 開發 / 測試

```bash
pip install -e ".[test]"
pytest -q
```

測試完全跑得動沒有 GPU / 沒有 Ollama / 沒有 SDXL — `tests/conftest.py` 會把三個 provider factory 換成 `tests/_fakes/` 裡的測試替身。production code 不會被 mock 出來。
