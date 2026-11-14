# ed-dit-server: Native MiniMax H3 Decoupled DiT Microservice (C++)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Arch](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-orange.svg)]()
[![Release](https://img.shields.io/badge/Release-v1.0.0-success.svg)](https://github.com/obhasashare2024-namo/ed-dit-server/releases/tag/v1.0.0)

A high-performance, resident, standalone HTTP microservice and CLI runner written in pure C++ for **MiniMax H3 HyperFlow Flash 14B (FL2VA)** audio-visual video generation. Designed specifically for consumer-grade 8GB VRAM GPUs (e.g., RTX 3070 / RTX 4060) utilizing fine-grained dynamic graph-cut tensor offloading and native PyTorch `.pt` output generation without requiring Python or Rust runtimes during inference.

---

## Language Navigation / 語言導航 / 言語ナビゲーション

- [English](#-english)
  - [Architecture & Design Philosophy](#architecture--design-philosophy)
  - [Hardware & Software Requirements](#hardware--software-requirements)
  - [Compilation & Installation](#compilation--installation)
  - [REST API Reference](#rest-api-reference)
  - [End-to-End Pipeline Walkthrough](#end-to-end-pipeline-walkthrough)
  - [Daemon & Service Management](#daemon--service-management)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Physical Memory Limits (3s vs 1080p 8s)](#physical-memory-limits-3s-vs-1080p-8s)
- [繁體中文](#-繁體中文)
  - [架構理念與核心優勢](#架構理念與核心優勢)
  - [軟硬體環境要求](#軟硬體環境要求)
  - [編譯與安裝指引](#編譯與安裝指引)
  - [REST API 接口規範](#rest-api-接口規範)
  - [端到端完整生成工作流](#端到端完整生成工作流)
  - [後台常駐與系統服務配置](#後台常駐與系統服務配置)
  - [物理實測性能數據](#物理實測性能數據)
  - [顯存物理邊界深入解析 (3s vs 1080p 8s)](#顯存物理邊界深入解析-3s-vs-1080p-8s)
- [日本語](#-日本語)
  - [アーキテクチャと設計思想](#アーキテクチャと設計思想)
  - [動作要件](#動作要件)
  - [ビルドとインストール手順](#ビルドとインストール手順)
  - [REST API 仕様](#rest-api-仕様)
  - [エンドツーエンドの実行手順](#エンドツーエンドの実行手順)
  - [デーモン化と自動起動](#デーモン化と自動起動)
  - [ベンチマーク測定結果](#ベンチマーク測定結果)
  - [VRAM 物理制限の分析 (3s と 1080p 8s)](#vram-物理制限の分析-3s-と-1080p-8s)

---

<a name="-english"></a>
## 🌐 English

### Architecture & Design Philosophy

Video generation using 14B parameter models traditionally requires datacenter GPUs with 24GB+ VRAM or heavy multi-GPU setups. `ed-dit-server` breaks this barrier for consumer 8GB GPUs through a three-stage decoupled pipeline:

```
[ Stage 1: Condition Encoding ]
  Host RAM / CPU / Remote Node
  Qwen3-VL text/image/audio tokenization -> stage1_input.edbin
               │
               ▼ HTTP POST (latent path)
[ Stage 2: In-Process DiT Velocity Integration (This Work) ]
  RTX 3070 8GB (Peak Dynamic VRAM < 2.2 GB)
  ed-dit-server (Pure C++ Resident Microservice)
  ├── Preloaded GGUF Weights in Host RAM (1.8s cold load, zero reload penalty)
  ├── Fine-Grained Graph-Cut Engine (≤ 200MB execution segments)
  ├── In-Process Flash Attention & AdaLN Modulation Fastpaths
  └── Native In-Memory PyTorch ZIP/Pickle Builder -> stage2_denoised.pt
               │
               ▼ Native PyTorch .pt
[ Stage 3: VAE Latent Decoding ]
  Host RAM / CPU / Remote Node
  MiniMax VAE decode -> final video MP4
```

#### Why Pure C++?
1. **Zero Python Overhead**: Eliminates GIL bottlenecks, Python garbage collection latency, and IPC socket serialization overhead.
2. **Zero Rust FFI Copies**: Removes cross-language ABI boundaries and double buffering between Rust and C++.
3. **Direct CUDA Memory Management**: Uses `ggml-cuda` unified memory management with fine-grained partial parameter swapping.

---

### Hardware & Software Requirements

- **GPU**: NVIDIA GPU with Compute Capability ≥ 8.6 (Ampere RTX 3070, RTX 3060, RTX 4060, etc.) with at least 8 GB VRAM.
- **System Memory**: 16 GB+ Host RAM (32 GB recommended to hold GGUF weights in RAM).
- **OS**: Linux x86_64 (Arch, Artix, Ubuntu 20.04+, Debian 11+, RHEL 8+).
- **Compiler**: GCC ≥ 10.0 / Clang ≥ 12.0 with C++17 support.
- **CUDA Toolkit**: CUDA ≥ 12.0 (tested on CUDA 12.4 / 13.4 with `nvcc`).
- **CMake**: CMake ≥ 3.18.

---

### Compilation & Installation

#### 1. Clone the Standalone Repository
```bash
git clone https://github.com/obhasashare2024-namo/ed-dit-server.git
cd ed-dit-server
```

#### 2. Configure and Compile
Ensure `nvcc` is available in your `PATH` (typically `/opt/cuda/bin` or `/usr/local/cuda/bin`):
```bash
export PATH="/opt/cuda/bin:$PATH"

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

make -j$(nproc)
```

The build produces two self-contained binaries in `build/bin/`:
- `ed-dit-server`: HTTP microservice listening for continuous generation requests.
- `ed-dit-runner`: Standalone CLI execution tool for one-shot batch tasks.

---

### REST API Reference

The server exposes an HTTP REST API on `0.0.0.0:9005` (configurable via `--port` and `--host`).

#### 1. Health Probe (`GET /health`)
Returns real-time service health, loaded model metadata, active compute device, and graph memory budget.

- **Request**:
  ```bash
  curl -s http://127.0.0.1:9005/health
  ```
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "ed_dit_server_cpp",
    "engine": "native C++ in-process resident (zero python, zero rust)",
    "in_process_runner_resident": true,
    "device": "NVIDIA GeForce RTX 3070 Laptop GPU (CUDA0)",
    "model": "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf",
    "port": 9005,
    "budget_mb": 200,
    "threads": 8
  }
  ```

#### 2. Euler Denoise Sampling (`POST /sample`)
Triggers resident multistep Euler flow integration. Reads Stage 1 inputs, evaluates velocity fields, and writes Stage 2 denoised latents.

- **Request**:
  ```bash
  curl -X POST http://127.0.0.1:9005/sample \
    -H "Content-Type: application/json" \
    -d '{
      "latent_path": "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin",
      "output_path": "/mnt/R2D/tmp/minimax_bridge/stage2_denoised.pt",
      "steps": 6,
      "seed": 42,
      "budget_mb": 200
    }'
  ```
- **Parameters**:
  - `latent_path` (*string, required*): Absolute path to the Stage 1 `.edbin` bundle.
  - `output_path` (*string, optional*): Destination path for `.pt` (PyTorch format) or `.edlat` (raw binary). Defaults to `.pt`.
  - `steps` (*integer, optional*): Number of Euler integration steps (default: `6`).
  - `seed` (*integer, optional*): Random seed for Philox Gaussian noise initialization (default: `42`).
  - `budget_mb` (*integer, optional*): Dynamic graph-cut VRAM budget in megabytes (default: `200`).
- **Response**:
  ```json
  {
    "status": "success",
    "engine": "pure C++ in-process resident (zero python, zero rust)",
    "elapsed_seconds": 47.26188,
    "steps": 6,
    "file_size": 2871091,
    "out_path": "/mnt/R2D/tmp/minimax_bridge/stage2_denoised.pt"
  }
  ```

#### 3. Dynamic Budget Adjustment (`POST /budget`)
Dynamically adjusts the maximum graph VRAM threshold without restarting the service.

- **Request**:
  ```bash
  curl -X POST http://127.0.0.1:9005/budget \
    -H "Content-Type: application/json" \
    -d '{"budget_mb": 250}'
  ```
- **Response**:
  ```json
  {
    "status": "ok",
    "budget_mb": 250
  }
  ```

---

### End-to-End Pipeline Walkthrough

#### Step 1: Export ComfyUI Stage 1 Latents to `.edbin`
Use the included helper script to convert a ComfyUI Stage 1 `.pt` file into `.edbin`:
```bash
python3 scripts/export_stage1_for_ed.py stage1_input.pt stage1_input.edbin
```

#### Step 2: Query the Resident DiT Microservice
Run generation using `curl` or the Python client:
```bash
python3 scripts/sample_client.py stage1_input.edbin stage2_denoised.pt 6 200 http://127.0.0.1:9005
```

#### Step 3: Verify the Generated PyTorch `.pt` Bundle
The generated `.pt` file uses the PyTorch ZIP container format with `NestedTensor` layout:
```python
import torch

data = torch.load("stage2_denoised.pt", weights_only=False)
samples = data["samples"]["samples"]
video_latent = samples.tensors[0]  # shape: [1, 24, 22, 28, 48]
audio_latent = samples.tensors[1]  # shape: [1, 32, 2, 122]

print(f"Video latent shape: {video_latent.shape}, min={video_latent.min():.4f}, max={video_latent.max():.4f}")
print(f"Audio latent shape: {audio_latent.shape}, min={audio_latent.min():.4f}, max={audio_latent.max():.4f}")
```

---

### Daemon & Service Management

#### Option A: systemd
Copy the provided unit file to `/etc/systemd/system/`:
```bash
sudo cp services/systemd/ed-dit-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ed-dit-server
sudo systemctl status ed-dit-server
```

#### Option B: runit (Artix / Void Linux)
```bash
sudo mkdir -p /etc/service/ed-dit-server
sudo cp services/runit/run /etc/service/ed-dit-server/
sudo chmod +x /etc/service/ed-dit-server/run
sudo sv status ed-dit-server
```

---

### Performance Benchmarks

All tests were physically performed on an **NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, 80W TGP)** on Linux x86_64.

| Engine / Implementation | Model Format | Resolution & Duration | Steps | Step Time | Total Sampling Time | Dynamic VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python ComfyUI Baseline** | FP8 / BF16 | 768×448, 3s (73 frames) | 6 | ~15.20s | ~92.00s | ~5.8 GB |
| **Legacy ed-dit-runner (CLI)** | Q3_K_M GGUF | 768×448, 3s (73 frames) | 6 | ~17.01s | ~110.20s | ~2.5 GB |
| **ed-dit-server (This Work)** | **Q3_K_M GGUF** | **768×448, 3s (73 frames)** | **6** | **7.87s** | **47.26s** | **< 2.2 GB** |

> **Key takeaway**: `ed-dit-server` cuts total generation time by **57.1%** compared to the legacy CLI runner and **48.6%** compared to Python ComfyUI, while maintaining an operating VRAM footprint under **2.2 GB**.

---

### Physical Memory Limits (3s vs 1080p 8s)

Why does 1080p 8s fail on an 8GB GPU? Here is the exact physical accounting:

$$\text{Sequence Length} = \text{Prompt Tokens} + \left(\frac{1920}{2 \times 16}\right) \times \left(\frac{1088}{2 \times 16}\right) \times 57 = 120,721\text{ tokens}$$

$$\text{KV Workspace Bytes} = 2 \times 128 \times 120,721 \times 56 \times 2\text{ bytes} = \mathbf{3,460\text{ MiB (3.46 GB)}}$$

$$\text{Activation Compute Buffer} = \mathbf{5,080\text{ MiB (5.08 GB)}}$$

$$\text{Total Instantaneous VRAM Required} = 3.46\text{ GB} + 5.08\text{ GB} = \mathbf{8.54\text{ GB}} > 7.84\text{ GB (Usable 8GB VRAM)}$$

- **3s (768×448, 8,439 tokens)**: Compute buffer is ~1.5 GB + KV workspace is 246 MB. Peak dynamic VRAM is under 2.2 GB, allowing seamless operation on 8GB cards alongside other background workloads.
- **1080p 8s (120,721 tokens)**: Memory footprint hits 8.54 GB, which strictly exceeds the physical capacity of an 8GB GPU without spatial/temporal sub-tiling.

---

<a name="-繁體中文"></a>
## 🇹🇼 繁體中文

### 架構理念與核心優勢

在消費級 8GB 顯卡上運行 140 億（14B）參數量級的音視頻擴散模型（DiT），過去常面臨顯存不足（OOM）或依賴 CPU 漫長交換導致數分鐘以上的極慢延遲。`ed-dit-server` 透過三階段解耦流水線徹底根治了此痛點：

```
【第一階段：條件編碼（Stage 1）】
  主機內存 / CPU 或局域網算力節點
  Qwen3-VL 文本/圖像/音頻標記化 ──> stage1_input.edbin
               │
               ▼ HTTP POST (發送二進制路徑)
【第二階段：進程內常駐 DiT 速度場積分（本專案核心）】
  RTX 3070 8GB（動態顯存峰值嚴格壓制在 2.2 GB 以內）
  ed-dit-server（純 C++ 原生微服務）
  ├── 主機內存 GGUF 權重池（冷加載僅 1.8 秒，運行時零重複讀盤）
  ├── 細粒度動態計算圖切割引擎（Graph Cut，單段 ≤ 200MB）
  ├── 進程內 Flash Attention 與 AdaLN 調制極速核函數
  └── 內存直出合規 PyTorch ZIP/Pickle 二進制 ──> stage2_denoised.pt
               │
               ▼ 標準 PyTorch .pt
【第三階段：潛空間解碼（Stage 3）】
  主機內存 / CPU 或局域網算力節點
  MiniMax VAE 解碼 ──> 最終 MP4 視頻成品
```

#### 為何全面改寫為純 C++？
1. **徹底根除 Python 運行時負擔**：消除 GIL 鎖死、垃圾回收（GC）延遲與進程間通訊（IPC）封裝損耗。
2. **免除跨語言 Rust FFI 邊界拷貝**：杜絕 Rust 與 C++ 之間的雙重緩衝，內核至外層零內存複製。
3. **原生 CUDA 顯存精細化管線**：基於 `ggml-cuda` 實現毫秒級的主機-顯卡部分權重流式換入換出。

---

### 軟硬體環境要求

- **顯卡（GPU）**：NVIDIA GPU，計算架構（Compute Capability）≥ 8.6（Ampere RTX 3070, RTX 3060, RTX 4060 等），實體顯存 ≥ 8 GB。
- **主機內存（RAM）**：16 GB 以上（建議 32 GB，以便將 GGUF 模型全量常駐於內存）。
- **作業系統**：Linux x86_64（Arch, Artix, Ubuntu 20.04+, Debian 11+, RHEL 8+ 等）。
- **編譯器**：GCC ≥ 10.0 / Clang ≥ 12.0（需支持 C++17）。
- **CUDA 工具包**：CUDA ≥ 12.0（已於 CUDA 12.4 / 13.4 + `nvcc` 實測通過）。
- **CMake**：CMake ≥ 3.18。

---

### 編譯與安裝指引

#### 1. 克隆獨立開源倉庫
```bash
git clone https://github.com/obhasashare2024-namo/ed-dit-server.git
cd ed-dit-server
```

#### 2. 配置並執行編譯
請確保 `nvcc` 路徑已加入至環境變數 `PATH` 中（常見於 `/opt/cuda/bin` 或 `/usr/local/cuda/bin`）：
```bash
export PATH="/opt/cuda/bin:$PATH"

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

make -j$(nproc)
```

編譯完成後，二進制產物將生成於 `build/bin/` 目錄：
- `ed-dit-server`：常駐 HTTP 微服務，用於長跑流水線的高併發呼叫。
- `ed-dit-runner`：單次執行的 CLI 命令行測試工具。

---

### REST API 接口規範

服務啟動後，默認於 `0.0.0.0:9005` 提供高效能 HTTP 接口（可透過 `--port` 與 `--host` 自行調整）。

#### 1. 服務健康檢查 (`GET /health`)
實時返回服務健康狀態、模型掛載路徑、激活顯卡名稱及當前計算圖顯存預算。

- **請求示例**：
  ```bash
  curl -s http://127.0.0.1:9005/health
  ```
- **響應格式**：
  ```json
  {
    "status": "ok",
    "service": "ed_dit_server_cpp",
    "engine": "native C++ in-process resident (zero python, zero rust)",
    "in_process_runner_resident": true,
    "device": "NVIDIA GeForce RTX 3070 Laptop GPU (CUDA0)",
    "model": "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf",
    "port": 9005,
    "budget_mb": 200,
    "threads": 8
  }
  ```

#### 2. 多步 Euler 採樣生成 (`POST /sample`)
觸發進程內多步速度場流積分。讀取第一階段生成的 `.edbin` 文件，完成降噪並直接落地為 PyTorch `.pt` 文件。

- **請求示例**：
  ```bash
  curl -X POST http://127.0.0.1:9005/sample \
    -H "Content-Type: application/json" \
    -d '{
      "latent_path": "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin",
      "output_path": "/mnt/R2D/tmp/minimax_bridge/stage2_denoised.pt",
      "steps": 6,
      "seed": 42,
      "budget_mb": 200
    }'
  ```
- **參數說明**：
  - `latent_path`（*字符串，必填*）：第一階段輸入打包文件（`.edbin`）的絕對路徑。
  - `output_path`（*字符串，選填*）：生成產物的落地路徑，支持 `.pt`（PyTorch 兼容格式）或 `.edlat`（原生二進制）。
  - `steps`（*整數，選填*）：Euler 採樣步數（默認為 `6`）。
  - `seed`（*整數，選填*）：Philox 高斯噪聲隨機種子（默認為 `42`）。
  - `budget_mb`（*整數，選填*）：動態圖切割的單段顯存上限（默認為 `200` MB）。
- **響應格式**：
  ```json
  {
    "status": "success",
    "engine": "pure C++ in-process resident (zero python, zero rust)",
    "elapsed_seconds": 47.26188,
    "steps": 6,
    "file_size": 2871091,
    "out_path": "/mnt/R2D/tmp/minimax_bridge/stage2_denoised.pt"
  }
  ```

#### 3. 動態顯存預算熱調整 (`POST /budget`)
無需重啟服務，實時熱更新最大顯存預算門檻。

- **請求示例**：
  ```bash
  curl -X POST http://127.0.0.1:9005/budget \
    -H "Content-Type: application/json" \
    -d '{"budget_mb": 250}'
  ```

---

### 端到端完整生成工作流

#### 步驟 1：轉換 ComfyUI 第一階段潛空間為 `.edbin`
利用倉庫提供的轉換腳本，將 ComfyUI 產生的 `stage1_input.pt` 轉為純 C++ 可極速解析的二進制包：
```bash
python3 scripts/export_stage1_for_ed.py stage1_input.pt stage1_input.edbin
```

#### 步驟 2：呼叫常駐 DiT 微服務
可直接使用 `curl` 或 Python 客戶端腳本發起降噪請求：
```bash
python3 scripts/sample_client.py stage1_input.edbin stage2_denoised.pt 6 200 http://127.0.0.1:9005
```

#### 步驟 3：驗證並讀取生成產物
輸出的 `.pt` 文件遵循 PyTorch 標準 ZIP 規範與 `NestedTensor` 佈局，可直接透過 `torch.load()` 載入並餵入 ComfyUI VAE：
```python
import torch

data = torch.load("stage2_denoised.pt", weights_only=False)
samples = data["samples"]["samples"]
video_latent = samples.tensors[0]  # 形狀: [1, 24, 22, 28, 48]
audio_latent = samples.tensors[1]  # 形狀: [1, 32, 2, 122]

print(f"視頻潛空間維度: {video_latent.shape}, 最小值={video_latent.min():.4f}, 最大值={video_latent.max():.4f}")
print(f"音頻潛空間維度: {audio_latent.shape}, 最小值={audio_latent.min():.4f}, 最大值={audio_latent.max():.4f}")
```

---

### 後台常駐與系統服務配置

#### 方案一：systemd 服務配置
將服務模板拷貝至 `/etc/systemd/system/`：
```bash
sudo cp services/systemd/ed-dit-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ed-dit-server
sudo systemctl status ed-dit-server
```

#### 方案二：runit 服務配置（適用於 Artix / Void Linux）
```bash
sudo mkdir -p /etc/service/ed-dit-server
sudo cp services/runit/run /etc/service/ed-dit-server/
sudo chmod +x /etc/service/ed-dit-server/run
sudo sv status ed-dit-server
```

---

### 物理實測性能數據

以下數據均於 **NVIDIA GeForce RTX 3070 Laptop GPU（8GB 顯存，80W 功耗上限）** Linux 主機上物理實測：

| 引擎架構與實現方案 | 模型權重格式 | 分辨率與長度 | 積分步數 | 單步前向耗時 | 全流程採樣耗時 | 動態峰值顯存 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python ComfyUI 基準** | FP8 / BF16 | 768×448, 3秒 (73幀) | 6 步 | ~15.20 秒 | ~92.00 秒 | ~5.8 GB |
| **舊版單進程 ed-dit-runner (CLI)** | Q3_K_M GGUF | 768×448, 3秒 (73幀) | 6 步 | ~17.01 秒 | ~110.20 秒 | ~2.5 GB |
| **ed-dit-server (本次重構)** | **Q3_K_M GGUF** | **768×448, 3秒 (73幀)** | **6 步** | **7.87 秒** | **47.26 秒** | **< 2.2 GB** |

> **效能結論**：重構為純 C++ 常駐微服務後，耗時相較舊版 CLI 縮短了 **57.1%**，相較原生 Python ComfyUI 縮短了 **48.6%**，且動態顯存穩定錨定在 **2.2 GB 以下**。

---

### 顯存物理邊界深入解析 (3s vs 1080p 8s)

為何 1080p 8s 無法直接運行於 8GB 顯卡？以下為底層精確算術核算：

$$\text{序列長度（Tokens）} = \text{提示詞長度} + \left(\frac{1920}{2 \times 16}\right) \times \left(\frac{1088}{2 \times 16}\right) \times 57 = 120,721\text{ Tokens}$$

$$\text{KV 工作區顯存需求} = 2 \times 128 \times 120,721 \times 56 \times 2\text{ 字節} = \mathbf{3,460\text{ MiB (3.46 GB)}}$$

$$\text{計算圖激活顯存（Compute Buffer）} = \mathbf{5,080\text{ MiB (5.08 GB)}}$$

$$\text{瞬時顯存總需求} = 3.46\text{ GB} + 5.08\text{ GB} = \mathbf{8.54\text{ GB}} > 7.84\text{ GB (8GB 卡物理可用顯存)}$$

- **3s 輕量任務（768×448，8,439 Tokens）**：計算圖激活值約 1.5 GB，KV 工作區僅 246 MB，峰值顯存 < 2.2 GB，可在 8GB 卡上與其他後台進程完美並存。
- **1080p 8s 任務（120,721 Tokens）**：瞬時顯存需求高達 8.54 GB，已從物理上擊穿 8GB 顯存上限。若要支援該規格，後續需進一步引入分塊注意力（Tiled Attention）或層間跨步 KV 主機卸載機制。

---

<a name="-日本語"></a>
## 🇯🇵 日本語

### アーキテクチャと設計思想

140億パラメータ（14B）の大規模動画生成モデルは、通常 24GB 以上の VRAM を搭載したデータセンター向け GPU を必要とします。`ed-dit-server` は、3段階の分離パイプラインにより、一般的な 8GB VRAM GPU（RTX 3070 など）での快適な動画生成を実現します：

```
【ステージ 1: 条件エンコード】
  ホスト RAM / CPU または LAN 内の計算ノード
  Qwen3-VL テキスト/画像/音声トークナイズ ──> stage1_input.edbin
               │
               ▼ HTTP POST (バイナリパスの送信)
【ステージ 2: 常駐型 DiT 速度場積分（本プロジェクト）】
  RTX 3070 8GB（動的 VRAM ピークを 2.2 GB 未満に抑制）
  ed-dit-server（純 C++ 常駐マイクロサービス）
  ├── ホスト RAM 内 GGUF 重みプール（起動時 1.8 秒ロード、以降ゼロ再読込）
  ├── 微細粒度動的グラフカットエンジン（各セグメント ≤ 200MB）
  ├── プロセス内 Flash Attention および AdaLN 変調高速パス
  └── メモリ内ダイレクト PyTorch ZIP/Pickle 生成 ──> stage2_denoised.pt
               │
               ▼ 標準 PyTorch .pt
【ステージ 3: 潜在空間デコード】
  ホスト RAM / CPU または LAN 内の計算ノード
  MiniMax VAE デコード ──> 最終 MP4 動画出力
```

#### なぜ純粋な C++ なのか？
1. **Python のオーバーヘッドを完全排除**：GIL、ガベージコレクション（GC）の停止時間、プロセス間通信（IPC）のオーバーヘッドをゼロに。
2. **言語間 FFI コピーの解消**：Rust と C++ 間の不要なメモリ往復バッファリングを完全に排除。
3. **正確な CUDA メモリ制御**：`ggml-cuda` に基づく高精度な部分パラメータストリーミングを実現。

---

### 動作要件

- **GPU**: NVIDIA GPU（Compute Capability ≥ 8.6、RTX 3070, RTX 3060, RTX 4060 など）、8GB 以上の VRAM。
- **ホストメモリ（RAM）**: 16 GB 以上（GGUF 重みを保持するため 32 GB 以上を推奨）。
- **OS**: Linux x86_64（Arch, Artix, Ubuntu 20.04+, Debian 11+ など）。
- **コンパイラ**: GCC ≥ 10.0 / Clang ≥ 12.0（C++17 対応）。
- **CUDA Toolkit**: CUDA ≥ 12.0（CUDA 12.4 / 13.4 + `nvcc` で動作確認済み）。
- **CMake**: CMake ≥ 3.18。

---

### ビルドとインストール手順

#### 1. リポジトリのクローン
```bash
git clone https://github.com/obhasashare2024-namo/ed-dit-server.git
cd ed-dit-server
```

#### 2. ビルドの実行
`nvcc` が `PATH` に含まれていることを確認してください：
```bash
export PATH="/opt/cuda/bin:$PATH"

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

make -j$(nproc)
```

ビルド完了後、`build/bin/` にバイナリが生成されます：
- `ed-dit-server`: HTTP マイクロサービスデーモン。
- `ed-dit-runner`: 単発実行用 CLI ツール。

---

### REST API 仕様

サービス起動後、デフォルトで `0.0.0.0:9005` にて HTTP リクエストを受け付けます。

#### 1. ヘルスチェック (`GET /health`)
- **リクエスト**:
  ```bash
  curl -s http://127.0.0.1:9005/health
  ```
- **レスポンス**:
  ```json
  {
    "status": "ok",
    "service": "ed_dit_server_cpp",
    "engine": "native C++ in-process resident (zero python, zero rust)",
    "in_process_runner_resident": true,
    "device": "NVIDIA GeForce RTX 3070 Laptop GPU (CUDA0)",
    "model": "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf",
    "port": 9005,
    "budget_mb": 200,
    "threads": 8
  }
  ```

#### 2. サンプリング生成 (`POST /sample`)
- **リクエスト**:
  ```bash
  curl -X POST http://127.0.0.1:9005/sample \
    -H "Content-Type: application/json" \
    -d '{
      "latent_path": "/path/to/stage1_input_3s.edbin",
      "output_path": "/path/to/stage2_denoised.pt",
      "steps": 6,
      "seed": 42,
      "budget_mb": 200
    }'
  ```
- **レスポンス**:
  ```json
  {
    "status": "success",
    "engine": "pure C++ in-process resident (zero python, zero rust)",
    "elapsed_seconds": 47.26188,
    "steps": 6,
    "file_size": 2871091,
    "out_path": "/path/to/stage2_denoised.pt"
  }
  ```

---

### ベンチマーク測定結果

**NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, 80W TGP)** 実機での測定結果：

| 構成 / 実装 | モデル形式 | 解像度・動画長 | ステップ数 | ステップ時間 | 合計生成時間 | ピーク動的 VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python ComfyUI** | FP8 / BF16 | 768×448, 3秒 (73フレーム) | 6 | ~15.20 秒 | ~92.00 秒 | ~5.8 GB |
| **旧 CLI ランナー** | Q3_K_M GGUF | 768×448, 3秒 (73フレーム) | 6 | ~17.01 秒 | ~110.20 秒 | ~2.5 GB |
| **ed-dit-server (本実装)** | **Q3_K_M GGUF** | **768×448, 3秒 (73フレーム)** | **6** | **7.87 秒** | **47.26 秒** | **< 2.2 GB** |

---

### VRAM 物理制限の分析 (3s と 1080p 8s)

なぜ 1080p 8s は 8GB GPU で OOM となるのか？物理メモリの計算内訳は以下の通りです：

$$\text{シーケンス長} = 120,721\text{ トークン}$$
$$\text{KV ワークスペース容量} = 2 \times 128 \times 120,721 \times 56 \times 2 = \mathbf{3.46\text{ GB}}$$
$$\text{計算グラフ活性化容量} = \mathbf{5.08\text{ GB}}$$
$$\text{必要 VRAM 合計} = 3.46\text{ GB} + 5.08\text{ GB} = \mathbf{8.54\text{ GB}} > 7.84\text{ GB (実効利用可能容量)}$$

- **3s (768×448, 8,439 トークン)**: 必要 VRAM は 2.2 GB 未満に収まり、8GB GPU で他タスクと安全に共存可能。
- **1080p 8s (120,721 トークン)**: 必要 VRAM が 8.54 GB に達し、8GB カードの物理限界を超過します。

---

## License
Apache-2.0 License. Copyright (c) 2026.
