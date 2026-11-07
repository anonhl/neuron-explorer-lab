# Neuron Explorer：Windows 离线使用详细教程

[English](OFFLINE_WINDOWS.md) | **简体中文**
本文针对当前仓库，使用 Windows PowerShell 命令。目标是在准备完成后，关闭外网，仍能通过本机浏览器使用神经元地图、缓存分析，以及 Qwen3.5-4B 的本地推理和干预。

**离线运行需要同时准备 Python 依赖、模型文件、前端 Three.js 和项目分析数据。只下载模型还不够。** 本文中的下载步骤需要联网执行；启动步骤在断网后执行。

当前项目已在 `explorer/static/vendor/three/` 内提供 Three.js 0.169.0，并已把页面加载地址改为本地路径；复制项目时带上该目录即可。模型下载、Python 依赖安装仍需按本文准备。

## 1. 先选择使用方式

| 使用方式 | 必需材料 | 可以做什么 | 限制 |
|---|---|---|---|
| A：只浏览缓存 | Python 环境、模型的 `config.json`、本地 Three.js、随仓库提供的 `explorer/cache/` | 地图、缓存 lens 搜索、预计算连接；有实验数据时可查看历史结果 | 不加载权重，不能生成文本、计算实时激活或使用配置干预；部分自动请求的注意力分析会报模型已禁用 |
| B：本地模型推理 | A 的材料，加完整 Qwen3.5-4B 模型与 tokenizer | 本地聊天、神经元加法/乘法干预、激活分析、应用保存的神经元配置 | 需要足够的内存/显存；长文本和注意力分析会增加占用 |

两条路线都需要安装当前项目的 Python 依赖。`EXPLORER_NO_MODEL` 只控制权重加载，并不会移除服务对 PyTorch、Transformers 等包的导入。

本地 Qwen 聊天与界面中的 Claude/Codex 智能体是不同入口。本文的离线流程使用前者，不依赖后者，也不包含访问 neuronary 在线社区。原有的 vLLM 神经元扫描流水线仍面向 Linux/CUDA。

## 2. 约定目录与工具

示例把项目放在：

```text
D:\AI\neuron-explorer-main
```

请按自己的位置修改下面的 `Set-Location`。所有命令默认在项目根目录执行；代码块中的命令不要附加 `PS>` 提示符。

```powershell
Set-Location "D:\AI\neuron-explorer-main"
```

准备以下工具：

- Windows 64 位 Python 3.12 完整安装版，以及 Python Launcher（命令 `py`）。为方便复现，准备机与目标机使用相同的 Python 3.12 补丁版本。
- uv，用于联网安装项目依赖。可以从 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装；安装后重新打开 PowerShell。
- 支持 WebGL 和 import map 的浏览器，例如已安装并能正常使用本项目的 Edge 或 Chrome。
- 准备启用 CUDA 时，安装适合目标 NVIDIA 显卡的驱动。提前保留驱动和 Python 的完整离线安装程序。
- 下载、解压和环境重建会占用额外空间，建议预留几十 GB，最终以实际文件大小为准。

确认工具：

```powershell
py -3.12 --version
uv --version
tar --version
```

本教程不要求激活虚拟环境，后续直接调用 `.venv\Scripts\python.exe`，可以避免 PowerShell 的激活脚本执行策略问题。

## 3. 联网准备：安装项目依赖

### 3.1 基础环境

先确认当前窗口没有之前留下的离线设置：

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue

uv sync --locked --python 3.12
```

安装完成后检查关键包：

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, numpy, fastapi, uvicorn; from transformers import Qwen3_5ForConditionalGeneration; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('imports OK')"
```

先使用仓库的锁文件，不要为解决单个导入错误而单独升级 Transformers 或 PyTorch。当前推理代码会调用 Qwen3.5 的内部实现，版本不匹配可能影响模型加载和激活捕获。

如果 `uv sync --locked` 提示锁文件中的版本不可用或没有适用的 Windows 包，应在联网阶段解决并重新验证。没有可安装的依赖集时，复制文件也无法补齐环境。

### 3.2 可选：安装权重量化后端

只浏览缓存，或使用未量化的 BF16/FP16 时，可以跳过这一小节。

按实际需要选择一种命令，不必全部执行：

```powershell
# 只安装 bitsandbytes
uv sync --locked --python 3.12 --extra bitsandbytes

# 或者，只安装 HQQ
uv sync --locked --python 3.12 --extra hqq

# 或者，同时安装两个后端
uv sync --locked --python 3.12 --extra quantization
```

项目仅支持在 NVIDIA CUDA 上进行量化加载；CPU/MPS 使用未量化权重。量化的是部分线性层权重，激活、输出头等仍占用浮点内存，因此不能仅按“4B × 4 bit”估算全部显存需求。

需要加载本地 LoRA 时，联网准备阶段另选 `uv sync --locked --python 3.12 --extra lora`，把 PEFT 纳入同一锁文件管理；操作步骤见 [启动教程的 LoRA 小节](START_WINDOWS.zh-CN.md)。要同时保留量化后端和 LoRA 依赖，使用 `--extra quantization --extra lora`。后续再次执行 `uv sync` 时仍需带上所需 extras，否则可选依赖可能被移除。LoRA 仅可在全量 BF16/FP16 权重下加载。

### 3.3 确认 PyTorch 是否能使用显卡

如果选择路线 B 且希望使用 NVIDIA 显卡，运行：

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`nvidia-smi` 能看到显卡，并不代表当前 Python 环境中的 PyTorch 可以使用 CUDA；以 `torch.cuda.is_available()` 和后续实际推理为准。

如果需要更换 CUDA 构建，按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Windows/Pip/CUDA，并把安装命令用于本项目的 Python 环境。先安装 pip：

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

然后将官方命令开头的 `pip` 或 `pip3` 替换为 `.\.venv\Scripts\python.exe -m pip`。保留官方给出的包名和索引地址，同时确认所选版本与项目其余依赖兼容，重新执行上一小节的导入检查和本小节的 CUDA 检查。

后续打包以**实际通过检查和推理验收的环境**为准。手动调整过 PyTorch 后，直接用虚拟环境的 Python 启动；再次运行 `uv sync` 或默认的 `uv run` 可能重新同步依赖，改变这一环境。

## 4. 联网准备：下载模型到固定目录

采用普通本地目录保存模型，便于复制到另一台机器。底模放在 `models\Qwen3.5-4B\Qwen3.5-4B`，可选 LoRA 放在 `models\Qwen3.5-4B\my-lora`；外层 `Qwen3.5-4B` 是归组目录，`EXPLORER_MODEL` 指向内层底模目录。`snapshot_download` 支持 `local_dir`，会保留仓库文件结构，详见 [Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

### 4.1 路线 B：完整本地推理模型

在项目根目录运行以下整段命令：

```powershell
@'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "Qwen/Qwen3.5-4B"
revision = HfApi().model_info(repo).sha
directory = "models/Qwen3.5-4B/Qwen3.5-4B"
snapshot_download(repo_id=repo, revision=revision, local_dir=directory)
Path(directory, "offline-source.json").write_text(
    json.dumps({"repo_id": repo, "revision": revision, "download": "full"}, indent=2),
    encoding="utf-8",
)
print("Downloaded:", directory)
print("Revision:", revision)
'@ | .\.venv\Scripts\python.exe -
```

记录下载时的提交编号，方便后续复现实验。本流程下载原始模型，bitsandbytes/HQQ 在加载时进行量化；不要把 GGUF、AWQ 或其他不同加载格式的文件当作这个目录的替代品。

检查本地配置、tokenizer 和权重分片是否齐全：

```powershell
@'
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

p = Path("models/Qwen3.5-4B/Qwen3.5-4B")
cfg = AutoConfig.from_pretrained(str(p), local_files_only=True).get_text_config()
tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
index = p / "model.safetensors.index.json"
if index.exists():
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
else:
    shards = {"model.safetensors"}
missing = [name for name in sorted(shards) if not (p / name).is_file()]
if missing:
    raise SystemExit("Missing weights: " + ", ".join(missing))
print("Layers:", cfg.num_hidden_layers)
print("MLP neurons per layer:", cfg.intermediate_size)
print("Tokenizer size:", len(tok))
print("Weight files present:", len(shards))
'@ | .\.venv\Scripts\python.exe -
```

此检查确认文件存在并能读取配置/tokenizer，不代替加载全部权重。真正的模型验收在第 8 节进行。

### 4.2 路线 A：只下载启动必需的配置

如果只浏览随仓库提供的缓存，可以用下面这段**代替 4.1 节的完整模型下载**：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B/Qwen3.5-4B', allow_patterns=['config.json'])"
```

当前 `server.py` 在启动时无条件读取模型配置，因此即使不加载权重，也必须准备 `config.json`。此外必须保留 `explorer/cache/vocab.json`，否则服务可能尝试读取 tokenizer。此路线不执行 4.1 节的 tokenizer/权重分片检查。

以后要改为本地推理，需要联网重新执行 4.1 节，补齐完整模型。

## 5. 联网准备：把 Three.js 改成本地资源

**当前项目已经完成此步骤，可以直接跳到第 6 节。** `explorer/static/index.html` 的 import map 已指向随项目提供的 `vendor/three/`，不再从 jsDelivr 加载 Three.js。可用第 5.1 节末尾的 `Test-Path` 命令检查文件。

下面保留从原始 CDN 版本恢复本地资源的完整步骤，供目录丢失或重新下载上游源码时使用。浏览器曾访问过页面，不能保证换电脑、清缓存或无痕窗口后仍能离线打开地图。

### 5.1 下载与解压

保持与项目一致的 `0.169.0` 版本，下载完整 npm 包，保留插件之间的相对导入关系及许可证：

```powershell
New-Item -ItemType Directory -Force ".\offline-cache" | Out-Null
New-Item -ItemType Directory -Force ".\explorer\static\vendor\three" | Out-Null

Invoke-WebRequest -UseBasicParsing -Uri "https://registry.npmjs.org/three/-/three-0.169.0.tgz" -OutFile ".\offline-cache\three-0.169.0.tgz"
tar -xzf ".\offline-cache\three-0.169.0.tgz" -C ".\explorer\static\vendor\three" --strip-components=1
if ($LASTEXITCODE -ne 0) { throw "Three.js extraction failed" }
```

检查这些文件都存在，结果应全部为 `True`：

```powershell
Test-Path ".\explorer\static\vendor\three\build\three.module.js"
Test-Path ".\explorer\static\vendor\three\examples\jsm\controls\OrbitControls.js"
Test-Path ".\explorer\static\vendor\three\examples\jsm\lines\LineSegments2.js"
Test-Path ".\explorer\static\vendor\three\LICENSE"
```

### 5.2 修改 import map

先备份，再执行替换：

```powershell
Copy-Item ".\explorer\static\index.html" ".\offline-cache\index.before-offline.html"

@'
from pathlib import Path

p = Path("explorer/static/index.html")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
    "./vendor/three/build/three.module.js",
).replace(
    "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
    "./vendor/three/examples/jsm/",
)
if '"three": "./vendor/three/build/three.module.js"' not in text:
    raise SystemExit("Unexpected import map; inspect index.html before continuing")
if '"three/addons/": "./vendor/three/examples/jsm/"' not in text:
    raise SystemExit("Missing local addons import map")
p.write_text(text, encoding="utf-8")
print("Three.js now loads from local files")
'@ | .\.venv\Scripts\python.exe -
```

最终应为：

```json
{"imports": {
  "three": "./vendor/three/build/three.module.js",
  "three/addons/": "./vendor/three/examples/jsm/"
}}
```

以后更新或覆盖 `index.html`，需要重新确认这两项。不要只复制 `three.module.js`，也不要只复制最上层几个插件；它们还有依赖文件。Three.js 的模块与 import map 机制可参考 [官方手册的 installation 页面](https://threejs.org/manual/#en/installation)。

## 6. 联网准备：保留分析缓存与可选实验数据

当前项目已包含的基础缓存如下，复制项目时全部保留：

```text
explorer/cache/
├── act_scale.npz
├── connections.npz
├── jpp_index.npz
├── layout_writes.npz
├── lens_index.npz
├── r_index.npz
├── s_index.npz
├── steer_index.npz
└── vocab.json
```

这些缓存针对原始 Qwen3.5-4B。把 `EXPLORER_MODEL` 指向同一模型的本地目录即可，不要直接替换成其他尺寸或架构并继续使用同一批缓存。

**不要为“释放空间”删除 `act_scale.npz`。** 加载实时模型时，如果它不存在，会尝试通过在线流式 FineWeb-Edu 数据集重建。离线标志无法替你生成缺失的数据。删除 `lens_index.npz` 也会触发额外的索引重建计算。

按需求补齐以下文件：

| 需要的功能 | 需要复制到项目里的文件 | 缺少时的表现 |
|---|---|---|
| 历史生成示例、神经元排名 | `runs/<run>.jsonl`、同名 `<run>.meta.json`；建议同时带 `<run>.units.jsonl` | 没有对应层的历史结果；只有 jsonl、没有 meta 时不会被发现 |
| 历史结果的 embeddings 视图 | 同一 run 的 `<run>.umap.json`，以及上述 run 文件 | embeddings 没有数据 |
| 原始实验的残差尺度 | `stats/q35-4b_L<层号>.npz` | 实时干预使用内置参考文本估算尺度，不会因此自动下载原始 stats |
| 网页激活上下文与领域统计 | `explorer/cache/ffw/corpus.npz`、`urls.json`、`acts.npz` | 网页示例和领域统计为空 |
| 超过预计算范围的 R/J++ lens 计算 | `explorer/cache/rjpp/` 下的原始发布文件 | 只能使用已有索引支持的结果 |

当前仓库副本没有 `runs/`、`stats/` 和 `explorer/cache/ffw/`。这些是实验产物，不能通过 `uv sync` 补出，也不能因为启动成功就认为已具备。

常规预计算 R/J++ lens 展示已有索引，不要求下载原始映射；只有确实需要额外计算时，才在联网阶段执行：

```powershell
.\.venv\Scripts\hf.exe download bubskee/qwen3.5-4b-r-jpp-lenses --local-dir explorer/cache/rjpp
```

单纯查看已有 embeddings 不需要再下载嵌入模型。如果想重新运行 `explorer/build_embeddings.py`，则还需提前下载它使用的 `BAAI/bge-small-en-v1.5`，并通过脚本的 `--model` 参数指向本地目录。这与普通离线启动是两件事。

## 7. 断网启动：路线 A，只浏览缓存

完成依赖、配置、Three.js 和缓存准备后，关闭外网，在项目根目录打开 PowerShell：

```powershell
Set-Location "D:\AI\neuron-explorer-main"

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:UV_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
$env:EXPLORER_NO_MODEL = "1"
$env:EXPLORER_QUANTIZATION = "none"
$env:EXPLORER_COMPUTE_DTYPE = "bfloat16"

.\.venv\Scripts\python.exe explorer\server.py
```

设置 `HF_HUB_OFFLINE=1` 会阻止 Hugging Face Hub 的 HTTP 请求，缺少本地文件时会报错，详见 [官方离线变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)。这些变量仅约束对应组件，不能代替物理断网，也不会自动把 CDN 改成本地文件。

浏览器打开 [本机 Explorer](http://127.0.0.1:8791/)。保持 PowerShell 窗口开启，结束时按 `Ctrl+C`。

另开一个 PowerShell 窗口检查状态：

```powershell
Invoke-RestMethod "http://127.0.0.1:8791/api/status" | ConvertTo-Json -Depth 6
Invoke-RestMethod "http://127.0.0.1:8791/api/lens?layer=22&n=2742" | ConvertTo-Json -Depth 6
```

预期状态是 `no model`，lens 接口可返回缓存词表。触发不支持的实时分析请求时，对应功能会提示模型已禁用。

可以查看地图、搜索 lens 和预计算神经元连接。有 run 数据时，还可以查看历史示例。当前界面并没有对缓存模式的所有控件做降级处理：打开神经元时会自动请求注意力头分析，可能出现模型禁用相关错误；实时激活、注意力计算和生成入口也不能使用。这不等于缓存文件损坏。需要完整的交互分析时使用路线 B。

## 8. 断网启动：路线 B，本地推理与干预

此路线要求第 4.1 节的完整模型已经下载，并且已验证使用的 PyTorch/量化后端可在目标硬件上工作。

### 8.1 在网页中选择加载方式

| 加载方式 | 面板选项 | 条件 |
|---|---|---|
| BF16 | BF16（未量化） | 设备能够运行该精度，内存/显存足够 |
| FP16 | FP16（未量化） | 可用于不适合 BF16 的设备，需观察数值稳定性 |
| bitsandbytes 4 位权重 | 量化加载权重 → bitsandbytes → NF4（4 位），计算精度选 BF16 或 FP16 | NVIDIA CUDA + 已安装 bitsandbytes |
| HQQ 4 位权重 | 量化加载权重 → HQQ → INT4（4 位），计算精度选 BF16 或 FP16 | NVIDIA CUDA + 已安装 HQQ |

项目默认 BF16；RTX 2080 Ti 等缺少原生 BF16 支持的设备可尝试 FP16。FP16 数值范围更小，选择后仍要验证生成和激活是否正常。CPU 可以尝试未量化加载，但不保证性能与全部分析操作满足实际需求。Runtime 自动选择 CUDA → MPS → CPU，当前没有强制设备的环境变量。

### 8.2 启动服务

```powershell
Set-Location "D:\AI\neuron-explorer-main"

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:UV_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

**不要把 `EXPLORER_NO_MODEL` 设置成字符串 `"0"`。** 当前实现判断的是变量是否为非空字符串，`"0"` 也会禁用模型。启用实时模型必须删除这个变量。

浏览器打开 [本机 Explorer](http://127.0.0.1:8791/)，在「加载模型」面板选好加载方式，再点击「加载」。页面随后显示加载阶段和权重进度，就绪后进入探索器。也可检查 `/api/status`，点击前 `status` 为 `idle`，就绪后为 `ready`。打开第一个神经元时，注意力相关的首次计算可能继续耗时。

模型加载失败时，页面会直接显示原因。解决问题或修改选择后，可再次点击「加载」。

面板会预选上一次成功加载的设置。`EXPLORER_QUANTIZATION` 和 `EXPLORER_COMPUTE_DTYPE` 可覆盖预选项。离线阶段直接调用已安装的 Python，避免为了启动而触发新的依赖同步。

### 8.3 第一次操作练习

1. 在页面右上角选择 **中文** 或 **English**。刷新后会保留所选语言。
2. 在跳转输入框输入 `L22 N2742`，按 Enter，打开该神经元。
3. 进入「对话」，选择「评测」和「偏移对比」，点击「添加当前神经元」。在新增行选择「偏移」，先尝试较小剂量，例如 `0.2`。
4. 输入 `Who are you?`。比较 `+`、`0`、`−` 三列，再逐步调整剂量。
5. 进入左侧 run/推理面板，输入一两句话，观察不同 token 对应的激活。初次先用短文本。
6. 要查看输入文本上的激活，在聊天的「评测」中选择「读取激活」。可以输入其他神经元编号并点击「添加神经元」，一次查看这些神经元各自的结果。

剂量越大不一定越有解释价值。重复、乱码和输出崩坏也可能来自过强干预。缓存 lens 的关键词与实时输出不会必然一致，量化或更换精度也可能改变结果。

三列聊天中的乘法干预尤其需要注意：当前实现分别使用 `×dose`、`×1`、`×(-dose)`。例如 dose 为 `2` 时，负向列是乘以 `-2`，并非除以 `2`。要做明确的神经元消融，使用配置批次中的乘法 `0`。

### 8.4 保存与复用神经元配置

1. 打开顶部“神经元配置”。
2. 新建一个分组，例如“离线实验”。
3. 添加操作，填写层号 `0–31`、神经元编号 `0–9215`。
4. `add` 使用带正负号的相对剂量；`mul` 使用直接乘数，`0` 表示置零。
5. 填写批次名称和备注，保存后点击“使用”。
6. 配置会影响后续实时推理；“停用”可以移除效果。

同一神经元上的操作按保存顺序执行。激活了配置批次时，聊天的 `0` 列也包含该配置，表示“没有本次临时调整”，不能当成完全未干预的模型。想恢复原始基准，先停用配置。

批次存于 `explorer/cache/neuron_batches.json`。刷新网页保留当前服务器进程里的激活批次；重启服务或重新加载模型会清除当前激活状态，需要再点“使用”。批次还会匹配模型标识、revision、量化格式和计算精度。

本教程统一使用 `models/Qwen3.5-4B/Qwen3.5-4B` 作为模型标识并从项目根目录启动。以后改成绝对路径或其他字符串，旧批次可能被判定为不匹配；本地目录加载时 revision 也可能为空，因此保留 `offline-source.json`，避免在同名目录里悄悄更换模型权重。

## 9. 迁移到另一台完全离线的 Windows 电脑

如果仍在同一台机器、同一目录运行，可以跳过本节。迁移时需重新建立虚拟环境，不能只复制 `.venv`。Python 官方说明虚拟环境通常不具备可搬移性，参见 [venv 文档](https://docs.python.org/3/library/venv.html)。

### 9.1 在联网 Windows 准备机制作 wheel 安装包

准备机与目标机使用相同的 Windows 架构、Python 3.12 补丁版本，尽量使用相同 GPU/驱动配置。不要直接用当前 macOS 环境生成 Windows 依赖包。以下步骤在已经通过运行验证的 Windows 环境执行。

先给虚拟环境补充 pip，导出实际版本：

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
New-Item -ItemType Directory -Force ".\offline-cache\wheelhouse" | Out-Null
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding UTF8 ".\offline-cache\requirements-offline.txt"
Get-Content ".\offline-cache\requirements-offline.txt"
```

如果清单包含 `-e`、`@ file:`、某个盘符的本地路径或 Git 地址，需要先把这些包制作成可分发 wheel，并使用对应版本说明，避免目标机继续查找准备机路径。正常的本项目依赖环境应主要是 `包名==版本`。

确保清除了第 7/8 节的离线设置，再构建或下载全部包的 wheel。命令会根据**已经安装的 PyTorch**判断是否需要额外的官方 CUDA wheel 索引：

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue

$cudaBuild = .\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda or '')"
$wheelArgs = @("wheel", "--no-deps", "--wheel-dir", ".\offline-cache\wheelhouse", "-r", ".\offline-cache\requirements-offline.txt")
if ($cudaBuild) {
    $cudaTag = "cu" + $cudaBuild.Trim().Replace(".", "")
    $wheelArgs += @("--extra-index-url", "https://download.pytorch.org/whl/$cudaTag")
}
.\.venv\Scripts\python.exe -m pip @wheelArgs
if ($LASTEXITCODE -ne 0) { throw "Wheel preparation failed; do not transfer an incomplete package" }
```

`--no-deps` 在这里成立的前提是清单来自完整运行环境，包含全部已安装依赖。`pip wheel` 会将需要的源码包在联网阶段构建为 wheel，离线目标机不用再次下载构建工具，机制参见 [pip wheel 文档](https://pip.pypa.io/en/stable/cli/pip_wheel/)。

CUDA 构建的索引目录必须确实提供当前版本。自动拼接出的索引不适用于某些预览版或特殊构建时，使用你安装该构建时实际采用的官方索引，重新执行打包。不要为了打包成功，把已验证的 CUDA 包替换成 CPU 包。

可选量化后端也必须在导出前安装并验证。复制 wheel 文件不等于后端已在目标 GPU 上可用。

### 9.2 在准备机先演练一次离线安装

创建一个新的、尚不存在的检查环境，完全从本地 wheel 安装：

```powershell
py -3.12 -m venv .offline-check
.\.offline-check\Scripts\python.exe -m pip --isolated install --no-index --only-binary=:all: --find-links ".\offline-cache\wheelhouse" -r ".\offline-cache\requirements-offline.txt"
if ($LASTEXITCODE -ne 0) { throw "Offline installation rehearsal failed" }
.\.offline-check\Scripts\python.exe -m pip check
.\.offline-check\Scripts\python.exe -c "import torch, transformers, fastapi, numpy; from transformers import Qwen3_5ForConditionalGeneration; print('offline environment OK'); print('CUDA:', torch.cuda.is_available())"
```

`--no-index` 关闭包索引查询，`--find-links` 指定本地包目录，`--only-binary` 确保没有落到源码构建；参见 [pip 本地安装选项](https://pip.pypa.io/en/stable/cli/pip_install/#install-no-index)。

完成安装后，关闭外网，使用 `.offline-check\Scripts\python.exe` 代替 `.venv\Scripts\python.exe`，按第 7 或第 8 节实际启动一次。尤其要验证模型生成和浏览器地图，不能只看导入成功。演练失败时先补齐材料，再复制到目标机。

### 9.3 需要带走的内容

```text
离线介质/
├── installers/                       # Python 完整安装程序、目标驱动等
└── neuron-explorer-main/
    ├── explorer/                     # 包括 cache 和已本地化的 static/vendor/three
    ├── sweep/                        # server 会导入其中的 rank.py
    ├── models/
    │   └── Qwen3.5-4B/
    │       ├── Qwen3.5-4B/            # 完整模型，或缓存模式使用的配置
    │       └── my-lora/               # 可选：LoRA 配置与权重
    ├── runs/                         # 有需要时携带
    ├── stats/                        # 有需要时携带
    ├── offline-cache/
    │   ├── requirements-offline.txt
    │   └── wheelhouse/*.whl
    ├── pyproject.toml
    ├── uv.lock
    ├── README.md
    └── docs/
```

`sweep/rank.py` 是 Explorer 服务的导入依赖，即使不运行云端 sweep，也不要漏掉它。

`.venv` 和 `.offline-check` 不作为可移植安装包。复制大模型时使用能保存大于 4 GB 单文件的介质格式；可生成校验清单，并在复制后验证：

```powershell
Get-ChildItem ".\models", ".\offline-cache\wheelhouse", ".\explorer\cache" -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv ".\offline-cache\sha256.csv" -NoTypeInformation -Encoding UTF8
```

该清单的 `Path` 含准备机的绝对路径；换盘符后要按相对项目位置查找对应文件，再比较 `Hash`，不要把旧路径直接用于目标机校验。

### 9.4 在离线目标机安装

先安装携带的 Python 3.12 和必要驱动，打开新的 PowerShell，进入复制后的项目：

```powershell
Set-Location "D:\AI\neuron-explorer-main"
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install --no-index --only-binary=:all: --find-links ".\offline-cache\wheelhouse" -r ".\offline-cache\requirements-offline.txt"
if ($LASTEXITCODE -ne 0) { throw "Offline dependency installation failed" }
.\.venv\Scripts\python.exe -m pip check
```

目标机不需要再安装 uv，也不需要运行 `uv sync`。按第 7 或第 8 节设置离线变量，直接用新建环境的 Python 启动即可。

## 10. 断网验收清单

关闭所有已运行的 Explorer 服务，断开外网，然后按选定路线重新启动。

| 检查 | 合格表现 |
|---|---|
| 本地配置 | 服务没有为了获取模型配置而报联网错误 |
| 服务状态 | 路线 A 为 `no model`。路线 B 在点击「加载」前为 `idle`，加载成功后为 `ready` |
| 新的浏览器窗口 | 无痕窗口打开本机地址后，路线 A 直接显示地图，路线 B 显示加载面板并在加载成功后显示地图 |
| 前端资源 | 开发者工具 Network 中，Three.js 从 `127.0.0.1:8791/vendor/three/` 加载 |
| 缓存 lens | 搜索词或 `/api/lens?layer=22&n=2742` 能返回结果 |
| 历史数据 | 已携带 run 的层能够显示历史结果；没有携带的层允许为空 |
| 本地生成（路线 B） | 短提示词能完成三列生成，服务没有尝试重新下载权重 |
| 实时激活（路线 B） | 一两句话能显示 token 和激活结果 |
| 配置保存（路线 B） | 保存、使用、停用成功；重启后记录仍在，但需重新激活 |

也可在开发者工具中启用 Disable cache 后刷新，确认所有必要前端文件来自本地。物理断网不会阻止本机 `127.0.0.1` 通信；不要把浏览器整个页面切到会同时拦截本机请求的 Offline 模式来判断服务是否正常。

## 11. 常见问题

### 页面能打开，但地图空白

先查 Three.js 本地化步骤。打开开发者工具 Network/Console，检查有没有仍在请求 `cdn.jsdelivr.net`，或 `vendor/three/` 返回 404。还要检查浏览器 WebGL 是否可用。

不要双击 `explorer/static/index.html` 用 `file://` 打开，必须通过运行中的 [本地 HTTP 服务](http://127.0.0.1:8791/)访问。

### `LocalEntryNotFoundError`、无法读取配置，或断网后仍尝试下载

确认 `EXPLORER_MODEL` 指向实际本地目录，且当前工作目录为项目根目录。检查 `models/Qwen3.5-4B/Qwen3.5-4B/config.json`；路线 B 还要有 tokenizer 和所有权重分片。

```powershell
Get-Location
$env:EXPLORER_MODEL
Test-Path ".\models\Qwen3.5-4B\Qwen3.5-4B\config.json"
```

离线变量应在启动 Python **之前**设置。设置后需要重新启动进程。缺文件时回到联网准备阶段补齐。

### 页面显示 `no model`，无法聊天

检查是否残留 `EXPLORER_NO_MODEL`，删除后重启。设置成 `"0"` 无效：

```powershell
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
```

### CUDA 不可用，或量化后端不能加载

确认检查的是项目 `.venv` 中的 Python；检查驱动和 CUDA 版 PyTorch，并确认导出 wheel 时包含所需量化后端。程序仅检查到包已安装，不代表其本地组件和 GPU 组合已经通过实测。

如果只有未量化模型可用，设 `EXPLORER_QUANTIZATION=none` 并重启；如果内存不够，应回到联网阶段准备适合的加载方案。

### BF16/FP16 错误、非有限值或生成异常

根据 GPU 支持切换计算精度，先用未干预的短输入验证，再启用神经元调整。FP16 可能发生数值溢出，过强干预也会造成输出退化。切换精度后，旧配置批次可能不兼容。

### 加载或输入长文本时内存不足

关闭其他占用 GPU 的程序，先缩短文本和聊天历史。必要时使用已安装并验证的量化预设。完整 1024-token 激活和评分数组合计约 2.25 GiB 主机内存，还不包含模型、注意力捕获和中间副本；权重量化不能消除这些开销。

### Examples、排名、网页上下文或 embeddings 没有内容

按第 6 节核对数据。`runs/` 必须同时有 `<run>.jsonl` 与同名 `<run>.meta.json`。网页上下文需要 FFW 的三个文件。基础地图缓存无法反推出历史生成文本。

### 界面提示“模型忙”，不能切换精度或配置

停止当前生成，等待模型加载/计算结束后重试。当前模型实例通过锁串行执行相关操作，多个窗口同时使用时会互相等待。

### `8791` 端口被占用

在启动 Python 前改端口：

```powershell
$env:PORT = "8792"
.\.venv\Scripts\python.exe explorer\server.py
```

之后访问 [本机 8792 端口](http://127.0.0.1:8792/)，状态检查地址也相应修改。当前服务绑定 `127.0.0.1` 并校验本机来源，教程不包含从其他电脑访问该服务的配置。

### 离线 wheel 安装提示没有匹配版本

先核对 Python 版本、64 位架构和 wheel 是否完整。确认没有把 macOS/Linux 的包带到 Windows，也没有只打包项目直接依赖却漏掉间接依赖。回到联网准备机完成第 9.2 节的全新环境演练。

### 想保存数据、换电脑或升级项目

除模型与分析数据外，保留以下个人文件：

- `explorer/names.json`：自定义神经元名称。
- `explorer/cache/neuron_batches.json`：分组与已保存的干预配置。
- `explorer/cache/model_loading.json`：界面成功应用后保存的加载设置。
- `models/Qwen3.5-4B/Qwen3.5-4B/offline-source.json`：按本教程下载时记录的模型来源。

项目的 `.gitignore` 会排除许多实验数据和个人设置，因此 Git 提交或重新下载源码并不等于完成备份。更新项目后重新检查本地 Three.js 路径、依赖版本、模型兼容性和断网验收结果。
