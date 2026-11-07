# Neuron Explorer：Windows 启动教程

[English](START_WINDOWS.md) | **简体中文**
首次安装依赖和下载模型需要联网。准备好以后，可以离线启动，并在浏览器里打开 [http://127.0.0.1:8791/](http://127.0.0.1:8791/)。

## 1. 在项目文件夹打开 PowerShell

用文件资源管理器打开能看到 `README.md`、`pyproject.toml` 和 `explorer` 文件夹的目录。在资源管理器地址栏输入 `powershell`，按回车。后面的命令都在这个目录执行。

## 2. 安装依赖（首次运行）

如果还没安装 `uv`，运行下面的命令，然后关闭并重新打开 PowerShell。没有 `winget` 时，可以用 [uv 官方说明里的 Windows 安装方法](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)。

```powershell
winget install --id=astral-sh.uv -e
```

回到项目目录，运行：

```powershell
uv sync --locked --python 3.12
```

这会在项目内创建 `.venv` 并安装依赖。后续命令直接调用里面的 Python，无需激活环境。

## 3. 下载模型（首次运行，二选一）

项目使用 **Qwen3.5-4B**。完整底模目录是 **`models\Qwen3.5-4B\Qwen3.5-4B`**，LoRA 目录为 **`models\Qwen3.5-4B\my-lora`**。`models` 与 `README.md` 同级；外层 `Qwen3.5-4B` 用于归组，内层同名文件夹才存放底模文件。目录尚不存在时，下面的下载命令会自动创建底模目录。

```text
neuron-explorer-main\
├─ models\
│  └─ Qwen3.5-4B\
│     ├─ Qwen3.5-4B\               ← 完整底模（config.json、分词器和权重）
│     └─ my-lora\
│        ├─ adapter_config.json
│        └─ adapter_model.safetensors
├─ explorer\
└─ README.md
```

**要用本地聊天和神经元干预：**下载完整模型。如果已经下载过，确认该目录下直接有 `config.json`、分词器文件、权重索引和全部 `.safetensors` 分片，就可以跳过下载。

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B/Qwen3.5-4B')"
```

**只想先看神经元地图：**如果目录里还没有 `config.json`，只下载这一份文件即可。这个模式可以浏览已有分析数据，聊天和实时分析需要完整模型。

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B/Qwen3.5-4B', allow_patterns=['config.json'])"
```

## 4. 启动服务并打开网页

以后每次启动，只需要做这一步。在项目目录的 PowerShell 中，先运行下面三行：

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
```

然后在**同一个窗口**按你准备的模型文件选择一种启动方式。

**完整模型：**

```powershell
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

**只浏览地图：**

```powershell
$env:EXPLORER_NO_MODEL = "1"
.\.venv\Scripts\python.exe explorer\server.py
```

看到 `Uvicorn running` 后，用 Edge 或 Chrome 打开 [http://127.0.0.1:8791/](http://127.0.0.1:8791/)。完整模型模式会先打开「加载模型」面板。选择「全量加载」或「量化加载」，再设置 BF16 或 FP16 计算精度。量化加载时，点击 bitsandbytes 或 HQQ 展开分支，再选择量化精度。点击「加载」后才会读取权重并显示进度。加载成功后会进入探索器，加载失败时会显示具体原因，可以修改选择后重试。关闭面板可先浏览缓存地图。

只浏览地图模式会直接进入探索器，并显示“未加载模型”。

保持 PowerShell 窗口开启。要停止服务，在该窗口按 `Ctrl+C`。

## 可选：加载本地 LoRA

需要使用 LoRA 时，先在联网环境中向项目的 Python 环境安装 PEFT：

```powershell
uv sync --locked --python 3.12 --extra lora
```

将 `EXPLORER_MODEL` 指向训练 LoRA 时使用的完整底模。例如，基于官方预训练版训练的 LoRA 应搭配 `models/Qwen3.5-4B/Qwen3.5-4B-Base`。在「加载模型」面板选择「全量加载」，并在 LoRA 字段中填写本地适配器目录，例如 `models\Qwen3.5-4B\my-lora`。该目录需要包含 `adapter_config.json` 和 `adapter_model.safetensors`，且适配器应沿用底模的 tokenizer 和对话格式。

加载时会在内存中合并 LoRA。三列「评测」和「仅对话」都会使用合并后的权重。每次加载 LoRA 时，程序会用随项目附带的中英文短文本离线校准激活范围。校准完成后，可以在「激活」页输入文本并点击某个 token，查看该位置激活最强的神经元。校准仅在当前加载会话中使用。地图上的历史扫描仍是原模型的参考数据，依赖原模型预计算连接或透镜索引的功能在 LoRA 模式下受限。要恢复底模，重新打开加载面板，清空 LoRA 路径后点击「加载」。重启服务后，LoRA 路径默认留空。

## 常见问题

| 现象 | 怎么处理 |
| --- | --- |
| 找不到 `.venv\Scripts\python.exe` | 确认 PowerShell 打开在项目目录，并且 `uv sync --locked --python 3.12` 已成功结束。 |
| 提示找不到模型配置 | 运行 `Test-Path .\models\Qwen3.5-4B\Qwen3.5-4B\config.json`。如果返回 `False`，确认配置在内层 `Qwen3.5-4B` 中，或重新执行下载命令。 |
| 页面能打开，但聊天提示模型已禁用 | 确认完整权重已下载，按 `Ctrl+C` 停止服务，再用“完整模型”的命令启动。 |
| 模型加载时报显存不足 | 可以先用“只浏览地图”启动。如果要运行完整模型，参考详细教程里的 CUDA 和量化设置。 |

显卡、计算精度、量化和离线迁移见 [Windows 离线使用详细教程](OFFLINE_WINDOWS.zh-CN.md)。


## 全部层残差流扫描

在左侧「运行」输入词或文本，完成推理后打开「网络」与「运行」之间的「残差流」。每次运行都会采集模型的**全部层、每个输入 token 位置**的残差状态；选择某层只展开详情，不会缩小扫描范围。

- 在残差流中选择 token，与「运行」共享同一次采集结果；未选中位置时默认查看最后一个 token。已有运行结果切换层、状态或神经元时不会重复推理。
- 全层总览显示每层输出隐藏向量的范数。选择层后，可查看层输入、注意力后（MLP 归一化前）、层输出的完整维度值、正负最大分量，以及注意力和 MLP 各自的残差更新。
- 中间地图用一维向量带显示全部层的隐藏状态及层间残差流。全景色格表示分组维度均值；放大或点击「展开维度」后，每个格子对应一个真实维度。移动「维度窗口」可遍历整个隐藏向量，所有层共享同一维度区间。金色为正、粉色为负，亮度表示各带内的相对大小；选中神经元的写入显示在本层输出下方的细带，彩色入线落到对应维度。悬停可读具体数值，点击格子可联动左侧维度检查。神经元写入是实际激活乘以该神经元的下投影列，通常同时影响许多隐藏维度。
- 激活正负、向量维度正负和 token 得分变化正负分开显示。「局部 token Δlogit」是本层输出中移除选中神经元写入前后的透镜差值；「下一层读取者」固定下一层注意力结果后重新计算归一化与 MLP，属于局部估计，均未重新运行后续全部层。
- 点击「＋」保留候选，再点击「对候选神经元进行偏移评测」，进入既有评测面板调整偏移。可以导出当前 token 在全部层的原始残差向量（JSON）。

该功能需要实时模型。当前使用的神经元配置会反映在采集中；切换配置或重载权重后需要重新运行文本。除了已有激活数组，残差采集保存一份初始输入和每层的注意力后、层输出，沿用模型的计算精度；32 层、2560 维、1024 token、16 位精度时约增加 325 MiB 主机内存。初次建议使用短文本。

仓库根目录的 `neuron-explorer-demo.html` 同步提供操作练习，其中残差、激活及回复均为模拟数据，不读取模型权重。
