# Neuron Explorer Lab

[English](README.md) | **简体中文**

面向 Windows 本地环境的 Qwen3.5-4B 神经元与残差流实验工作台。基于 [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) 独立维护，增加可复用干预、多文本对照、灵活的模型加载和本地 LoRA 分析。

[项目仓库](https://github.com/anonhl/neuron-explorer-lab) · [文档目录](docs/README.zh-CN.md) · [版本记录](CHANGELOG.zh-CN.md) · [参与贡献](CONTRIBUTING.zh-CN.md)

![神经元探索器](explorer.png)

## 本版本增加了什么

- **残差流实验**：在指定层、阶段和 token 位置添加方向、移除写入、去除投影或替换状态，重跑模型并比较行为。
- **多文本对照**：最多 8 条独立输入，比较生成、下一 token 概率、KL、候选续写得分和激活变化，导出实验 JSON。
- **联合或独立干预**：多选神经元后，可一起施加设置，也可让每个神经元分别测试全部输入。独立结果按神经元分组并可筛选；5 个神经元、5 条文本会得到 5 套结果，共 25 组对照。导出 JSON 会记录模式及每条结果所属的分组。保持到对话中的干预仍同时应用全部勾选设置。
- **可复用神经元配置**：按分组保存有序操作，记录模型和精度，支持启用、停用和复制编辑。
- **模型加载选择**：BF16/FP16、bitsandbytes/HQQ 权重量化，以及全量权重下的本地 LoRA。
- **本地使用与教学**：Windows/离线教程、本地 Three.js、频率分析和无需模型的单文件模拟演示。

三维地图、语言切换、基础神经元 steering、三栏对话、原有研究与扫描流水线来自上游。本项目不是上游官方版本。

## 残差流干预的灵感

一个灵感来源是**甘露特钠（GV-971，sodium oligomannate）**与肠—脑轴间接调控的研究：改变神经元所处的环境，也可能影响整体表现。对应到本项目，我们把残差流视作神经元共同读取和写入的“信息环境”，通过改变上游写入，观察下游神经元与模型行为的变化。这是设计类比，不表示生物机制与模型机制等同。[原始研究](https://www.nature.com/articles/s41422-019-0216-x)

## 从这里开始

| 目的 | 入口 |
|---|---|
| 不安装模型，先体验界面 | 双击 `neuron-explorer-demo.html`；全部数值与回复均为模拟 |
| Windows 安装与启动 | [中文启动教程](docs/START_WINDOWS.zh-CN.md) |
| 完全断网使用与迁移 | [离线详细教程](docs/OFFLINE_WINDOWS.zh-CN.md) |
| 使用配置与残差实验 | [实验指南](docs/EXPERIMENTS.zh-CN.md) |
| 开发或准备发布 | [贡献指南](CONTRIBUTING.zh-CN.md)、[发布指南](docs/RELEASING.zh-CN.md) |
| 了解实际验证情况 | [发布检查记录](docs/RELEASE_CHECKS.zh-CN.md) |

文档建议使用 **64 位 Python 3.12**（项目声明最低 3.11）、uv 和支持 WebGL/import map 的现代浏览器。量化需要 NVIDIA CUDA；CPU/MPS 使用未量化权重。继承的扫描脚本仍面向 Linux/CUDA。

在项目根目录的 PowerShell 中安装：

```powershell
uv sync --locked --python 3.12
# 可选依赖，按需选择：
uv sync --locked --python 3.12 --extra quantization
uv sync --locked --python 3.12 --extra lora
# 如果要保留两类依赖，在同一命令中同时使用两个 --extra。
```

模型权重不随源码分发。按[启动教程](docs/START_WINDOWS.zh-CN.md)下载到 `models/Qwen3.5-4B/Qwen3.5-4B` 后启动：

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

打开[本机页面](http://127.0.0.1:8791/)，选择加载格式并点击“加载”后才读取权重。只浏览缓存时，按教程准备模型配置并设置 `EXPLORER_NO_MODEL="1"`。

## 模型加载与分析

加载面板提供全量 BF16/FP16、bitsandbytes INT8/NF4/FP4（包括 4 位双重量化），以及 HQQ 8/4/3/2/1 位。仅部分线性层权重量化，激活仍为 BF16/FP16。INT8 临时解包权重做浮点乘法，不能据此保证速度更快或某个峰值显存。成功应用的精度与预设保存在本地。

使用 LoRA 时先安装 `--extra lora`，选择“全量加载”，填入含 `adapter_config.json` 与 `adapter_model.safetensors` 的本地目录，底模须匹配。程序在内存中合并并执行会话级激活校准，不改写原始权重。本项目不组合使用 LoRA 与量化加载。

探索器保留上游地图、排名、透镜/token 搜索、实时激活、干预聊天、连接、生成嵌入和可选智能体。历史生成、FFW 上下文与原始 R/J++ 透镜矩阵需要另外准备数据。随附的 9 个参考缓存已[与上游校验](docs/upstream-cache-provenance.json)。

## 限制与验证状态

- 量化、FP16 与适配器可能改变结果；历史扫描和索引对应参考模型，部分原有透镜与连接功能在 LoRA 下受限。
- 局部残差读出属于估计；因果实验分别使用全新生成缓存重跑对照组和干预组。
- 长文本激活采集占用较多主机内存，权重量化不能消除此开销，建议从短输入开始。
- 可选 Claude/Codex 智能体依赖另外安装并登录的 CLI，可能联网并访问本机文件或命令，与离线 Qwen 推理分开。服务供受信任的单用户本机使用。

## 上游研究

[上游 README](https://github.com/lyramakesmusic/neuron-explorer#readme)、[研究日志](research_log.md)和[来源笔记](NOTES.md)记录原始研究及 CUDA/vLLM 扫描流程。上游研究记录与执行用智能体提示词保留英文原文，维护中的中英文使用文档见[文档目录](docs/README.zh-CN.md)。

## 许可证

项目代码沿用 [MIT](LICENSE)，保留原作者版权。模型、数据集与第三方资源各自遵循其许可，见[第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)。源码发布包不含权重、私人适配器、个人配置、聊天和网页语料。
