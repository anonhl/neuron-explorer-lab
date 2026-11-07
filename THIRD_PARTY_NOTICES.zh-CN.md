# 第三方资源说明

[English](THIRD_PARTY_NOTICES.md) | **简体中文**

项目的 MIT 许可证不替代第三方资源自己的条款。再分发时须保留原版权与许可声明。

## 原始 neuron-explorer

- 作者版权：`Copyright (c) 2026 lyramakesmusic`。
- 来源：[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)，MIT；完整文本保存在 [LICENSE](LICENSE)。本地修改署名为 anonhl。
- 本次核对的上游参考提交为 `7ec363d2d9686802a1b4aa7c6d8a2627b1759159`，这是比较基准，不代表原始下载时的确切提交。
- 原始探索器、扫描工具、研究笔记、截图和参考缓存来自上游。Windows/离线文档、加载控制、神经元配置、频率分析、残差分析与干预、LoRA 集成、测试和演示／发布工具包含本地修改。本项目不是上游官方发布。

## 随附 Three.js

- 使用未修改的 `three@0.169.0` 官方 npm 发布包，来源与完整性记录见 [vendor 说明](explorer/static/vendor/README.zh-CN.md)。
- 主许可为 MIT，版权为 `Copyright © 2010-2024 three.js authors`，见[完整声明](explorer/static/vendor/three/LICENSE)。
- 完整 npm 包中的示例、库、字体和二进制还可能有各自的许可，原目录内的声明均予保留；尤其包括[字体许可](explorer/static/vendor/three/examples/fonts/LICENSE)与 [Droid Apache 声明](explorer/static/vendor/three/examples/fonts/droid/NOTICE)。不能把整包所有资源统一改称 MIT。
- 单文件演示只内嵌实际使用的 Three.js 模块、控制器和线条 addons，并在 HTML 中保留原项目与 Three.js 的完整 MIT 声明，可以单独分享。

## 随附参考缓存

发布工具仅保留[来源清单](docs/upstream-cache-provenance.json)中列出的 9 个上游缓存，并核对 Git blob SHA-1 与 SHA-256。它们来自公开上游，不是发布者私人会话生成的结果。SHA-1 用于标识原始 blob，SHA-256 用于校验发布字节。

`r_index.npz` 与 `jpp_index.npz` 使用 [bubskee/qwen3.5-4b-r-jpp-lenses](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses) 生成；上游 README 与构建脚本将该资源标识为 Apache-2.0。衍生索引随附来源说明和 [Apache-2.0 全文](LICENSES/Apache-2.0.txt)，不附原始透镜矩阵。此处记录来源，不宣称本 Fork 已独立重新许可上游全部产物。

其余缓存包括参考模型的统计、布局、连接、token 字符串与透镜索引。切换量化、精度或适配器后，它们仍只是原始模型参考；发布清单记录确切字节与来源。

## 不随源码分发的模型、适配器与数据

| 资源 | 来源与处理方式 |
|---|---|
| Qwen3.5-4B | [官方模型](https://huggingface.co/Qwen/Qwen3.5-4B)，Apache-2.0。另行下载；再分发时保留模型 LICENSE/NOTICE 和模型卡。 |
| R/J++ 原始矩阵 | [原始发布](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses)，按上游说明为 Apache-2.0；另行下载并保留其元数据和许可。 |
| FineFineWeb | [m-a-p/FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb)，由用户选择下载。发布包不含语料、来源 URL 与激活捕获；再分发片段和衍生数据前需核对数据集及原网页条款，不统一声称 MIT。 |
| 用户 LoRA | 用户自行提供，遵循适配器、底模和训练数据的各自条款。 |
| vLLM steering fork | [RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)，可选 Linux 扫描流程另行下载；单独分发运行时时保留其许可与声明。 |

## 安装依赖与智能体工具

PyTorch、Transformers、Accelerate、PEFT、bitsandbytes、HQQ、NumPy、FastAPI、Uvicorn、datasets、sentence-transformers、umap-learn 等由 `pyproject.toml` 与 `uv.lock` 安装，代码不复制到源码包，保留各自发行许可。另行分发离线 wheel 包时，须保留 wheel 内的许可并记录确切版本。

Claude/Codex CLI、账户服务和凭据不随项目分发，由用户按提供方条款另行安装和认证。兼容性名称不表示对方认可或背书。
