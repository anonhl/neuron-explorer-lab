# 开源发布指南

[English](RELEASING.md) | **简体中文**
本项目以 MIT 发布衍生代码，保留原作者版权。首次发布版本使用 `0.1.0`。

## 仓库名称与描述

建议仓库名：`neuron-explorer-lab`。展示名称：**Neuron Explorer Lab**。

GitHub description：

> Windows-focused neuron and residual-stream experiments for Qwen3.5-4B, with multi-text comparisons, reusable interventions, quantized loading, and local LoRA support. Based on neuron-explorer.

## 当前交付方式

目标仓库已确认为 [anonhl/neuron-explorer-lab](https://github.com/anonhl/neuron-explorer-lab)，是保留上游历史的公开 Fork。当前本地目录没有 `.git`，属于源码副本，无法在本目录审计或提交历史；不要把整个本地目录直接拖入网页上传。

可克隆[自己的 Fork](https://github.com/anonhl/neuron-explorer-lab)，把经过检查的源码 ZIP 解压内容覆盖进去，保留克隆目录的 `.git`。查看差异后提交，并在自己的仓库发布。这样保留上游历史、署名与后续同步能力。`RELEASE_MANIFEST.json` 是源码包内容清单，可作为 Release 附件，不必提交到源码树。

如果独立建新仓库，也必须保留 `LICENSE` 和来源说明，并如实说明初始提交来自源码快照，不能宣称保留了历史。

## 本地准备与检查

以下命令在项目根目录执行，兼容 Windows PowerShell：

```powershell
python explorer/build_demo.py
python -m unittest discover -s tests -p test_demo_bundle.py -v
python -m unittest discover -s tests -p test_release.py -v
node tests/test_residual_vector.mjs
node tests/test_residual_lab.mjs
python tools/prepare_release.py --check
python tools/prepare_release.py
```

最后一步生成 `dist/neuron-explorer-lab-0.1.0-source.zip` 和同名 `.sha256` 文件。源码包有且仅有明确允许的源码、文档、原始第三方资源和通过来源校验的参考缓存，包内含逐文件 SHA-256 清单。工具不会删除或修改个人运行数据。`dist/` 已忽略，不应提交 ZIP 到源码仓库。

完整 Python 测试与锁文件检查：

```powershell
uv lock --check
uv sync --locked --python 3.12 --group dev --extra quantization --extra lora
uv run --locked --group dev --extra quantization --extra lora python -m unittest discover -s tests -v
```

测试结果和缺失检查记入 [发布检查记录](RELEASE_CHECKS.zh-CN.md)。CI 配置检查源码包、演示和现有测试。

## 发布内容与许可

- 保留 `LICENSE` 中的 `Copyright (c) 2026 lyramakesmusic` 和完整 MIT 文本。本地修改另署 `Copyright (c) 2026 anonhl (modifications)`，不替换上游署名。
- 保留 `THIRD_PARTY_NOTICES.md`、`LICENSES/` 和 `explorer/static/vendor/` 原始许可文件。
- `explorer/cache/` 默认随源码保留 9 个经过校验的上游参考缓存，合计约 196 MiB；来源和哈希见 `upstream-cache-provenance.json`。不采用 Git LFS，避免用户下载源码 ZIP 时拿到指针而不是数据。
- 模型、LoRA、原始 lens 矩阵、网页语料、离线 wheel 包、个人 SQLite、聊天和导出实验不随源码发布。如另行提供离线大包，需要重新确认所有资源许可和隐私。
- 单文件演示可以作为独立 Release 附件。它内嵌所用代码的 MIT 声明，数据均为模拟，不是模型效果证明。

## 首次公开前

- [ ] 在目标 Fork 的真实 Git checkout 中重新运行发布检查，检查所有本地 refs 的历史补丁；模式扫描不能证明不存在任意秘密。
- [ ] 检查实际提交差异，确认没有 `.env`、权重、私人输入、账户配置或会话文件。`.gitignore` 不影响已被 Git 跟踪的文件。
- [ ] 如历史中曾有真实凭据，先撤销／轮换，再清理历史，不能只删除当前文件。
- [ ] 在 GitHub 设置简短描述与 topics，例如 `mechanistic-interpretability`、`qwen`、`steering`、`windows`；README 中保持来源和差异说明。
- [ ] 如需要私密漏洞报告，在 GitHub 开启 Private vulnerability reporting；未开启时，不声称此功能可用。
- [ ] 查看 GitHub Actions 实际结果；配置存在不代表 CI 已运行通过。
- [ ] 发布 `v0.1.0`；附源码 ZIP、校验文件、模拟 HTML 和下方版本说明。

## 首版发布说明

Neuron Explorer Lab v0.1.0 是基于 lyramakesmusic/neuron-explorer 的独立衍生版本，沿用 MIT 并保留上游署名。

新增 Windows/离线文档、BF16/FP16 与权重量化加载、全量权重下的本地 LoRA、可保存的神经元配置、全层残差分析和最多 8 条文本的残差干预对照实验。另附无需模型的单文件模拟演示。

模型权重不随源码分发。缓存参考原始 Qwen3.5-4B；量化、FP16 和 LoRA 可能改变结果。局部读取估计与真正重跑模型的因果实验分别标识。安装及验证状态以仓库的发布检查记录为准。

源码 ZIP 是经过筛选的发布内容；模拟 HTML 中的所有生成和分析数值均为示例数据。
