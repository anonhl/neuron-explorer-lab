# Contributing

[English](CONTRIBUTING.md) | **简体中文**
欢迎提交问题、文档改进与代码贡献。请先说明问题、预期行为和复现方式；功能贡献尽量保持范围明确。上游研究及基础功能归属原作者，本衍生版重点维护 Windows 本地使用与新增实验流程。

## Development

建议使用 Python 3.12。安装 uv 后在项目根目录执行：

```powershell
uv sync --locked --python 3.12 --group dev --extra quantization --extra lora
```

`dev` 提供 API 测试所需的 httpx。量化测试需要两个量化后端；不做量化开发时可以省略 `--extra quantization`，单独运行不依赖后端的测试。LoRA 功能依赖 `--extra lora`。

测试使用小张量、模拟对象和随机初始化的极小 Qwen 模型，不下载真实模型权重：

```powershell
uv run --locked --group dev --extra quantization --extra lora python -m unittest discover -s tests -v
node tests/test_residual_vector.mjs
node tests/test_residual_lab.mjs
node tests/test_residual_modes.mjs
```

前端检查建议使用 Node.js 22.7 或更新版本（支持无 package.json 时自动识别 ES 模块）。不需要 npm 安装或前端打包器。

仅检查单文件演示、发布工具与配置存储时，可以只用 Python 标准库：

```powershell
python -m unittest discover -s tests -p test_demo_bundle.py -v
python -m unittest discover -s tests -p test_release.py -v
python -m unittest discover -s tests -p test_neuron_batches.py -v
```

最后一条会在缺少运行依赖时跳过 runtime/API 测试。报告测试结果时写明跳过项，不把跳过当成通过。

界面修改后重新生成演示并检查：

```powershell
python explorer/build_demo.py
python tools/prepare_release.py --check
```

## Pull requests

- 描述触发条件、修改后的行为、验证方法与已知限制。
- 新增或修复实验行为时，用能验证实际性质的测试覆盖它，尤其注意残差守恒、token 位置、对照缓存隔离和 hook 清理。
- 修改依赖时运行 `uv lock` 并提交 `pyproject.toml` 与 `uv.lock`；不要用 `--frozen` 掩盖锁文件不同步。
- 保留第三方版权和许可；引入资源时更新 `THIRD_PARTY_NOTICES.zh-CN.md`。
- 不提交 `.env`、模型权重、聊天、私人输入、LoRA、运行缓存或含账户信息的日志。
- 不把性能预期写成已测结果。

贡献将按仓库 MIT 许可证发布；请只提交你有权公开的内容。

文档修改请同步中英文版本和语言切换链接；法律许可、第三方原包、上游研究记录与执行用提示词保留原文。
