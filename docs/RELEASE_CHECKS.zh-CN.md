# 发布检查记录

[English](RELEASE_CHECKS.md) | **简体中文**

检查日期：2026-11-07。目标仓库：[anonhl/neuron-explorer-lab](https://github.com/anonhl/neuron-explorer-lab)。本记录描述源码准备阶段的本地验证，不代表已经发布正式版本。

## 已完成

| 检查 | 结果 |
|---|---|
| 完整 Python 测试 | **80 项通过，0 跳过**；覆盖批次存储、真实小张量 hook、量化配置／小矩阵、加载状态、接口、极小随机 Qwen 模型的残差守恒和因果干预、演示与发布工具 |
| JavaScript 检查 | 两个脚本通过：残差向量完整维度覆盖／传输校验，以及中文流式事件的字节拆分、错误与提前结束处理 |
| Python 语法 | explorer、sweep、tests、tools 编译检查通过 |
| 依赖清单与锁文件 | 新增 lora 可选依赖与 dev/httpx；uv lock --check 通过，未改动原有依赖版本 |
| 单文件演示 | 已从当前界面重新生成；包含原作者、anonhl 修改与 Three.js 的完整 MIT 声明，标明模拟数据 |
| 上游缓存 | 9 个文件、195.74 MiB，Git blob SHA-1 与 SHA-256 均与记录一致 |
| 发布内容 | 通过明确文件筛选、参考缓存完整性和许可检查；模型、聊天、个人 SQLite、配置批次与用户数据不入包 |
| 敏感信息 | 对发布候选文本进行令牌、私钥、凭据赋值、凭据 URL 和个人绝对路径模式扫描，未发现匹配；这不证明不存在任意秘密 |
| 文档 | 使用／贡献／发布文档拆分中英文并提供语言切换；本地文档链接逐项检查 |

本地测试环境：macOS arm64、Python 3.12.7、Node.js 26.4.0、uv 0.13.7；torch 2.14.1、transformers 5.20.1、bitsandbytes 0.50.2、HQQ 0.2.8.post1、PEFT 0.21.2。测试强制 Hugging Face 离线，未下载真实 Qwen 权重。

## 尚未验证或不适用

- **真实 Qwen 权重的生成、LoRA 与大规模实验效果**：本次未下载权重；单元测试使用小模型／模拟对象，不能代替效果评测。
- **Git 历史**：本地目录没有 .git，无法做本地全 refs 历史审计。已读取公开 Fork，准备前 main 与上游参考提交同为 7ec363d2d9686802a1b4aa7c6d8a2627b1759159；新增提交须以该历史为父级，禁止覆盖上游历史。模式扫描只覆盖本次准备的公开内容。
- **GitHub Actions**：本地通过不代表远端已通过；实际结果以 main 分支的工作流运行状态为准。

源码包由 tools/prepare_release.py 生成，内含逐文件 SHA-256 清单并附 ZIP 校验文件。
