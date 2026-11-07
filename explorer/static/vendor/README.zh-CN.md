# 本地 Three.js 资源

[English](README.md) | **简体中文**

`three/` 保存完整的官方 npm 发布包，供 Explorer 在离线环境加载核心模块和 addons。请连同项目一起复制此目录。

- 包名及版本：`three@0.169.0`，与项目原先的 CDN 版本一致。
- 来源：[npm 发布包](https://registry.npmjs.org/three/-/three-0.169.0.tgz)。
- 下载日期：2026-11-03。
- 完整性：下载后使用 npm 版本元数据中的 SHA-512 校验值验证。
- 许可证：MIT，原始许可证保存在 `three/LICENSE`。
- 上游文件未作修改；入口配置位于 `../index.html` 的 import map。

```text
sha512-Ed906MA3dR4TS5riErd4QBsRGPcx+HBDX2O5yYE5GqJeFQTPU+M56Va/f/Oph9X7uZo3W3o4l2ZhBZ6f6qUv0w==
```

更新版本时，应同步替换整套 npm 包，保留许可证，并检查核心模块与 addons 的导入关系。不要混用不同版本。
