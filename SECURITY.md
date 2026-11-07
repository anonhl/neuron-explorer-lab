# Security and local data

**English** | [简体中文](SECURITY.zh-CN.md)

Explorer is intended for a trusted, single-user local machine. It binds to `127.0.0.1` and validates local origins. Do not expose it directly as a public or multi-user hosted service.

Optional Claude/Codex agents can access local files or execute commands through their tools; these permissions differ from model inference. Use them only when you understand and trust the workflow. Chats, uploads and isolated CLI account files may be stored under `explorer/cache/agent/`. Do not share that directory.

Model-loading preferences, neuron names, saved batches and browser drafts may also contain personal information. The release builder selects explicit source files and upstream reference caches, excluding user models and session data. `.gitignore` cannot remove files already committed to history.

## Reporting

If the [repository](https://github.com/anonhl/neuron-explorer-lab) enables GitHub **Report a vulnerability**, use it for private reports. No independent security email is configured. Do not guess an address or post credentials, private inputs or directly exploitable full details in a public issue.

Ordinary installation/use problems can be reported with a sanitized minimal reproduction if Issues is enabled. If credentials were exposed, revoke or rotate them before cleaning files and history; deleting the current file alone does not make the credential safe.
