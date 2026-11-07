"""Check and build an allowlisted source release, using only Python's standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    '.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md',
    'CONTRIBUTING.md', 'SECURITY.md', 'NOTES.md', 'research_log.md', 'explorer.png',
    'pyproject.toml', 'uv.lock', 'neuron-explorer-demo.html',
}
ROOT_FILES |= {name[:-3] + '.zh-CN.md' for name in tuple(ROOT_FILES)
               if name in {'README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md', 'THIRD_PARTY_NOTICES.md'}}
SOURCE_DIRS = {'explorer', 'sweep', 'tests', 'tools', 'docs', '.github', 'LICENSES'}
SOURCE_SUFFIXES = {'.py', '.js', '.mjs', '.html', '.css', '.md', '.txt', '.toml', '.yml', '.yaml', '.sh'}
PRIVATE_PARTS = {'.git', '.venv', '__pycache__', 'models', 'runs', 'stats', 'captures',
                 'logs', 'dist', 'offline-cache', '.offline-check', '.aws', '.codex', '.claude', '.agents'}
UPSTREAM_SCRIPTS = {'scripts/mirror.py', 'scripts/owl_delta_vs_base.py'}
MAX_FILE_SIZE = 100 * 1024 * 1024
SECRET_PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
    'API secret': re.compile(r'\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{24,}\b'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'personal absolute path': re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\'),
    'credential URL': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@[^\s/]+'),
    'assigned credential': re.compile(
        r'''(?i)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret)["']?\s*[:=]\s*["']([A-Za-z0-9_+/=.-]{20,})["']'''),
}


def cache_metadata(root=ROOT):
    return json.loads((root / 'docs/upstream-cache-provenance.json').read_text(encoding='utf-8'))


def collect_files(root=ROOT):
    """Unknown root files and all runtime caches are excluded, regardless of .gitignore."""
    caches = {f['path'] for f in cache_metadata(root)['files']}
    files = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        parts = rel.parts
        name = rel.as_posix()
        if any(part in PRIVATE_PARTS for part in parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlinks are not allowed in release inputs: {name}')
        if not path.is_file():
            continue
        if len(parts) == 1:
            include = name in ROOT_FILES
        elif name in UPSTREAM_SCRIPTS:
            include = True
        elif parts[0] in SOURCE_DIRS:
            if name.startswith('explorer/cache/'):
                include = name in caches
            elif name.startswith('explorer/static/vendor/'):
                include = True  # Preserve the unmodified package and all of its notices.
            else:
                include = path.suffix.lower() in SOURCE_SUFFIXES or name == 'docs/upstream-cache-provenance.json'
        else:
            include = False
        if (path.name.startswith('.env') or path.name.endswith('.local.md')
                or path.name in {'auth.json', 'credentials.json', '.DS_Store', 'Thumbs.db', 'Desktop.ini'}
                or any(part.startswith('.env') for part in parts)):
            include = False
        if include:
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def secret_findings(text):
    """Return only category/line, never the secret itself."""
    return [(category, text.count('\n', 0, match.start()) + 1)
            for category, pattern in SECRET_PATTERNS.items() for match in pattern.finditer(text)]


def validate(root=ROOT):
    files = collect_files(root)
    selected = {p.relative_to(root).as_posix() for p in files}
    required = ROOT_FILES - {'.gitignore'}
    required |= {'explorer/static/vendor/three/LICENSE', 'LICENSES/Apache-2.0.txt',
                 'docs/upstream-cache-provenance.json', 'docs/RELEASING.zh-CN.md'}
    errors = [f'Missing release file: {name}' for name in sorted(required - selected)]
    for item in cache_metadata(root)['files']:
        path = root / item['path']
        if item['path'] not in selected:
            errors.append(f'Missing reference cache: {item["path"]}')
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if (blob != item['git_blob_sha1'] or hashlib.sha256(data).hexdigest() != item['sha256']
                or len(data) != item['size']):
            errors.append(f'Reference cache differs from recorded upstream bytes: {item["path"]}')
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.stat().st_size >= MAX_FILE_SIZE:
            errors.append(f'File reaches GitHub regular-file size limit: {rel}')
        if path.suffix.lower() in SOURCE_SUFFIXES | {'.json', '.lock'} or path.name == 'LICENSE':
            try:
                source = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            for category, line in secret_findings(source):
                errors.append(f'{category}: {rel}:{line}')
    demo = root / 'neuron-explorer-demo.html'
    if demo.is_file():
        html = demo.read_text(encoding='utf-8')
        for notice in ('LICENSE', 'explorer/static/vendor/three/LICENSE'):
            if (root / notice).is_file() and (root / notice).read_text(encoding='utf-8') not in html:
                errors.append(f'Portable demo is missing the full notice from {notice}')
    # Do not imply a history audit occurred for a downloaded source snapshot.
    history = 'not available (source snapshot has no .git)'
    if (root / '.git').exists():
        try:
            result = subprocess.run(['git', 'log', '--all', '-p', '--no-ext-diff', '--no-textconv'],
                                    cwd=root, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=120, check=True)
            for category, line in secret_findings(result.stdout):
                errors.append(f'{category}: Git history patch line {line} (value withheld)')
            history = 'scanned all local refs (pattern-based; not a proof of absence)'
        except (OSError, subprocess.SubprocessError):
            errors.append('Git history could not be checked; fix Git before publishing this checkout.')
    if errors:
        raise ValueError('\n'.join(errors))
    return files, history


def build_release(root=ROOT, output=None):
    files, history = validate(root)
    output = output or root / 'dist/neuron-explorer-lab-0.1.0-source.zip'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'neuron-explorer-lab-0.1.0/'
    manifest = {
        'version': '0.1.0', 'kind': 'source release; no model weights or personal runtime data',
        'upstream_reference': cache_metadata(root)['reference_commit'],
        'history_check': history,
        'files': [],
    }
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            manifest['files'].append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            info = zipfile.ZipInfo(prefix + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.suffix == '.sh' else 0o100644) << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo(prefix + 'RELEASE_MANIFEST.json', date_time=(2026, 1, 1, 0, 0, 0))
        archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    return output, len(files), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check release inputs without creating an archive')
    parser.add_argument('--output', type=Path, help='Output ZIP path (default: dist/)')
    args = parser.parse_args()
    try:
        if args.check:
            files, history = validate()
            print(f'PASS: {len(files)} allowlisted files; upstream cache hashes and notices verified.')
        else:
            path, count, history = build_release(output=args.output)
            print(f'Built {path.name}: {count} files, {path.stat().st_size:,} bytes; SHA-256 sidecar written.')
        print(f'Git history: {history}')
        print('Pattern checks do not replace review.')
    except (ValueError, OSError) as exc:
        print(f'Release check failed:\n{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
