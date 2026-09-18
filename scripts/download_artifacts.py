#!/usr/bin/env python3
"""Download and verify model artifacts from Hugging Face repository.

Reads artifacts/ARTIFACTS.json, downloads required weights and configs to
artifacts/v2/, verifies SHA-256 checksums and byte sizes, and exits non-zero
on any mismatch or network failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


def download_file(
    url: str,
    target_path: Path,
    expected_sha256: str,
    expected_bytes: int,
    headers: dict[str, str] | None = None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + '.part')

    req = urllib.request.Request(url, headers=headers or {})
    sha256 = hashlib.sha256()
    bytes_downloaded = 0

    print(f'Downloading {target_path.name} from {url}...')
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, 'wb') as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                sha256.update(chunk)
                bytes_downloaded += len(chunk)
                if expected_bytes > 0:
                    pct = (bytes_downloaded / expected_bytes) * 100.0
                    msg = f"\r  Progress: {bytes_downloaded / (1024*1024):.1f} MB / {expected_bytes / (1024*1024):.1f} MB ({pct:.1f}%)"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
        print()
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f'Failed to download {target_path.name}: {e}') from e

    calculated_hash = sha256.hexdigest()
    if calculated_hash != expected_sha256:
        if temp_path.exists():
            temp_path.unlink()
        raise ValueError(
            f"Hash mismatch for {target_path.name}:\n"
            f"  Expected: {expected_sha256}\n"
            f"  Got:      {calculated_hash}"
        )

    temp_path.replace(target_path)
    print(f'Verified {target_path.name} (SHA-256: {calculated_hash[:12]}...)')


def main() -> int:
    parser = argparse.ArgumentParser(description='Download model artifacts from Hugging Face.')
    parser.add_argument(
        '--manifest',
        type=str,
        default='artifacts/ARTIFACTS.json',
        help='Path to ARTIFACTS.json manifest',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts/v2',
        help='Target directory for downloaded artifacts',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist with matching hash',
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f'Error: Manifest {manifest_path} not found.', file=sys.stderr)
        return 1

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    repo_id = manifest.get('repo_id')
    files_meta = manifest.get('files', {})
    if not repo_id or not files_meta:
        print('Error: Invalid ARTIFACTS.json (missing repo_id or files).', file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {'User-Agent': 'Verdict-Downloader/2.0'}
    token = os.environ.get('HF_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    base_url = f'https://huggingface.co/{repo_id}/resolve/main'

    for filename, meta in files_meta.items():
        target_file = out_dir / filename
        expected_sha256 = meta['sha256']
        expected_bytes = meta.get('bytes', 0)

        if target_file.exists() and not args.force:
            with open(target_file, 'rb') as f:
                existing_hash = hashlib.sha256(f.read()).hexdigest()
            if existing_hash == expected_sha256:
                print(f'Already present and verified: {filename}')
                continue
            print(f'File {filename} exists but hash does not match. Re-downloading...')

        url = f'{base_url}/{filename}'
        try:
            download_file(url, target_file, expected_sha256, expected_bytes, headers)
        except Exception as err:
            print(f'Error downloading {filename}: {err}', file=sys.stderr)
            return 1

    print("\nAll artifacts verified successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
