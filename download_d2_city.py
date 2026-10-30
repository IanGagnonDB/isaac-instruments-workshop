"""Download D2-City, extract raw videos, and build deterministic benchmark copies.

CLI: --root data/ --copies 9 --workers 16.
Databricks: use notebooks/00_setup_and_preflight.py to call this module.
The raw/ and benchmark/ folders must be empty or absent. Copies means the total
number of benchmark instances per source, with copy_index starting at 1.
Manifest source_file values are paths relative to raw/.

The downloader uses 16 threads and 32 MiB ranges by default. One measured Azure
Canada Central download took 3m 07s, or 3m 36s including ZIP validation and saving.
Those historical timings exclude the extraction and duplication added here.
"""
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MODE = 'concurrent'
URL = 'https://download.scidb.cn/download?fileId=601b96a0d0628e70b10937ac'
HEADERS = {'Accept-Encoding': 'identity', 'User-Agent': 'd2-city-workshop-benchmark/1.0'}


def run(volume_root, url=URL, chunk_bytes=32*1024*1024, workers=16, run_label=None):
    if not volume_root or not str(volume_root).strip():
        raise ValueError("volume_root is required and must not be blank")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    run_label = run_label or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    if Path(run_label).name != run_label or run_label in ('.', '..'):
        raise ValueError('run_label must be one path component')
    if workers < 1 or chunk_bytes < 1:
        raise ValueError('workers and chunk_bytes must be positive')
    destination = Path(volume_root) / 'd2-city-download-benchmark' / MODE / run_label
    destination.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS, method='HEAD'), timeout=60) as response:
        size = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    if size < 1:
        raise RuntimeError('Empty source file')
    metadata_seconds = time.monotonic()-started
    common_headers = dict(HEADERS)
    if etag and not etag.startswith('W/'):
        common_headers['If-Match'] = etag
    actual_workers = 1 if MODE == 'sequential' else workers
    print(json.dumps({'event': 'start', 'mode': MODE, 'bytes': size, 'workers': actual_workers,
                      'etag': etag, 'destination': str(destination)}), flush=True)
    transferred = 0
    lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix='d2-city-') as staging:
        if shutil.disk_usage(staging).free < size + 256*1024*1024:
            raise RuntimeError('Insufficient local disk space for the archive')
        local = Path(staging) / '1003_4.zip'
        with local.open('wb') as handle:
            handle.truncate(size)
        download_started = time.monotonic()

        def transfer(start, end, ranged):
            nonlocal transferred
            headers = dict(common_headers)
            if ranged:
                headers['Range'] = f'bytes={start}-{end}'
            expected = end-start+1
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                        if response.status != (206 if ranged else 200):
                            raise RuntimeError(f'Unexpected HTTP status {response.status}; range support is required')
                        if ranged and response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Incorrect Content-Range; refusing to assemble an invalid file')
                        if etag and response.headers.get('ETag') != etag:
                            raise RuntimeError('Source ETag changed during download')
                        count = 0
                        with local.open('r+b') as handle:
                            handle.seek(start)
                            while True:
                                block = response.read(1024*1024)
                                if not block:
                                    break
                                count += len(block)
                                if count > expected:
                                    raise RuntimeError('Response exceeded requested byte count')
                                handle.write(block)
                        if count != expected:
                            raise OSError(f'Truncated response: {count} of {expected} bytes')
                    with lock:
                        transferred += count
                        elapsed = time.monotonic()-download_started
                        print(json.dumps({'event': 'progress', 'bytes': transferred, 'total': size,
                                          'seconds': round(elapsed, 2),
                                          'MB_per_second': round(transferred/elapsed/1e6, 3)}), flush=True)
                    return
                except (urllib.error.URLError, OSError) as exc:
                    if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                        raise
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)

        if MODE == 'sequential':
            transfer(0, size-1, False)
        else:
            ranges = [(start, min(start+chunk_bytes, size)-1) for start in range(0, size, chunk_bytes)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = [executor.submit(transfer, start, end, True) for start, end in ranges]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        download_seconds = time.monotonic()-download_started
        validation_started = time.monotonic()
        if local.stat().st_size != size or transferred != size:
            raise RuntimeError('Archive size mismatch')
        digest = hashlib.sha256()
        with local.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        with zipfile.ZipFile(local) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f'ZIP CRC failed: {bad_member}')
            members = archive.infolist()
            video_files = sum(member.filename.lower().endswith('.mp4') for member in members)
            uncompressed_bytes = sum(member.file_size for member in members)
        validation_seconds = time.monotonic()-validation_started
        copy_started = time.monotonic()
        output = destination / '1003_4.zip'
        partial = destination / '1003_4.zip.partial'
        shutil.copyfile(local, partial)
        if partial.stat().st_size != size:
            raise RuntimeError('Persisted file size mismatch')
        partial.rename(output)
        copy_seconds = time.monotonic()-copy_started
    total_seconds = time.monotonic()-started
    metrics_path = destination / 'metrics.json'
    result = {'mode': MODE, 'source_url': url, 'started_at_utc': started_at, 'bytes': size,
              'workers': actual_workers, 'chunk_bytes': chunk_bytes if MODE == 'concurrent' else None,
              'etag': etag, 'sha256': digest.hexdigest(), 'zip_crc_verified': True,
              'video_files': video_files, 'uncompressed_bytes': uncompressed_bytes,
              'metadata_seconds': metadata_seconds, 'download_seconds': download_seconds,
              'validation_seconds': validation_seconds, 'volume_copy_seconds': copy_seconds,
              'total_seconds': total_seconds, 'download_MB_per_second': size/download_seconds/1e6,
              'download_under_10_minutes': download_seconds < 600,
              'durable_copy_under_10_minutes': total_seconds < 600,
              'output_path': str(output), 'metrics_path': str(metrics_path),
              'scope': 'One ZIP, no extraction; total excludes compute startup and metrics write.'}
    metrics_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def _check_output_folders(root):
    if not root or not str(root).strip():
        raise ValueError('root is required and must not be blank')
    root = Path(root)
    for folder in (root / 'raw', root / 'benchmark'):
        if folder.is_symlink() or (folder.exists() and
                                  (not folder.is_dir() or any(folder.iterdir()))):
            raise FileExistsError(f'Output folder must be empty or absent: {folder}')
    return root


def prepare_benchmark(zip_path, root='data/', copies=9, workers=16):
    """Extract a ZIP and create renamed copies; never overwrite existing outputs."""
    started = time.monotonic()
    if copies < 1 or workers < 1:
        raise ValueError('copies and workers must be positive integers')
    root = _check_output_folders(root)
    raw = root / 'raw'
    benchmark = root / 'benchmark'
    with zipfile.ZipFile(zip_path) as archive:
        seen = set()
        for member in archive.infolist():
            name = member.filename.replace('\\', '/')
            parts = Path(name).parts
            if (not parts or Path(name).is_absolute() or '..' in parts or
                    ':' in parts[0] or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
            normalized = Path(name).as_posix()
            if normalized in seen:
                raise ValueError(f'Duplicate ZIP member: {member.filename}')
            seen.add(normalized)
        raw.mkdir(parents=True, exist_ok=True)
        archive.extractall(raw)
    extraction_seconds = time.monotonic() - started
    sources = sorted((path for path in raw.rglob('*')
                      if path.is_file() and path.suffix.lower() == '.mp4'),
                     key=lambda path: path.relative_to(raw).as_posix())
    if not sources:
        raise ValueError('The ZIP contains no MP4 files')
    file_count = len(sources) * copies
    if file_count > 999999:
        raise ValueError('Requested copies exceed the six-digit event ID capacity')
    benchmark.mkdir(parents=True, exist_ok=True)
    rows = []
    for copy_index in range(1, copies + 1):
        for source in sources:
            event_id = f'evt_{len(rows) + 1:06d}'
            rows.append({'event_id': event_id, 'file_name': event_id + '.mp4',
                         'source_file': source.relative_to(raw).as_posix(),
                         'copy_index': copy_index})

    def copy_one(row):
        output = benchmark / row['file_name']
        shutil.copyfile(raw / row['source_file'], output)
        return output.stat().st_size

    copying_started = time.monotonic()
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_one, row) for row in rows]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            total_bytes += future.result()
            if completed % 100 == 0:
                print(f'Copied {completed}/{file_count} files', flush=True)
    copying_seconds = time.monotonic() - copying_started
    manifest = benchmark / 'manifest.csv'
    with manifest.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['event_id', 'file_name', 'source_file', 'copy_index'])
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.monotonic() - started
    result = {'source_count': len(sources), 'copies': copies, 'file_count': file_count,
              'total_bytes': total_bytes, 'total_size_gb': total_bytes / 1_000_000_000,
              'extraction_seconds': extraction_seconds, 'copying_seconds': copying_seconds,
              'elapsed_seconds': elapsed, 'raw_path': str(raw),
              'benchmark_path': str(benchmark), 'manifest_path': str(manifest)}
    print(f"Benchmark: {file_count} MP4 files, {result['total_size_gb']:.3f} GB. "
          f'Elapsed: {elapsed:.2f} seconds.', flush=True)
    return result


def run_workshop(root='data/', copies=9, workers=16, url=URL, run_label=None):
    """Download, extract and duplicate using the same local or volume base path."""
    started = time.monotonic()
    if copies < 1 or workers < 1:
        raise ValueError('copies and workers must be positive integers')
    root = _check_output_folders(root)
    download = run(volume_root=root, url=url, workers=workers, run_label=run_label)
    benchmark = prepare_benchmark(download['output_path'], root=root, copies=copies, workers=workers)
    elapsed = time.monotonic() - started
    print(f"Finished: {benchmark['file_count']} MP4 files, {benchmark['total_size_gb']:.3f} GB. "
          f'Elapsed including download: {elapsed:.2f} seconds.', flush=True)
    return {'download': download, 'benchmark': benchmark, 'total_elapsed_seconds': elapsed}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/', help='Base folder for raw/ and benchmark/')
    parser.add_argument('--copies', type=int, default=1, help='Instances per source MP4; use 1 for workshop setup')
    parser.add_argument('--url', default=URL)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.copies < 1 or args.workers < 1:
        parser.error('copies and workers must be positive integers')
    run_workshop(root=args.root, copies=args.copies, url=args.url, workers=args.workers)
