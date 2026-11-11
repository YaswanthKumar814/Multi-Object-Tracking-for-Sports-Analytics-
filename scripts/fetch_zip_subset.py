"""Download only selected members of a large remote ZIP using HTTP range requests.

SportsMOT (6.5 GB of validation frames) and UVY-Track (3.3 GB) are published as
single ZIP archives. The experiments in this project use a handful of
sequences, so instead of downloading whole archives this tool reads the ZIP
central directory remotely and fetches just the members that match a pattern.

    # list what is inside (writes the listing to a text file)
    python scripts/fetch_zip_subset.py URL --list listing.txt

    # extract every member whose path matches a regular expression
    python scripts/fetch_zip_subset.py URL --match "val/v_abc_c001/" --out data/raw/x

The server must support ``Range`` requests (Google Cloud Storage and Zenodo do).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP range requests.

    Data is fetched in aligned blocks and a few blocks are cached, so the many
    small reads made by ``zipfile`` turn into a small number of requests.
    """

    def __init__(self, url: str, block_size: int = 8 * 1024 * 1024, max_blocks: int = 6):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._pos = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        self.requests_made = 0
        self.size = self._probe_size()

    def _probe_size(self) -> int:
        r = self._get(0, 0)
        total = r.headers.get("Content-Range", "").rpartition("/")[2]
        if r.status_code != 206 or not total.isdigit():
            raise RuntimeError(f"Server does not support range requests for {self.url} (HTTP {r.status_code})")
        return int(total)

    def _get(self, start: int, end: int) -> requests.Response:
        for attempt in range(5):
            r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            self.requests_made += 1
            if r.status_code in (200, 206):
                self.bytes_fetched += len(r.content)
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(int(r.headers.get("Retry-After", 5 * (attempt + 1))), 90)
                print(f"  HTTP {r.status_code}; waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {r.status_code} for {self.url}")
        raise RuntimeError(f"Giving up on {self.url} after repeated server errors")

    def _block(self, idx: int) -> bytes:
        if idx in self._blocks:
            self._blocks.move_to_end(idx)
            return self._blocks[idx]
        start = idx * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = self._get(start, end).content
        self._blocks[idx] = data
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return data

    # -- io.RawIOBase interface -------------------------------------------- #
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self._pos, self.block_size)
            chunk = self._block(idx)[off : off + n]
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def safe_target(out_dir: Path, member: str, strip: int) -> Path | None:
    """Destination for a ZIP member, or None if the name would escape ``out_dir``."""
    parts = [p for p in Path(member.replace("\\", "/")).parts if p not in ("", ".", "/")]
    parts = parts[strip:]
    if not parts or any(p == ".." or ":" in p for p in parts):
        return None
    target = out_dir.joinpath(*parts)
    if out_dir.resolve() not in target.resolve().parents:
        return None
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--list", metavar="FILE", help="Write 'size<TAB>name' for every member to FILE and exit")
    ap.add_argument("--match", action="append", default=[], help="Regex; members matching any are extracted")
    ap.add_argument("--out", help="Output directory for extracted members")
    ap.add_argument("--strip", type=int, default=0, help="Leading path components to drop")
    ap.add_argument("--max-files", type=int, default=None, help="Stop after N members per pattern (sorted by name)")
    ap.add_argument("--block-mb", type=int, default=8)
    args = ap.parse_args()

    remote = HttpRangeFile(args.url, block_size=args.block_mb * 1024 * 1024)
    print(f"Remote archive: {remote.size / 1e9:.2f} GB")
    zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1024 * 1024))
    infos = [i for i in zf.infolist() if not i.is_dir()]
    print(f"{len(infos)} files in archive (central directory read with {remote.requests_made} requests)")

    if args.list:
        Path(args.list).parent.mkdir(parents=True, exist_ok=True)
        with open(args.list, "w", encoding="utf-8") as fh:
            for i in infos:
                fh.write(f"{i.file_size}\t{i.filename}\n")
        print(f"Listing written to {args.list}")
        return 0

    if not args.match or not args.out:
        ap.error("--match and --out are required unless --list is given")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: list[zipfile.ZipInfo] = []
    for pattern in args.match:
        rx = re.compile(pattern)
        hits = sorted((i for i in infos if rx.search(i.filename)), key=lambda i: i.filename)
        if args.max_files:
            hits = hits[: args.max_files]
        print(f"pattern {pattern!r}: {len(hits)} files, {sum(i.file_size for i in hits) / 1e6:.1f} MB")
        selected.extend(hits)
    # Fetch in archive order so consecutive members share cached blocks.
    selected = sorted({i.filename: i for i in selected}.values(), key=lambda i: i.header_offset)

    done = skipped = 0
    for n, info in enumerate(selected, 1):
        target = safe_target(out_dir, info.filename, args.strip)
        if target is None:
            print(f"  skipping unsafe member name: {info.filename!r}", file=sys.stderr)
            continue
        if target.exists() and target.stat().st_size == info.file_size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            dst.write(src.read())
        done += 1
        if n % 100 == 0:
            print(f"  {n}/{len(selected)} files, {remote.bytes_fetched / 1e6:.0f} MB transferred")
    print(
        f"Extracted {done} files ({skipped} already present) to {out_dir}; "
        f"{remote.bytes_fetched / 1e6:.0f} MB transferred in {remote.requests_made} requests"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
