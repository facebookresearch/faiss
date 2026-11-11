# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Download exactly the data the paper's results need, for the scales you choose.

    python paper_download.py --results 1M            # the 1M results only (smallest download)
    python paper_download.py --results 10M           # the 1M and 10M results
    python paper_download.py --results 100M          # the 1M, 10M and 100M results
    python paper_download.py --results 10M --dry-run # sizes only, nothing downloaded

The corpora are billions of rows, but each result reads only a prefix of a file and, for SPACEV
and SimSearchNet++, 20,000 held-out rows near its end. This script fetches those bytes and nothing
else, with parallel HTTP range requests, into the layout paper_data.py expects under --data
(default $EMEANS_DATA): the prefix as a file of the corpus's own format, and the held-out rows in a
small heldout_rows.npy beside it. Before downloading, it prints what each choice costs and asks for
confirmation (--yes skips the question). An interrupted download resumes where it stopped. At the
end every downloaded dataset is checked against the paper's fingerprints.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as data  # noqa: E402

HF = "https://huggingface.co/datasets"
FB = "https://dl.fbaipublicfiles.com"
SPACEV_PART = 4294967295               # bytes per part of the SPACEV1B release

# local file (under --data): where it comes from and how its rows are laid out.
#   layout "bin": 8-byte header then rows of d values; "bvecs": a 4-byte header per row;
#   "whole": downloaded entirely (size given)
SOURCES = {
    "deep1b/base.1B.fbin": dict(url=HF + "/suchun/deep1b-mirror/resolve/main/base.1B.fbin",
                                layout="bin", row=96 * 4),
    "deep1b/query.public.10K.fbin": dict(
        url=HF + "/suchun/deep1b-mirror/resolve/main/query.public.10K.fbin", layout="whole",
        size=3840008),
    "bigann/base.1B.u8bin": dict(url=FB + "/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
                                 layout="bin", row=128),
    "bigann/query.public.10K.u8bin": dict(
        url=FB + "/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin", layout="whole",
        size=1280008),
    "spacev1b/base.1.4B.i8bin": dict(
        parts=HF + "/jkhe/spacev1b/resolve/main/vectors.bin/vectors_%d.bin", layout="bin",
        row=100),
    "spacev100m/base.100M.i8bin": dict(
        url=HF + "/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin", layout="whole",
        size=10000000008),
    "spacev100m/query.30K.i8bin": dict(
        url=HF + "/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin", layout="whole",
        size=2931608),
    "fb_ssnpp/FB_ssnpp_database.u8bin": dict(
        url=FB + "/billion-scale-ann-benchmarks/FB_ssnpp_database.u8bin", layout="bin", row=256),
    "dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs": dict(
        url=FB + "/large_objects/dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs",
        layout="bvecs", row=4 + 1024),
    "dino_vitl_10B/queries_clean.bvecs": dict(
        url=FB + "/large_objects/dino_vitl_10B/queries_clean.bvecs", layout="whole",
        size=102800000),
    "vibe/imagenet-clip-512-normalized.hdf5": dict(
        url=HF + "/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5",
        layout="whole", size=2627494208),
}

# What each choice gives, and the datasets it needs (each level includes the ones before it).
LEVELS = {
    "1M": ("Table 3, 1M row; Figure 3, DEEP* 1M and ImageNet-CLIP 1M panels",
           ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M",
            "imagenet-clip-512-normalized"]),
    "10M": ("Table 3, 10M row; Figure 4(b), upper row; Figure 3, SPACEV 10M and SIFT 10M panels",
            ["deep1b-10M", "dino10b-10M-div4096", "spacev-10M", "fb-ssnpp-10M",
             "sift1b-10M-div512", "sift1b-10M-normalized"]),
    "100M": ("Table 3, 100M row (DEEP*, SPACEV, SimSearchNet++); Figure 4(b), lower row",
             ["deep1b-100M", "spacev-100M", "fb-ssnpp-100M", "sift1b-100M-div512"]),
}
ORDER = ["1M", "10M", "100M"]
CHUNK = 64 << 20                       # bytes per range request


def local_name(candidates):
    """The candidate file name this script provides (paper_data accepts several)."""
    return next(c for c in candidates if c in SOURCES)


def needs(name):
    """{local file: [(start, stop) byte ranges]} that dataset `name` reads."""
    spec = data.DATASETS[name]
    base = local_name(spec["base"])
    src = SOURCES[base]
    out = {}
    if src["layout"] == "whole":
        out[base] = [(0, src["size"])]
    else:
        head = 8 if src["layout"] == "bin" else 0
        rows = spec["n"] + (10_000 if spec.get("eval") == "next" else 0)
        out[base] = [(0, head + rows * src["row"])]
    if isinstance(spec.get("eval"), tuple):
        q = local_name(spec["eval"])
        out[q] = [(0, SOURCES[q]["size"])]
    return out


def heldout_needs(name):
    """(local corpus file, held-out row numbers) for a dataset whose test rows are held out of
    its corpus file, else None."""
    spec = data.DATASETS[name]
    if "held_out" not in spec or spec["base"] == data.IMAGENET:
        return None
    return local_name(spec["base"]), np.load(os.path.join(data.SPLITS, spec["held_out"]))


def heldout_bytes(f, rows):
    return len(rows) * SOURCES[f]["row"]


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(ranges, done):
    """The parts of `ranges` not yet covered by `done` (both merged)."""
    out = []
    for a, b in ranges:
        for c, d in done:
            if d <= a or c >= b:
                continue
            if c > a:
                out.append((a, c))
            a = max(a, d)
            if a >= b:
                break
        if a < b:
            out.append((a, b))
    return out


def plan(level):
    files = {}
    for lv in ORDER[:ORDER.index(level) + 1]:
        for name in LEVELS[lv][1]:
            for f, rs in needs(name).items():
                files.setdefault(f, []).extend(rs)
    held = {}
    for lv in ORDER[:ORDER.index(level) + 1]:
        for name in LEVELS[lv][1]:
            h = heldout_needs(name)
            if h:
                held[h[0]] = h[1]
    return {f: merge(rs) for f, rs in files.items()}, held


def plan_bytes(level):
    files, held = plan(level)
    return (sum(b - a for rs in files.values() for a, b in rs)
            + sum(heldout_bytes(f, r) for f, r in held.items()))


def requests_for(f, a, b):
    """(url, remote start, remote stop, local offset) pieces for local bytes [a, b)."""
    src = SOURCES[f]
    out = []
    while a < b:
        stop = min(b, a + CHUNK)
        if "parts" in src:                                   # SPACEV1B: concatenated parts
            i, off = divmod(a, SPACEV_PART)
            stop = min(stop, (i + 1) * SPACEV_PART)
            out.append((src["parts"] % (i + 1), off, off + stop - a, a))
        else:
            out.append((src["url"], a, stop, a))
        a = stop
    return out


def fetch(url, start, stop, tries=6):
    """Bytes [start, stop) of url, read with a buffer bounded by the request.

    The server must honour the range: HTTP 206 with exactly the requested
    Content-Range. HTTP 200 (the whole file) is accepted only when the request
    asked for the whole file, checked by start == 0 and a matching Content-Length.
    At most stop - start + 1 bytes are read, and the length must match exactly,
    so a server that ignores Range cannot pull a whole corpus file into memory.
    """
    want = stop - start
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, stop - 1)})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 206:
                    expected = "bytes %d-%d/" % (start, stop - 1)
                    answered = r.headers.get("Content-Range") or ""
                    if not answered.startswith(expected):
                        raise IOError("asked for %s, server sent %r" % (expected, answered))
                elif r.status == 200:
                    length = r.headers.get("Content-Length")
                    if start != 0 or length is None or int(length) != want:
                        raise IOError("server ignored the Range request (HTTP 200, %s bytes)"
                                      % length)
                else:
                    raise IOError("unexpected HTTP %d" % r.status)
                body = r.read(want + 1)
            if len(body) != want:
                raise IOError("read %d bytes, expected %d" % (len(body), want))
            return body
        except Exception as e:                                # noqa: BLE001
            if attempt == tries - 1:
                raise RuntimeError("%s [%d, %d): %s" % (url, start, stop, e))
            time.sleep(2 ** attempt)


def download(root, todo, threads):
    """Fetch the missing ranges with at most `threads` requests in flight.

    Each completed chunk is written and released before the next request is
    submitted, so memory stays near threads * CHUNK whatever the file size.
    """
    total = 0
    for f, rs in todo.items():                               # what is still missing
        state = os.path.join(root, f) + ".ranges.json"
        done = merge(json.load(open(state))) if os.path.exists(state) else []
        total += sum(b - a for a, b in subtract(rs, done))
    got = 0
    t0 = time.time()
    for f, rs in todo.items():
        path = os.path.join(root, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = path + ".ranges.json"
        done = merge(json.load(open(state))) if os.path.exists(state) else []
        missing = subtract(rs, done)
        if not missing:
            continue
        if not os.path.exists(path):
            open(path, "wb").close()
        pieces = iter([p for a, b in missing for p in requests_for(f, a, b)])
        with open(path, "r+b") as fh, concurrent.futures.ThreadPoolExecutor(threads) as pool:
            pending = {}

            def submit_next():
                for u, s, e, off in pieces:                  # at most one per call
                    pending[pool.submit(fetch, u, s, e)] = off
                    return

            for _ in range(threads):
                submit_next()
            while pending:
                finished, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in finished:
                    off = pending.pop(fut)
                    body = fut.result()
                    fh.seek(off)
                    fh.write(body)
                    done = merge(done + [(off, off + len(body))])
                    json.dump(done, open(state, "w"))
                    got += len(body)
                    del body, fut                            # release before the next request
                    submit_next()
                    rate = got / max(time.time() - t0, 1e-9)
                    print("\r%6.2f of %.2f GB (%.0f MB/s, %s)   " % (
                        got / 1e9, total / 1e9, rate / 1e6, f), end="", flush=True)
    print()


def download_heldout(root, held, threads):
    """The held-out rows, fetched block by block (consecutive rows share one request) and saved in
    the order of their row list as heldout_rows.npy beside the corpus file."""
    for f, rows in held.items():
        out = os.path.join(root, os.path.dirname(f), data.HELDOUT)
        if os.path.exists(out):
            continue
        src = SOURCES[f]
        dtype = {".u8bin": np.uint8, ".i8bin": np.int8, ".fbin": np.float32}[os.path.splitext(f)[1]]
        d = src["row"] // np.dtype(dtype).itemsize
        order = np.argsort(rows)
        srt = rows[order]
        cuts = np.flatnonzero(np.diff(srt) != 1) + 1
        blocks = np.split(srt, cuts)
        pieces = []
        for block in blocks:                                 # (local start, stop) per block
            a, b = 8 + int(block[0]) * src["row"], 8 + (int(block[-1]) + 1) * src["row"]
            pieces.append(requests_for(f, a, b))
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            bodies = list(pool.map(lambda reqs: b"".join(fetch(u, s, e) for u, s, e, _o in reqs),
                                   pieces))
        sorted_rows = np.frombuffer(b"".join(bodies), dtype=dtype).reshape(len(srt), d)
        result = np.empty_like(sorted_rows)
        result[order] = sorted_rows                          # back to the row list's order
        tmp = out[:-4] + ".tmp%d.npy" % os.getpid()
        np.save(tmp, result)
        os.replace(tmp, out)
        print("saved %d held-out rows (%d range requests) in %s" % (len(rows), len(blocks), out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", choices=ORDER, required=True,
                    help="1M: the 1M results; 10M: the 1M and 10M results; 100M: all three")
    ap.add_argument("--data", default=data.ROOT, help="where to put the files ($EMEANS_DATA)")
    ap.add_argument("--threads", type=int, default=16, help="parallel range requests")
    ap.add_argument("--dry-run", action="store_true", help="print the sizes and stop")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--no-check", action="store_true", help="skip the fingerprint check")
    a = ap.parse_args()

    print("What each choice downloads (each includes the ones above it). Larger scales read a\n"
          "longer prefix of the big corpora, so the download grows about fivefold per step:\n")
    for lv in ORDER:
        size = plan_bytes(lv)
        mark = "  <- your choice" if lv == a.results else ""
        print("  --results %-5s %7.1f GB   about %s at 25 MB/s (200 Mbit/s), %s at 250 MB/s%s" % (
            lv, size / 1e9, hours(size / 25e6), hours(size / 250e6), mark))
        print("                  adds: %s" % LEVELS[lv][0])
    todo, held = plan(a.results)
    print("\nFiles for --results %s, under %s:" % (a.results, a.data))
    for f, rs in todo.items():
        print("  %-50s %8.2f GB" % (f, sum(b - x for x, b in rs) / 1e9))
    for f, rows in held.items():
        print("  %-50s %8.4f GB (%d held-out rows)" % (
            os.path.join(os.path.dirname(f), data.HELDOUT), heldout_bytes(f, rows) / 1e9,
            len(rows)))
    if a.dry_run:
        return
    if not a.yes and input("\nDownload? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("nothing downloaded")
    download(a.data, todo, a.threads)
    download_heldout(a.data, held, a.threads)
    if a.no_check:
        return
    os.environ["EMEANS_DATA"] = a.data
    data.ROOT = a.data
    for lv in ORDER[:ORDER.index(a.results) + 1]:
        for name in LEVELS[lv][1]:
            data.load(name)                                  # raises on a fingerprint mismatch
            print("checked %s against the paper's fingerprints" % name)


def hours(seconds):
    if seconds < 60:
        return "under a minute"
    return "%.0f min" % (seconds / 60) if seconds < 5400 else "%.1f h" % (seconds / 3600)


if __name__ == "__main__":
    main()
