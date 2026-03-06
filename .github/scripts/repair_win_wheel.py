# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Windows wheel repair: bundle MKL runtime DLLs into the faiss package.

Copies MKL (and Intel OpenMP / TBB) DLLs from a staging directory into the
faiss/ package directory inside the wheel so they are co-located with
_swigfaiss.pyd and faiss.dll.  On Python 3.8+ Windows, DLL dependencies are
resolved from the directory containing the loading DLL, so co-location is
sufficient — no .pth file or os.add_dll_directory() call is needed.

Usage:
    python repair_win_wheel.py <wheel> <dest_dir> [--dll-dir C:/mkl/bin]
"""

import argparse
import base64
import glob
import hashlib
import os
import shutil
import tempfile
import zipfile


def repair(wheel_path, dest_dir, dll_dir):
    wheel_name = os.path.basename(wheel_path)
    tmpdir = tempfile.mkdtemp()

    try:
        # Unpack the wheel (it's a zip file).
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(tmpdir)

        # List DLLs/PYDs already in the wheel (diagnostic).
        print("Files in wheel:")
        for root, _dirs, files in os.walk(tmpdir):
            for f in sorted(files):
                if f.lower().endswith((".dll", ".pyd")):
                    rel = os.path.relpath(os.path.join(root, f), tmpdir)
                    size = os.path.getsize(os.path.join(root, f))
                    print(f"  {rel}  ({size:,} bytes)")

        # Locate the faiss package directory inside the wheel.
        faiss_dir = os.path.join(tmpdir, "faiss")
        if not os.path.isdir(faiss_dir):
            raise RuntimeError("faiss/ directory not found in wheel")

        # Find the RECORD file (in *.dist-info/).
        record_path = None
        for root, _dirs, files in os.walk(tmpdir):
            if root.endswith(".dist-info") and "RECORD" in files:
                record_path = os.path.join(root, "RECORD")
                break
        if not record_path:
            raise RuntimeError("RECORD file not found in wheel")

        # Copy runtime DLLs into faiss/ and update RECORD.
        new_records = []
        for dll in sorted(glob.glob(os.path.join(dll_dir, "*.dll"))):
            dll_name = os.path.basename(dll)
            dst = os.path.join(faiss_dir, dll_name)
            if os.path.exists(dst):
                print(f"  skip {dll_name} (already in wheel)")
                continue
            shutil.copy2(dll, dst)

            with open(dst, "rb") as f:
                digest = hashlib.sha256(f.read()).digest()
            b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            size = os.path.getsize(dst)
            new_records.append(f"faiss/{dll_name},sha256={b64},{size}")
            print(f"  bundled {dll_name} ({size:,} bytes)")

        with open(record_path, "a") as f:
            for rec in new_records:
                f.write(rec + "\n")

        # Repack the wheel.
        out_path = os.path.join(dest_dir, wheel_name)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(tmpdir):
                for fname in sorted(files):
                    fullpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fullpath, tmpdir)
                    zf.write(fullpath, arcname)

        print(f"  repaired wheel written to {out_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", help="Path to the input wheel")
    parser.add_argument("dest_dir", help="Directory to write the repaired wheel")
    parser.add_argument(
        "--dll-dir",
        default="C:/mkl/bin",
        help="Directory containing DLLs to bundle",
    )
    args = parser.parse_args()
    repair(args.wheel, args.dest_dir, args.dll_dir)
