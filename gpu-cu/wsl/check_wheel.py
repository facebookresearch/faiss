#!/usr/bin/env python3
"""Quick wheel content inspector."""
import zipfile, sys, pathlib

whl_dir = pathlib.Path("/mnt/f/GitHub/faiss/build_output")
wheels = list(whl_dir.glob("*.whl"))
if not wheels:
    print("No wheel found in", whl_dir)
    sys.exit(1)

for whl in wheels:
    print(f"\nWheel: {whl.name}  ({whl.stat().st_size/1024/1024:.1f} MB)")
    with zipfile.ZipFile(whl) as z:
        members = z.namelist()
        so_files = [n for n in members if n.endswith(".so")]
        print(f"  Total entries : {len(members)}")
        print(f"  .so files ({len(so_files)}):")
        for f in so_files:
            info = z.getinfo(f)
            print(f"    {info.compress_size/1024:6.0f} kB  {f}")
