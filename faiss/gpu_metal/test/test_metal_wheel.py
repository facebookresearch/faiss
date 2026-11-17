# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in installed-wheel test using a relocated, dedicated Metal build.

Build swigfaiss in a temporary build directory, then move that directory away
from its original path. Set FAISS_METAL_WHEEL_BUILD_DIR to the relocated build
and FAISS_METAL_WHEEL_ORIGINAL_BUILD_DIR to its now nonexistent original path.
The test never moves or edits the supplied build. Requires setuptools, wheel,
pip, numpy, packaging, and a Metal device. Run this file with unittest.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class TestMetalWheel(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("FAISS_METAL_WHEEL_BUILD_DIR"),
        "Requires a relocated dedicated Metal build",
    )
    def test_installed_wheel_without_build_tree(self):
        build = Path(os.environ["FAISS_METAL_WHEEL_BUILD_DIR"]).resolve()
        original = Path(
            os.environ["FAISS_METAL_WHEEL_ORIGINAL_BUILD_DIR"]
        ).absolute()
        self.assertFalse(
            original.exists(), "Original build tree must be unavailable"
        )
        python_build = build / "faiss/python"
        shader = build / "faiss/gpu_metal/MetalDistance.metallib"
        source = Path(__file__).resolve().parents[2] / "python"
        # Verify that the extension really refers to the absent path, rather
        # than accidentally testing another build with an available metallib.
        embedded_path = os.fsencode(
            str(original / "faiss/gpu_metal/MetalDistance.metallib")
        )
        self.assertTrue(
            embedded_path in (python_build / "_swigfaiss.so").read_bytes(),
            "Extension must refer to the unavailable original build tree",
        )
        with tempfile.TemporaryDirectory(
            prefix="faiss-metal-wheel-"
        ) as directory:
            stage = Path(directory) / "inputs"
            stage.mkdir()
            # Use the current source setup.py, not an older configured copy.
            shutil.copy2(source / "setup.py", stage / "setup.py")
            for path in python_build.glob("*.py"):
                if path.name != "setup.py":
                    shutil.copy2(path, stage / path.name)
            shutil.copy2(
                python_build / "_swigfaiss.so", stage / "_swigfaiss.so"
            )
            shutil.copytree(python_build / "contrib", stage / "contrib")
            shutil.copy2(shader, stage / shader.name)
            subprocess.run(
                [sys.executable, "setup.py", "bdist_wheel"],
                cwd=stage,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wheel, = (stage / "dist").glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                self.assertIn(
                    "faiss/MetalDistance.metallib", archive.namelist()
                )
                wheel_shader = archive.read("faiss/MetalDistance.metallib")
                self.assertTrue(
                    wheel_shader == shader.read_bytes(),
                    "The wheel must contain the newly built shader library",
                )
            target = Path(directory) / "installed"
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "--no-deps", "--no-index",
                    "--target", str(target), str(wheel),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            env = os.environ.copy()
            env.pop("FAISS_METALLIB_PATH", None)
            env["PYTHONPATH"] = str(target)
            # A fresh process avoids the global pipeline/library cache. Its
            # extension has an unavailable compiled-in build path, so successful
            # searches must load the packaged metallib through dladdr lookup.
            subprocess.run(
                [sys.executable, "-c", """
from pathlib import Path
import faiss
import numpy as np
import sys
assert Path(faiss.__file__).resolve().is_relative_to(
    Path(sys.argv[1]).resolve())
assert not Path(sys.argv[2]).exists()
assert faiss.get_num_gpus() > 0, 'A real Metal device is required'
r = faiss.StandardGpuResources()
x = np.array([[0, 1, 2], [255, 254, 253]], dtype=np.uint8)
cpu = faiss.IndexBinaryFlat(24)
cpu.add(x)
gpu = faiss.index_binary_cpu_to_gpu(r, 0, cpu)
d, labels = gpu.search(x, 2)
np.testing.assert_array_equal(d, [[0, 24], [0, 24]])
np.testing.assert_array_equal(labels, [[0, 1], [1, 0]])
flat = faiss.IndexFlatL2(2)
flat.add(np.array([[0, 0], [1, 1]], dtype=np.float32))
metal = faiss.index_cpu_to_gpu(r, 0, flat)
d, labels = metal.search(np.array([[0, 0]], dtype=np.float32), 2)
np.testing.assert_array_equal(d, [[0, 2]])
np.testing.assert_array_equal(labels, [[0, 1]])
""", str(target), str(original)], cwd=directory, env=env, check=True,
            )


if __name__ == "__main__":
    unittest.main()
