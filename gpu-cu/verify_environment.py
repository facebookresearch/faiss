#!/usr/bin/env python3
"""
Verify FAISS wheel build environment for the targeted CUDA version.
"""

import subprocess
import sys
import os
from pathlib import Path

# CUDA version single source of truth (mirror of gpu-cu/scripts/cuda_env.sh).
CUDA_VER = os.environ.get("FAISS_CUDA_VER", "13.2")

def run_command(cmd):
    """Run a command and return stdout or None if failed."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return None

def check_cuda():
    """Check CUDA installation."""
    print("\n[CUDA Check]")
    
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    nvcc_version = run_command('nvcc --version')
    
    if nvcc_version:
        print(f"  ✓ CUDA found")
        print(f"    Version: {nvcc_version.split('release')[-1].split(',')[0].strip()}")
        print(f"    Path: {cuda_home}")
        return True
    else:
        print(f"  ✗ CUDA not found")
        print(f"    Set CUDA_HOME or install CUDA {CUDA_VER}")
        return False

def check_python():
    """Check Python installation."""
    print("\n[Python Check]")
    
    version = run_command(f'{sys.executable} --version')
    if version:
        print(f"  ✓ Python found")
        print(f"    {version}")
        
        # Check Python version >= 3.10
        req_version = (3, 10)
        if sys.version_info >= req_version:
            print(f"    ✓ Version {req_version[0]}.{req_version[1]}+")
        else:
            print(f"    ✗ Version >= {req_version[0]}.{req_version[1]} required")
            return False
        return True
    else:
        print(f"  ✗ Python not found")
        return False

def check_build_tools():
    """Check required build tools."""
    print("\n[Build Tools Check]")
    
    tools = {
        'cmake': 'cmake --version | head -1',
        'make': 'make --version | head -1',
        'swig': 'swig -version | head -1',
        'gcc': 'gcc --version | head -1',
        'nvcc': 'nvcc --version | head -1',
    }
    
    all_found = True
    for tool, cmd in tools.items():
        output = run_command(cmd)
        if output:
            print(f"  ✓ {tool:8} - {output.split(chr(10))[0][:50]}")
        else:
            print(f"  ✗ {tool:8} - NOT FOUND")
            all_found = False
    
    return all_found

def check_python_packages():
    """Check required Python packages."""
    print("\n[Python Packages Check]")
    
    packages = {
        'numpy': 'import numpy; print(numpy.__version__)',
        'setuptools': 'import setuptools; print(setuptools.__version__)',
        'wheel': 'import wheel; print(wheel.__version__)',
    }
    
    all_found = True
    for pkg, cmd in packages.items():
        try:
            output = subprocess.run(
                [sys.executable, '-c', cmd],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            print(f"  ✓ {pkg:15} - {output}")
        except Exception:
            print(f"  ✗ {pkg:15} - NOT FOUND")
            all_found = False
    
    return all_found

def check_disk_space():
    """Check available disk space."""
    print("\n[Disk Space Check]")
    
    try:
        stat = os.statvfs('.')
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"  Available: {free_gb:.1f} GB")
        if free_gb >= 8:
            print(f"  ✓ Sufficient (8GB+ recommended)")
            return True
        else:
            print(f"  ⚠ Low - at least 8GB recommended for build")
            return False
    except Exception as e:
        print(f"  ? Could not determine: {e}")
        return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("FAISS GPU Wheel Build Environment Verification")
    print(f"CUDA {CUDA_VER} + Python 3.14")
    print("=" * 60)
    
    checks = [
        ("CUDA", check_cuda),
        ("Python", check_python),
        ("Build Tools", check_build_tools),
        ("Python Packages", check_python_packages),
        ("Disk Space", check_disk_space),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} - {name}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! Ready to build.")
        print("\nNext steps:")
        print("  1. make env-info          # Show build configuration")
        print("  2. make build             # Build wheel")
        print("  3. make install-wheel     # Install the wheel")
        return 0
    else:
        print("✗ Some checks failed. Please resolve issues before building.")
        print("\nCommon fixes:")
        print(f"  - Set CUDA_HOME=/usr/local/cuda-{CUDA_VER}")
        print("  - Install conda: conda env create -f gpu-cu/environment.yml")
        print("  - Install system deps: sudo apt install cmake make swig")
        return 1

if __name__ == '__main__':
    sys.exit(main())
