#!/bin/bash
# Convenience entry-point — delegates to gpu-cu/wsl/verify.sh --install
# Usage:  wsl -e bash gpu-cu/scripts/test_install.sh
exec "$(dirname "${BASH_SOURCE[0]}")/../wsl/verify.sh" --install "$@"
