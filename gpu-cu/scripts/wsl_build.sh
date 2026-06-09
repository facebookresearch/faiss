#!/bin/bash
# Convenience entry-point — delegates to gpu-cu/wsl/build.sh
# Usage:  wsl -e bash gpu-cu/scripts/wsl_build.sh
exec "$(dirname "${BASH_SOURCE[0]}")/../wsl/build.sh" "$@"
