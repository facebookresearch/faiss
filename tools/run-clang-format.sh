#!/usr/bin/env bash
set -euo pipefail

mode="${1:-format}"
image_name="${FAISS_FORMAT_IMAGE:-faiss-format}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
file_list="$(mktemp)"

cleanup() {
    rm -f "$file_list"
}

trap cleanup EXIT

case "$mode" in
    format|check)
        ;;
    *)
        echo "Usage: tools/run-clang-format.sh [format|check]" >&2
        exit 2
        ;;
esac

if ! git -C "$repo_root" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Expected to run from inside a git checkout" >&2
    exit 2
fi

git -C "$repo_root" ls-files -z | grep -zE '\.(cpp|h|cu|cuh)$' >"$file_list"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not on PATH" >&2
    exit 2
fi

if ! docker info >/dev/null 2>&1; then
    echo "docker is installed but the daemon is not available" >&2
    exit 2
fi

docker build -t "$image_name" -f - "$repo_root" <<'EOF'
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        wget \
        lsb-release \
        software-properties-common \
        gnupg \
        git-core \
        ca-certificates \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod u+x llvm.sh \
    && ./llvm.sh 21 \
    && apt-get install -y --no-install-recommends clang-format-21 \
    && rm -f llvm.sh \
    && rm -rf /var/lib/apt/lists/*

RUN cat <<'INNER_EOF' >/usr/local/bin/run-clang-format-in-container
#!/usr/bin/env bash
set -euo pipefail

mode="${1:-format}"

if [[ ! -d /work ]]; then
    echo "Expected the repository to be mounted at /work" >&2
    exit 2
fi

cd /work

case "$mode" in
    format|check)
        ;;
    *)
        echo "Usage: run-clang-format-in-container [format|check]" >&2
        exit 2
        ;;
esac

xargs -0 clang-format-21 -i </tmp/faiss-format-files

if [[ "$mode" == "check" ]]; then
    if [[ ! -d /git ]]; then
        echo "Expected the repository git metadata to be mounted at /git" >&2
        exit 2
    fi

    if git --git-dir=/git --work-tree=/work diff --quiet; then
        echo "Formatting OK!"
    else
        echo "Formatting not OK!"
        echo "------------------"
        git --git-dir=/git --work-tree=/work --no-pager diff --color
        exit 1
    fi
fi
INNER_EOF

RUN chmod 0755 /usr/local/bin/run-clang-format-in-container

ENTRYPOINT ["/usr/local/bin/run-clang-format-in-container"]
CMD ["format"]
EOF

git_dir="$(git -C "$repo_root" rev-parse --git-dir)"
if [[ "$git_dir" != /* ]]; then
    git_dir="$repo_root/$git_dir"
fi

docker run --rm \
    -v "$repo_root":/work \
    -v "$git_dir":/git \
    -v "$file_list":/tmp/faiss-format-files \
    "$image_name" \
    "$mode"
