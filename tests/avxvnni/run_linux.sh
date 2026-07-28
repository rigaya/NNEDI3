#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"

if [[ ! -f "${build_dir}/meson-private/coredata.dat" ]]; then
    meson setup "${build_dir}" "${script_dir}" --buildtype=release
fi

meson compile -C "${build_dir}"
meson test -C "${build_dir}" --print-errorlogs
"${script_dir}/verify_disassembly.sh" "${build_dir}"
meson test -C "${build_dir}" --benchmark --verbose
