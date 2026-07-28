#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"
binary="${build_dir}/nnedi3-avxvnni-test"

if [[ ! -x "${binary}" ]]; then
    echo "テスト実行ファイルがありません: ${binary}" >&2
    exit 2
fi

disassembly_file="$(mktemp)"
trap 'rm -f "${disassembly_file}"' EXIT
objdump -d -Mintel "${binary}" > "${disassembly_file}"

vex_count="$(grep -Ec '^[[:space:]]*[0-9a-f]+:[[:space:]]+c4 .*vpdpwssd.*ymm' "${disassembly_file}" || true)"
zmm_count="$(grep -Ec 'vpdpwssd.*zmm' "${disassembly_file}" || true)"

if [[ "${vex_count}" -eq 0 ]]; then
    echo "VEX形式のVPDPWSSD ymmを確認できませんでした" >&2
    exit 1
fi
if [[ "${zmm_count}" -ne 0 ]]; then
    echo "AVX-VNNIテストへVPDPWSSD zmmが混入しています" >&2
    exit 1
fi

echo "VEX形式のVPDPWSSD ymmを${vex_count}命令確認しました"
echo "VPDPWSSD zmmの混入はありません"
