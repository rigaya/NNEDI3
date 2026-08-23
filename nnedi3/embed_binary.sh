#!/bin/sh
set -eu

objcopy=$1
input=$2
output=$3
work_dir=$4
cpu_family=$5
base_dir=$(pwd)

case "$input" in
    /*) input_path=$input ;;
    *) input_path=$base_dir/$input ;;
esac
case "$output" in
    /*) output_path=$output ;;
    *) output_path=$base_dir/$output ;;
esac
case "$work_dir" in
    /*) work_path=$work_dir ;;
    *) work_path=$base_dir/$work_dir ;;
esac

mkdir -p "$work_path"
cp "$input_path" "$work_path/nnedi3_binary1.bin"

cd "$work_path"
case "$cpu_family" in
    x86|x86_64)
        output_format=elf64-x86-64
        output_arch=i386:x86-64
        ;;
    aarch64)
        output_format=elf64-littleaarch64
        output_arch=aarch64
        ;;
    *)
        echo "未対応のCPUアーキテクチャです: $cpu_family" >&2
        exit 1
        ;;
esac
"$objcopy" \
    -I binary \
    -O "$output_format" \
    -B "$output_arch" \
    --set-section-alignment .data=64 \
    --rename-section .data=.rodata,alloc,load,readonly,data,contents \
    --redefine-sym _binary_nnedi3_binary1_bin_start=nnedi3_binary1_start \
    --redefine-sym _binary_nnedi3_binary1_bin_end=nnedi3_binary1_end \
    --redefine-sym _binary_nnedi3_binary1_bin_size=nnedi3_binary1_size \
    nnedi3_binary1.bin "$output_path"
