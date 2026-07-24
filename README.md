NNEDI3
======

Update of nnedi3 to port it on x64.

Linux の Meson ビルド
---------------------

このリポジトリの Meson 定義は単体プロジェクトではなく、親プロジェクト
`AviSynthCUDAFilters` のサブディレクトリとして使用します。ビルドには親側が
定義する `avisynth_dep`、`cuda_dep`、`common_dep`、`threads_dep` と、`common`、
`KUtil` のソースが必要です。

このリポジトリを `AviSynthCUDAFilters/NNEDI3` に配置し、Meson の setup と compile
は `AviSynthCUDAFilters` のルートから実行してください。このリポジトリ直下の
`meson.build` は親プロジェクトから読み込むための入口であり、単体での
`meson setup` には対応していません。
