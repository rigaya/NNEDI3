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

CPU backend と `opt=8`
------------------------

`opt=8` は Windows x64 と Linux の AVX-512 backend を明示的に選択します。
実行には AVX2、FMA3、AVX512F、AVX512BW、AVX512DQ、AVX512VL のすべてが
必要です。明示指定時に不足する機能がある場合は、別 backend へ暗黙に
fallbackせず、不足しているCPU機能を示してエラーにします。Win32ビルドは
AVX-512対象外であり、`opt=8`を指定するとx64の使用を求めるエラーになります。

`opt=0`の自動選択は、AVX-512対応CPU上でも従来どおりAVX2 + FMA3を選びます。
7950X/WSL2で行った情報目的の測定では、処理内容によってAVX-512が速い場合と
遅い場合が混在しました。native Windows/Linux環境で自動選択に必要な性能を
測定できていないため、AVX-512は明示的な`opt=8`に限定しています。

積和をFMAへ合成する実装は、AVX2 + FMA3とAVX-512の経路で使用します。
C経路はFMAを持たないCPUでも実行できる必要があるため、乗算と加算を分離した
scalar実装を維持しています。Linuxのmain翻訳単位には`-ffp-contract=off`を指定し、
外部のCPU指定による暗黙のFMA合成も禁止します。`opt=5..7`は独立した
AVX2-only/FMA4 backendとはせず、サポートされるCPU経路ではAVX2 + FMA3の
`opt=6`として扱います。

AVX-512 の対応範囲
------------------

AVX-512化した主な処理は次のとおりです。8/16bit整数と32bit floatの各画素形式で、
WindowsとLinuxに共通のintrinsic実装を使用します。

- old/new prescreenerの入力変換、dot product、Elliott処理、判定mask生成
- predictorの入力抽出と統計、float/int16 dot product、e0/e1/e2活性化、
  weighted average
- processLine、castScale、YUY2とplanar 4:2:2、RGB24とplanar 4:4:4の相互変換

AVX-512専用関数へ置換しない制御処理、バッファ操作、重みの前処理などの補助処理は、
既存のAVX2 + FMA3 backendを基底として利用します。また、15/16bit入力でint16
predictorまたはold/new prescreenerのint16 dot-productを選ぶ経路はC実装へfallbackし、
C用の重み配置を使用します。CUDA kernelは今回のCPU AVX-512対応の対象外です。

初期実装ではAVX512 VNNI、VBMI、BF16、FP16を必須にしていません。現在の処理は
floatとint16を中心としており、これらを必須化すると対応CPUを狭める一方で、
追加のdispatch、重み配置、丸め順とoverflow semanticsの再検証が必要になります。
特にVNNIによるint16積和への置換は既存AVX2との加算結果を改めて検証する必要が
あるため、まずAVX512F/BW/DQ/VLとAVX2/FMA3で共通実装を成立させることを
優先しました。

AVX-512 のビルド境界
--------------------

Linuxでは次の5つの翻訳単位だけを専用static libraryとしてビルドし、
`-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma`を指定します。
通常のmain翻訳単位へAVX-512コンパイルオプションを付けないことで、非対応CPUが
CまたはAVX2経路を選んだ場合にAVX-512命令へ到達しない境界を維持しています。
既存のAVX2専用翻訳単位は`-mavx2 -mfma`だけを追加し、dispatchで
要求しないBMI、BMI2、POPCNTを暗黙の必須機能にしません。

- `nnedi3_intrinsic_AVX512.cpp`
- `nnedi3_intrinsic_AVX512_extract.cpp`
- `nnedi3_intrinsic_AVX512_pixel_convert.cpp`
- `nnedi3_intrinsic_AVX512_prescreener.cpp`
- `nnedi3_intrinsic_AVX512_process.cpp`

Windowsでは同じ5翻訳単位をx64だけで`/arch:AVX512 /utf-8`付きでコンパイルし、
whole program optimizationを無効にしてmain objectへの命令混入を防ぎます。
Win32からは専用翻訳単位を除外します。Visual Studio側には`/arch:AVX512`を
サポートするMSVCとWindows 10 SDKが必要です。Linux側には上記`-m`オプションを
サポートするC++17コンパイラ、Meson、`objcopy`、および親プロジェクトが提供する
AviSynth/CUDA/common/threads依存関係が必要です。
