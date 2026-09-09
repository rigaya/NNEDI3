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

CPU backend と `opt`
---------------------

`opt` の指定値は次のとおりです。

- `0`: 自動選択
- `1`: C
- `2`: SSE2
- `3`: SSE4.1
- `4`: AVX
- `5`: AVX2
- `6`, `7`: AVX2の旧設定互換alias
- `8`: AVX-VNNI
- `9`: AVX-512
- `10`: AVX-512 VNNI

AVX2 backendは旧FMA3実装を使用するため、CPU機能としてAVX2とFMA3の両方を
必要とします。`opt=5..7`はすべてAVX2 backendへ正規化します。旧AVX2-only実装と
FMA4実装は使用しません。

`opt=8..10`の明示指定時に必要なCPU機能が不足している場合は、別backendへ暗黙に
fallbackせず、不足しているCPU機能を示してエラーにします。Windowsでは`opt=5..7`
についても同様です。Linuxでは従来どおり、`opt=5..7`を利用できないCPUではC backendへ
fallbackします。Win32ビルドは`opt=8..10`の対象外です。`opt=0`では、CPUが対応する
最上位のbackendを自動的に選択します。

積和をFMAへ合成する実装は、AVX2とAVX-512の経路で使用します。
C経路はFMAを持たないCPUでも実行できる必要があるため、乗算と加算を分離した
scalar実装を維持しています。Linuxのmain翻訳単位には`-ffp-contract=off`を指定し、
外部のCPU指定による暗黙のFMA合成も禁止します。

AVX-512 の対応範囲
------------------

AVX-512化した主な処理は次のとおりです。8/16bit整数と32bit floatの各画素形式で、
WindowsとLinuxに共通のintrinsic実装を使用します。

- old/new prescreenerの入力変換、dot product、Elliott処理、判定mask生成
- predictorの入力抽出と統計、float/int16 dot product、e0/e1/e2活性化、
  weighted average
- processLine、castScale、YUY2とplanar 4:2:2、RGB24とplanar 4:4:4の相互変換

AVX-512専用関数へ置換しない制御処理、バッファ操作、重みの前処理などの補助処理は、
既存のAVX2 backendを基底として利用します。また、15/16bit入力でint16
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
