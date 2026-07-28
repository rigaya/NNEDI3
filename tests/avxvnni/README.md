# AVX-VNNI経路のテスト手順

## 対象

次の実行経路を検証します。

| `opt` | 実行経路 |
|---:|---|
| 6 | AVX2+FMA3 |
| 8 | AVX2+FMA3+AVX-VNNI |
| 9 | AVX512（VNNI不使用） |
| 10 | AVX512-VNNI |

`opt=0`はAVX-VNNI対応CPUで`opt=8`を自動選択します。AVX512系は自動選択せず、明示指定します。

AVX-VNNIとAVX512-VNNIは別のCPU機能です。`avx512_vnni`だけを持つCPUでは`opt=8`を実行できません。CPUID上ではAVX-VNNIを`CPUID.(EAX=7,ECX=1):EAX[4]`で判定します。

## 自動テストの内容

`avxvnni_test.cpp`は、実行前にOSのAVX状態、AVX2、FMA3、AVX-VNNIを確認します。条件を満たさない場合はSIGILLを起こさず、終了コード77でスキップします。

対応CPUでは次を実行します。

- NNEDI3で使用する全入力長（32、48、64、96、128、192、288）とニューロン数で、AVX2とAVX-VNNI予測器のfloat出力をbit単位で比較
- 旧prescreenerを1万入力比較
- 新prescreenerを1万入力比較
- `len=128, n=64`の予測器を9ラウンド測定し、中央値を表示
- 生成物を逆アセンブルし、VEX形式の`VPDPWSSD ymm`が存在し、`VPDPWSSD zmm`が混入していないことを確認

テストデータの乱数seedは固定しているため、別環境でも同じ入力を使用します。

## 必要なもの

- AVX2、FMA3、AVX-VNNI対応CPU
- C++17対応コンパイラ
- MesonとNinja
- Linuxの逆アセンブル確認ではGNU `objdump`
- WindowsではVisual Studio 2022 Developer PowerShellと`dumpbin`

Linuxでは事前確認として次を実行できます。

```bash
lscpu | grep -oE 'avx_vnni|avx512_vnni' | sort -u
```

`avx_vnni`が必要です。`avx512_vnni`だけが表示される環境はAVX-VNNIテスト対象外です。

## Linux

リポジトリルートから次を実行します。

```bash
tests/avxvnni/run_linux.sh 2>&1 | tee avxvnni-test.log
```

ビルドディレクトリを変更する場合は第1引数で指定します。

```bash
tests/avxvnni/run_linux.sh /tmp/nnedi3-avxvnni-build
```

個別に実行する場合は次のとおりです。

```bash
meson setup tests/avxvnni/build tests/avxvnni --buildtype=release
meson compile -C tests/avxvnni/build
meson test -C tests/avxvnni/build --print-errorlogs
tests/avxvnni/verify_disassembly.sh tests/avxvnni/build
meson test -C tests/avxvnni/build --benchmark --verbose
```

ベンチマークの反復回数は直接指定できます。

```bash
tests/avxvnni/build/nnedi3-avxvnni-test --benchmark --iterations 100000
```

可能ならCPUを固定し、他の重い処理を止めて測定します。

```bash
taskset -c 4 tests/avxvnni/build/nnedi3-avxvnni-test --benchmark --iterations 100000
```

## Windows

Visual Studio 2022 Developer PowerShellを開き、リポジトリルートから実行します。

```powershell
tests\avxvnni\run_windows.ps1
```

ビルドディレクトリを変更する場合は引数で指定します。

```powershell
tests\avxvnni\run_windows.ps1 C:\Temp\nnedi3-avxvnni-build
```

PowerShellの実行ポリシーで止まる場合は、そのプロセスだけ許可します。

```powershell
powershell -ExecutionPolicy Bypass -File tests\avxvnni\run_windows.ps1
```

Windows版プラグイン自体は従来どおりMSBuildでビルドします。

```powershell
msbuild nnedi3.sln /m /p:Configuration=Release /p:Platform=x64
```

古いMSVCで`_mm256_dpwssd_avx_epi32`が未定義になる場合はVisual Studioを更新してください。AVX512VLの`_mm256_dpwssd_epi32`へ置き換えると別命令経路になるため、代用してはいけません。

## 実映像を使う統合テスト

自動テストは内積カーネルを直接比較します。プラグインのCPU判定、重み配置、関数選択まで含めるため、対応環境では実映像でも確認します。

推奨入力例:

```text
Y:\キャプチャ\202606051500_MUSIC：S 欧州鉄道の旅・スペイン2 _BSフジ・182.ts
```

Linux/WSLでは環境に応じて`/mnt/y/キャプチャ/...`へ読み替えます。入力プラグインとフレーム出力方法はテスト環境で普段使用しているものを使い、次のNNEDI3条件を固定します。

```text
field=0, nsize=6, nns=1, qual=1, etype=0,
threads=1, fapprox=15
```

同じ入力フレームを次の組み合わせで処理します。

| `pscrn` | 主な確認対象 | 比較する経路 |
|---:|---|---|
| 0 | 予測器 | `opt=6`と`opt=8`、`opt=9`と`opt=10` |
| 1 | 旧prescreener | `opt=6`と`opt=8`、`opt=9`と`opt=10` |
| 2 | 新prescreener | `opt=6`と`opt=8`、`opt=9`と`opt=10` |

各組み合わせのY4Mまたはraw出力は完全一致が期待値です。

```bash
sha256sum opt6.y4m opt8.y4m
cmp opt6.y4m opt8.y4m
sha256sum opt9.y4m opt10.y4m
cmp opt9.y4m opt10.y4m
```

速度測定では最低3回ウォームアップしてから9回以上測り、中央値を記録します。AVX-VNNIの効果を見たい場合は、まず`pscrn=0`で予測器中心の差を確認し、その後`pscrn=1`と`pscrn=2`で実運用条件を確認します。比較中は入力、フレーム範囲、スレッド数、CPU affinityを変えないでください。

## 結果として残す情報

別環境のsessionから結果を受け取る際は、最低限次を保存します。

```text
コミット:
CPU名:
OS:
コンパイラとバージョン:
AVX-VNNIの検出結果:
出力一致テスト:
VEX VPDPWSSD ymmの確認数:
AVX2+FMA3の中央値:
AVX2+FMA3+AVX-VNNIの中央値:
AVX-VNNI / AVX2比率:
実映像のpscrn=0/1/2のhash:
```

Linuxでは次もログへ含めると環境差を追いやすくなります。

```bash
git rev-parse HEAD
uname -a
lscpu
c++ --version
meson --version
```
