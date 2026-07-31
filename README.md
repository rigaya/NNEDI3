NNEDI3
======

Update of nnedi3 to port it on x64.

## ビルド

### Windows

`nnedi3.sln` を Visual Studio / MSBuild でビルドしてください（meson は Windows 非対応）。

```powershell
msbuild nnedi3.sln /m /p:Configuration=Release /p:Platform=x64
```

出力: `x64\Release\nnedi3.dll`

### Linux

meson でビルドします。AviSynth+ の開発パッケージが必要です。

```bash
meson setup build
meson compile -C build
meson install -C build
```

## テスト

AVX-VNNI経路の検証方法は[tests/avxvnni/README.md](tests/avxvnni/README.md)を参照してください。
