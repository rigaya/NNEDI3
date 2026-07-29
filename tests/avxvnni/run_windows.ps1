$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = if ($args.Count -ge 1) { $args[0] } else { Join-Path $ScriptDir "build" }

if (-not (Test-Path (Join-Path $BuildDir "meson-private/coredata.dat"))) {
    meson setup $BuildDir $ScriptDir --buildtype=release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

meson compile -C $BuildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
meson test -C $BuildDir --print-errorlogs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Executable = Join-Path $BuildDir "nnedi3-avxvnni-test.exe"
$Disassembly = dumpbin /disasm $Executable
$YmmInstructions = @($Disassembly | Select-String -Pattern "vpdpwssd.*ymm")
$ZmmInstructions = @($Disassembly | Select-String -Pattern "vpdpwssd.*zmm")
if ($YmmInstructions.Count -eq 0) {
    throw "VPDPWSSD ymmを確認できませんでした"
}
if ($ZmmInstructions.Count -eq 0) {
    throw "VPDPWSSD zmmを確認できませんでした"
}
Write-Host "VPDPWSSD ymmを$($YmmInstructions.Count)命令確認しました"
Write-Host "VPDPWSSD zmmを$($ZmmInstructions.Count)命令確認しました"

meson test -C $BuildDir --benchmark --verbose
exit $LASTEXITCODE
