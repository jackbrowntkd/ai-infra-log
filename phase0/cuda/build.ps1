<#
.SYNOPSIS
  编译一个 .cu 文件。

.DESCRIPTION
  为什么不直接敲 nvcc：Windows 上 nvcc 只是个 driver，真正的 host 编译靠 MSVC 的
  cl.exe，而 cl.exe 不在默认 PATH 里，必须先执行 vcvars64.bat。

  另一个坑：本项目路径含中文。cmd.exe 默认用 GBK 代码页，绝对路径传进去会被
  搞成乱码（表现为 LNK1104 打不开文件）。所以这里统一用相对路径，并且先
  chcp 65001。

.EXAMPLE
  .\phase0\cuda\build.ps1 phase0\cuda\hello.cu
  .\phase0\cuda\build.ps1 phase0\cuda\vecadd.cu -LineInfo
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Source,
    [string]$Out,
    [string]$Arch = "sm_75",
    [switch]$LineInfo,
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"

$vcvars = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $vcvars) {
    throw "找不到 vcvars64.bat。需要先安装 VS Build Tools 的 VCTools workload。"
}

$srcAbs = (Resolve-Path $Source).Path
$root = (Get-Location).Path.TrimEnd('\')
if (-not $srcAbs.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
    throw "请从仓库根目录调用本脚本，否则相对路径会带上中文目录名。"
}
$rel = $srcAbs.Substring($root.Length).TrimStart('\')
if (-not $Out) { $Out = [IO.Path]::ChangeExtension($rel, ".exe") }

$flags = @("-arch=$Arch", "-Xcompiler", "/utf-8")
if ($LineInfo) { $flags += "-lineinfo" }
$flags += $Extra

$cmdline = "chcp 65001 >nul && call `"$vcvars`" >nul 2>&1 && nvcc $($flags -join ' ') -o `"$Out`" `"$rel`""
Write-Host "nvcc $($flags -join ' ') -o $Out $rel" -ForegroundColor DarkGray

cmd /c $cmdline
if ($LASTEXITCODE -ne 0) { throw "编译失败 (exit $LASTEXITCODE)" }
Write-Host "OK -> $Out" -ForegroundColor Green