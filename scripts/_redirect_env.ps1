# DecentralAI 环境路径配置
$cargo = "D:\cargo-home"
$rustup = "D:\rustup"
$npm_prefix = "D:\npm-global"
$npm_cache_dir = "D:\npm-cache"
$tmp_dir = "D:\tmp"

foreach ($d in @($cargo, $rustup, $npm_prefix, $npm_cache_dir, $tmp_dir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$env:CARGO_HOME = $cargo
$env:RUSTUP_HOME = $rustup
$env:RUSTUP_DIST_SERVER = "https://rsproxy.cn"
$env:RUSTUP_UPDATE_ROOT = "https://rsproxy.cn/rustup"
$env:npm_config_prefix = $npm_prefix
$env:npm_config_cache = $npm_cache_dir
$env:TEMP = $tmp_dir
$env:TMP = $tmp_dir

[Environment]::SetEnvironmentVariable("CARGO_HOME", $cargo, "User")
[Environment]::SetEnvironmentVariable("RUSTUP_HOME", $rustup, "User")
[Environment]::SetEnvironmentVariable("RUSTUP_DIST_SERVER", "https://rsproxy.cn", "User")
[Environment]::SetEnvironmentVariable("TEMP", $tmp_dir, "User")
[Environment]::SetEnvironmentVariable("TMP", $tmp_dir, "User")
[Environment]::SetEnvironmentVariable("npm_config_prefix", $npm_prefix, "User")

Write-Host "Done. CARGO_HOME=$cargo"
cargo --version
