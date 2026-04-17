$cargo_home = "D:\cargo-home"
$npm_home = "D:\npm-global"
$npm_cache = "D:\npm-cache"
$tmp_dir = "D:\tmp"

New-Item -Path $cargo_home -ItemType Directory -Force | Out-Null
New-Item -Path $npm_home -ItemType Directory -Force | Out-Null
New-Item -Path $npm_cache -ItemType Directory -Force | Out-Null
New-Item -Path $tmp_dir -ItemType Directory -Force | Out-Null

$env:CARGO_HOME = $cargo_home
$env:RUSTUP_HOME = "D:\rustup"
$env:npm_config_prefix = $npm_home
$env:npm_config_cache = $npm_cache
$env:TEMP = $tmp_dir
$env:TMP = $tmp_dir

[Environment]::SetEnvironmentVariable("CARGO_HOME", $cargo_home, "User")
[Environment]::SetEnvironmentVariable("RUSTUP_HOME", "D:\rustup", "User")
[Environment]::SetEnvironmentVariable("TEMP", $tmp_dir, "User")
[Environment]::SetEnvironmentVariable("TMP", $tmp_dir, "User")
[Environment]::SetEnvironmentVariable("npm_config_prefix", $npm_home, "User")

Write-Host "D directories ready"
Write-Host "CARGO_HOME: $env:CARGO_HOME"
Write-Host "TEMP: $env:TEMP"
cargo --version
