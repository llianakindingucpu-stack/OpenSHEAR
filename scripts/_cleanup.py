"""Aggressive cleanup to free C drive space."""
import os, shutil, glob

# 1. Clean workspace node_modules
ws = r"C:\Users\Administrator\.qclaw\workspace-agent-0b9a94a1"
nm = os.path.join(ws, "node_modules")
if os.path.exists(nm):
    sz = sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(nm) for f in fs)
    shutil.rmtree(nm, ignore_errors=True)
    print(f"Removed node_modules: {sz/1e6:.0f} MB")

# 2. Clean npm cache
npm_cache = os.path.expandvars(r"%LOCALAPPDATA\npm-cache")
if os.path.exists(npm_cache):
    sz = sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(npm_cache) for f in fs)
    shutil.rmtree(npm_cache, ignore_errors=True)
    print(f"Removed npm-cache: {sz/1e6:.0f} MB")

# 3. Clean temp
tmp = os.environ.get("TEMP","")
if tmp and os.path.exists(tmp):
    files = [f for f in glob.glob(os.path.join(tmp,"*")) 
             if os.path.isfile(f) and not os.path.basename(f).startswith("_MEI")]
    sz = sum(os.path.getsize(f) for f in files)
    for f in files: 
        try: os.remove(f)
        except: pass
    print(f"Cleaned temp ({len(files)} files): {sz/1e6:.0f} MB")

# 4. Clean pip cache
pip_cache = os.path.expandvars(r"%LOCALAPPDATA\pip\cache")
if os.path.exists(pip_cache):
    sz = sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(pip_cache) for f in fs)
    shutil.rmtree(pip_cache, ignore_errors=True)
    print(f"Removed pip-cache: {sz/1e6:.0f} MB")

# 5. Clean Rust target debug (keep release)
rust = r"D:\IdeaProjects\decentral-ai\src-rs\decentral-ai-core\target\debug"
if os.path.exists(rust):
    sz = sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(rust) for f in fs)
    shutil.rmtree(rust, ignore_errors=True)
    print(f"Removed rust target/debug: {sz/1e6:.0f} MB")

# 6. Check disk space
import psutil
for drive in ["C","D"]:
    try:
        u = psutil.disk_usage(drive)
        print(f"{drive}: {u.free/1e9:.1f} GB free / {u.total/1e9:.1f} GB total")
    except: pass
