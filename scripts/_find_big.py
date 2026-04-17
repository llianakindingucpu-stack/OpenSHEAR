"""Find and optionally remove largest directories on C drive."""
import os, shutil

user = os.path.expanduser("~")
TARGETS = {
    os.path.join(user, "AppData", "Roaming", "cargo", "registry"): "cargo registry",
    os.path.join(user, "AppData", "Roaming", "cargo", "target"): "cargo target",
    os.path.join(user, "AppData", "Roaming", "npm", "node_modules"): "npm global",
    os.path.join(user, "AppData", "Roaming", "pip", "Data"): "pip data",
    os.path.join(user, "AppData", "Local", "pip", "cache"): "pip cache",
    os.path.join(user, "AppData", "Local", "Temp"): "temp (cleanable files)",
}

THRESHOLD = 100 * 1024 * 1024  # 100MB

def get_size(path):
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except:
                    pass
    except:
        pass
    return total

print("Scanning large directories...")
for path, name in TARGETS.items():
    if os.path.exists(path):
        sz = get_size(path)
        if sz > THRESHOLD:
            print(f"  {sz//1e9:.1f}GB  {name}  [{path}]")

print("\nCleaning temp files...")
tmp = os.environ.get("TEMP","")
if tmp and os.path.exists(tmp):
    removed = 0
    count = 0
    for f in os.listdir(tmp):
        fp = os.path.join(tmp, f)
        try:
            if os.path.isfile(fp) and not f.startswith("_MEI"):
                sz = os.path.getsize(fp)
                os.remove(fp)
                removed += sz
                count += 1
        except:
            pass
    print(f"  Removed {count} files, {removed//1e6:.0f} MB")

print(f"\nC free: {shutil.disk_usage('C:').free//1e9:.1f}GB")
