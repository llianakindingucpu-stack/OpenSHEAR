"""Emergency C: drive cleanup - find largest deletable dirs in user profile."""
import os, shutil, glob

user = os.path.expanduser("~")

# dirs that are safe to wipe completely
SAFE_WIPE = [
    os.path.join(user, "AppData", "Local", "Temp"),
    os.path.join(user, "AppData", "Local", "CrashDumps"),
    os.path.join(user, "Downloads"),
]

# dirs that are large but need selective cleanup
SELECTIVE = [
    (os.path.join(user, "AppData", "Roaming", "cargo", "target"), ["debug", "tmp"]),
]

total_freed = 0

for path in SAFE_WIPE:
    if os.path.exists(path):
        try:
            files = list(glob.glob(os.path.join(path, "*")))
            sz = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
            count = 0
            for f in files:
                try:
                    if os.path.isfile(f) and not os.path.basename(f).startswith("_MEI"):
                        os.remove(f)
                        count += 1
                except:
                    pass
            print(f"[OK] Cleaned {path}: {count} files, {sz//1e6:.0f} MB")
            total_freed += sz
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

# Selective: cargo debug artifacts
cargo_target = os.path.join(user, "AppData", "Roaming", "cargo", "target")
if os.path.exists(cargo_target):
    for sub in ["debug", "tmp"]:
        subp = os.path.join(cargo_target, sub)
        if os.path.exists(subp):
            try:
                sz = sum(os.path.getsize(os.path.join(d,f)) 
                         for d,_,fs in os.walk(subp) for f in fs)
                shutil.rmtree(subp, ignore_errors=True)
                print(f"[OK] Removed cargo/{sub}: {sz//1e6:.0f} MB")
                total_freed += sz
            except Exception as e:
                print(f"[SKIP] cargo/{sub}: {e}")

print(f"\nTotal freed: {total_freed//1e6:.0f} MB")

# Disk check
try:
    for drive in ["C:\\", "D:\\"]:
        u = shutil.disk_usage(drive)
        print(f"{drive} {u.free//1e9:.1f} GB free")
except Exception as e:
    print(f"Disk check: {e}")
