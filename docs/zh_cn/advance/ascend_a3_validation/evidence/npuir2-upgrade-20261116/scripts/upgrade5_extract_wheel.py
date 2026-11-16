"""Extract a compiler wheel for inspection without pip/install side effects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('wheel', type=Path)
p.add_argument('--label', required=True)
a = p.parse_args()
dest = ROOT / 'artifacts' / a.label
dest.mkdir(exist_ok=False)
with zipfile.ZipFile(a.wheel) as archive:
    for entry in archive.infolist():
        out = dest / entry.filename
        assert out.resolve().is_relative_to(dest.resolve())
        archive.extract(entry, dest)
        mode = entry.external_attr >> 16
        if mode: out.chmod(mode & 0o777)
bins = list(dest.rglob('bishengir-compile'))
versions = []
for b in bins:
    v = subprocess.run([str(b),'--version'],capture_output=True,text=True)
    versions.append(dict(path=str(b),sha256=hashlib.sha256(b.read_bytes()).hexdigest(),
                         returncode=v.returncode,version=v.stdout+v.stderr))
record = dict(wheel=str(a.wheel),sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
              destination=str(dest), compilers=versions)
(ROOT/'inventory'/f'{a.label}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
