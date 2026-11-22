"""Expose the stable-ABI MegaMoE wheel alongside LMDeploy's native DeepGEMM."""

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


def install(wheel: Path, prefix: Path):
    target = prefix / 'deep_gemm_mega'
    if target.exists():
        raise FileExistsError(target)
    prefix.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=prefix) as temporary:
        with ZipFile(wheel) as archive:
            archive.extractall(temporary)
        package = Path(temporary) / 'deep_gemm'
        if not (package / '_C_extension.abi3.so').is_file():
            raise ValueError('The DeepGEMM provider requires a built stable-ABI wheel.')
        shutil.move(package, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('prefix', type=Path)
    args = parser.parse_args()
    install(args.wheel, args.prefix)
