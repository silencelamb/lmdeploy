"""Assemble the fixed R11 math and native ABI from the frozen research
release."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(root, dest):
    release = root / 'R11-GPT6.2-Sol'
    src = release / 'release'
    manifest = json.loads((release / 'artifacts/source-manifest.json').read_text())
    if manifest['sha256'] != 'eea660ad0a14067255e901abdc6a7da457774882019f80455613e5e2a45edd1f':
        raise ValueError('Expected the frozen R11 release manifest')
    pkg = dest / 'glm53_r11_artifact'
    pkg.mkdir()
    (pkg / '__init__.py').touch()
    kernels = Path('lmdeploy/pytorch/kernels/cuda')
    files = {
        kernels / 'glm53_decode' / (name + '.py'): name + '.py'
        for name in ('bf16_bmm', 'bf16_output', 'kda_norm', 'kda_norm_kernel', 'kpool', 'native', 'r8', 'tiled_weights')
    }
    files[kernels / 'dsv4/hc_prepost.py'] = 'hc_post.py'
    files[Path('benchmark/glm53_monokernel/whole_step.py')] = 'whole_step.py'
    for path, name in files.items():
        if digest(src / path) != manifest['verified_files'][str(path)]:
            raise ValueError(f'Frozen source mismatch: {path}')
        shutil.copy2(src / path, pkg / name)
    native = {
        'libglm53_mla.so': root / 'r5/release_decode',
        'libhc_vector.so': root / 'R09-GPT6.2-Sol/native/release_v2',
        'libglm53_topk_r09.so': root / 'R09-GPT6.2-Sol/native/release_v2',
        'libqkv_sink.so': root / 'r8/release_native_v3',
        'qkv_sink.cubin': root / 'r8/release_native_v3',
        'libpacket_sink.so': root / 'R10-GPT6.2-Sol/native/release/packet_prepare',
        'librouted.so': release / 'native/release',
        'libdevice_loop.so': root / 'vendor/whole_step',
    }
    for name, directory in native.items():
        shutil.copy2(directory / name, pkg / name)
    provider = 'lmdeploy_glm53_megagemm_r6'
    shutil.copytree(
        root / 'r6/release_vendor' / provider, dest / provider, ignore=shutil.ignore_patterns('__pycache__', '*.pyc')
    )
    subprocess.run(['bash', str(Path(__file__).parent / 'native/build.sh'), str(pkg)], check=True)
    # Include the source/build material corresponding to each optional binary.
    kit = dest / 'sources'
    kit.mkdir()
    bench = src / 'benchmark/glm53_monokernel'
    for name in (
        'build_decode_native.py',
        'build_device_loop.py',
        'build_megagemm.py',
        'build_r8.py',
        'build_r09.py',
        'build_r11.py',
        'deepgemm473-glm53.patch',
    ):
        shutil.copy2(bench / name, kit / name)
    shutil.copytree(bench / 'native/decode', kit / 'native/decode')
    shutil.copytree(bench / 'native/whole_step', kit / 'native/whole_step')
    shutil.copytree(root / 'R10-GPT6.2-Sol/native/release/paired8/include', kit / 'paired8/include')
    shutil.copy2(root / 'R10-GPT6.2-Sol/native/release/packet_prepare/packet.cu', kit / 'packet.cu')
    shutil.copytree(
        Path(__file__).parent / 'native', kit / 'projection', ignore=shutil.ignore_patterns('rebuild_r11.sh')
    )
    shutil.copy2(Path(__file__).parent / 'native/rebuild_r11.sh', kit / 'rebuild_r11.sh')
    licenses = dest / 'licenses'
    licenses.mkdir()
    for project, license_file in {
        'LMDeploy': src / 'LICENSE',
        'DeepGEMM': root / 'references/deepgemm_pr473/LICENSE',
        'FlashMLA': root / 'references/flashmla_roofline/LICENSE',
        'CUTLASS': root / 'references/flashmla_cutlass_pinned/LICENSE.txt',
    }.items():
        shutil.copy2(license_file, licenses / (project + '.txt'))
    sums = {str(f.relative_to(dest)): digest(f) for f in sorted(dest.rglob('*')) if f.is_file()}
    (dest / 'sha256.json').write_text(json.dumps(sums, indent=2) + '\n')


def build(root, dest):
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        stage = Path(scratch) / 'bundle'
        stage.mkdir()
        assemble(root, stage)
        stage.rename(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('research_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.research_root, args.output)
