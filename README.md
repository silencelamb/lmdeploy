# GLM-5.3 R11 generated runtime

Optional artifact for InternLM/lmdeploy pull request #5195. This package contains precompiled SM90a kernels, the isolated R6 expert runtime, the frozen Python math, corresponding native sources and rebuild scripts, and license notices. Model weights and request data are supplied separately.

Target environment is single-node Linux x86-64, Python 3.12, PyTorch 2.12.1+cu130, Triton 3.7.1, TileLang 0.1.11, and eight NVIDIA H200s with peer access. Verify SHA256SUMS and extract the archive into an empty directory. Add its root and the PR's examples/specialized_execution directory to the worker PYTHONPATH. Select glm53_full_r11:FullR11 on the LMDeploy PyTorch engine with the default fallback MoE backend, DP8, EP8 and TP1. The integration guide in the PR gives the serving and verification commands.

The archive root contains sha256.json with individual file digests. The sources directory contains the build scripts, the R10 paired8 headers, the R11 routed grid and packet sources, and the DeepGEMM/FlashMLA patches and pinned dependencies. For a source build, run bash sources/rebuild_r11.sh with a new output directory. The native runtime and source artifact can be mirrored together.
