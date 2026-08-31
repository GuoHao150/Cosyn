# Apptainer 容器化构建

本目录提供将 cosyn 及其 libtorch 运行时打包成 Apptainer (Singularity) 镜像的方案。

## 支持镜像

| 文件 | 用途 | libtorch 来源 | 大小 |
|---|---|---|---|
| `cosyn_cpu.def` + `build_cpu.sh` | CPU-only 运行 | 下载 libtorch 2.5.1+cpu | ~1 GB |
| `cosyn_gpu.def` + `build_gpu.sh` | GPU 加速 PLL（自包含） | 下载 libtorch 2.5.1+cu118 | ~4–8 GB |

## 构建前准备

1. 安装 Apptainer ≥ 1.0（本机已安装 `1.5.2`）。
2. 确保支持 `--fakeroot` 构建：
   ```bash
   apptainer build --fakeroot /tmp/test.sif docker://docker.m.daocloud.io/library/ubuntu:24.04
   ```
   如果失败，可改用 root 构建：`sudo apptainer build ...`
3. （可选）设置镜像源环境变量加速下载：
   ```bash
   # 使用清华大学镜像下载 libtorch（如果有对应镜像）
   export LIBTORCH_URL="https://your-mirror.com/libtorch-cpu.zip"
   ```

## 国内镜像源说明

为了在中国大陆网络环境下顺利构建，`.def` 文件中已默认配置：

- **CPU 基础镜像**：`docker.m.daocloud.io/library/ubuntu:24.04`
  - 如果该镜像失效，可替换为 `docker.1panel.live/library/ubuntu:24.04`。
- **GPU 基础镜像**：`docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
  - 如果 DaoCloud 没有该镜像或拉取失败，可替换为其他可用镜像源，例如 `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`（需要能访问 Docker Hub / NGC）。
- **APT**：清华大学 Ubuntu 镜像 `mirrors.tuna.tsinghua.edu.cn`。NVIDIA 容器镜像已预装 `ca-certificates`，故使用 HTTPS。
- **Rust**：`rustup-init` 脚本从官方 `sh.rustup.rs` 获取；Rust toolchain 下载使用清华大学镜像（通过 `RUSTUP_DIST_SERVER`）。
- **Cargo crates**：`sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/`。
- **libtorch**：默认从 PyTorch 官方下载；若速度慢，可手动下载后放到 `apptainer/cache/gpu/libtorch/`，脚本会自动跳过下载。

## 构建

### CPU 版

```bash
./apptainer/build_cpu.sh
```

构建产物：`apptainer/cosyn_cpu.sif`

### GPU 版

```bash
./apptainer/build_gpu.sh
```

构建产物：`apptainer/cosyn_gpu.sif`

该镜像基于 `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`，并在内部预装了：
- CUDA 11.8 runtime 及开发库（`libcudart`、`libcublas`、`libcusparse`、`libcurand`、`libcufft`、`libnvrtc` 等）
- cuDNN 8
- NCCL
- libtorch 2.5.1+cu118

因此运行时**不再需要**主机安装 CUDA toolkit、cuDNN 或 NCCL。主机只需满足：
- 有 NVIDIA GPU；
- 已安装 NVIDIA 驱动，且驱动版本对应的 CUDA 版本 **≥ 11.8**（即驱动 **≥ 450.80.02 / R450+**）；
- Apptainer 已安装并支持 `--nv`。

> 提示：如果你无法访问 Docker Hub / NGC，可修改 `cosyn_gpu.def` 第一行 `From:` 为本地可用镜像，例如 `docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`。

## 运行

### CPU 版

```bash
./apptainer/run_cpu.sh pll \
  -f example/pll_test/test_seqs_valid.fa \
  -j /mnt/data/Users/guohao/wet_lab_works/codon_transformer_test/codon_transformer_eval_26_5_19.pt \
  -o example/pll_test -p cosyn_container
```

运行脚本会自动：
- 将当前目录绑定到容器的 `/workspace`；
- 解析 `-j`/`--model_path`/`--model-path` 参数，将其所在目录也绑定进容器，因此绝对路径的模型文件在容器内依然可用。

### GPU 版

```bash
./apptainer/run_gpu.sh pll \
  -f example/pll_test/test_seqs_valid.fa \
  -j /path/to/codon_transformer_eval.pt \
  -o example/pll_test -p cosyn_container
```

GPU 版运行时会自动加上 `--nv`，将主机的 NVIDIA 驱动绑定进容器。镜像内部已包含 CUDA 11.8 runtime、cuDNN 8、NCCL 及 libtorch cu118，因此：

- **不需要**主机安装 CUDA toolkit；
- **不需要**主机安装 cuDNN / NCCL / cuSPARSELt / NVSHMEM；
- 只需主机驱动支持 CUDA 11.8（驱动 ≥ 450.80.02）。

运行脚本会自动：
- 将当前目录绑定到容器的 `/workspace`；
- 解析 `-j`/`--model_path`/`--model-path` 参数，将其所在目录也绑定进容器，因此绝对路径的模型文件在容器内依然可用。

如果确实需要覆盖镜像内置的 CUDA 库（例如主机有更新的 CUDA 版本且想使用它），可以手动绑定：

```bash
./apptainer/run_gpu.sh --bind /path/to/cuda:/opt/cuda-override \
  --env "LD_LIBRARY_PATH=/opt/cuda-override/lib64:$LD_LIBRARY_PATH" \
  pll -f ... -j ... -o ... -p ...
```

## 直接使用 apptainer run

```bash
# CPU
apptainer run --bind $(pwd):/workspace --pwd /workspace \
  apptainer/cosyn_cpu.sif pll -f example/pll_test/test_seqs_valid.fa ...

# GPU（镜像已内置 CUDA 11.8，只需 --nv 挂载驱动）
apptainer run --nv --bind $(pwd):/workspace --pwd /workspace \
  apptainer/cosyn_gpu.sif pll -f example/pll_test/test_seqs_valid.fa ...
```

## 注意事项

- 容器内会从源码重新编译 cosyn，以确保与容器内的 libtorch ABI 完全匹配。
- 构建脚本会把源码、`Cargo.toml`、`Makefile` 等复制到 `apptainer/cache/src/` 中；编译产物 `target/` 在 `%post` 末尾会被清理以减小镜像体积。
- 构建好的 `.sif` 文件较大，已加入 `.gitignore`，不会进入 git 仓库。
