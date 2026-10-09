#!/usr/bin/env bash
# Install the runtime and build tools. Does not start iKataGo or download models.
set -Eeuo pipefail
export LC_ALL=C

runtime_only=0
install_driver=0
check_only=0
dry_run=0
prefix=/opt/katago-rtx50
usage() {
  cat <<'EOF'
Usage: sudo bash setup-linux.sh [options]
  --runtime-only    Install runtime libraries without the full CUDA toolkit
  --install-driver  Also install NVIDIA open kernel driver on a physical host
  --prefix PATH     Directory for env.sh, logs and version records
                    Default: /opt/katago-rtx50
  --check           Check the environment without installing anything
  --dry-run         Print the plan without changing anything
  --help            Show this help

Supports Ubuntu 22.04/24.04 x86_64. Other Linux systems can build the engine
using their own package manager; this limit applies only to this installer.
CUDA 13.2; cuDNN 9.24.x (an already installed newer version is retained).
Cloud/container users: keep the host-provided GPU driver; do not add --install-driver.
EOF
}
while (($#)); do
  case "$1" in
    --runtime-only) runtime_only=1 ;;
    --install-driver) install_driver=1 ;;
    --check) check_only=1 ;;
    --dry-run) dry_run=1 ;;
    --prefix) (($# >= 2)) || { usage; exit 2; }; prefix=$2; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done
[[ $prefix == /* && $prefix != / && $prefix != *$'\n'* ]] || { echo 'Use an absolute --prefix other than /' >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || { echo 'This release requires x86_64.' >&2; exit 2; }
source /etc/os-release
case "${ID}:${VERSION_ID}" in
  ubuntu:22.04) repo=ubuntu2204 ;;
  ubuntu:24.04) repo=ubuntu2404 ;;
  *) echo 'Automatic installation supports Ubuntu 22.04/24.04. See BUILD.md for manual builds.' >&2; exit 2 ;;
esac
url="https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64"
if ((dry_run)); then
  printf 'Repository: %s\nCUDA: 13.2\ncuDNN: 9.24.x or retain newer installed version\nPrefix: %s\n' "$url" "$prefix"
  printf 'Runtime only: %s\nInstall host driver: %s\n' "$runtime_only" "$install_driver"
  echo 'Installs apt packages and NVIDIA repository; writes env.sh. Does not reboot or start services.'
  exit 0
fi

check_gpu() {
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > /dev/null 2>&1 || {
    echo 'NVIDIA driver is unavailable. On a cloud machine, enable its GPU driver first.' >&2
    echo 'On a new physical host, run with --install-driver, reboot, then run again.' >&2
    return 1
  }
  local rows
  rows=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)
  printf '%s\n' "$rows"
  if ! printf '%s\n' "$rows" | grep -Eq ',[[:space:]]*12\.0[[:space:]]*,'; then
    echo 'No SM120 GPU (compute capability 12.0) was found.' >&2; return 1
  fi
  if printf '%s\n' "$rows" | awk -F',' '$2+0 == 12.0 {split($3,v,"."); if(v[1]+0<580) exit 1}'; then :; else
    echo 'CUDA 13 requires a newer NVIDIA driver. Tested driver: 595.71.05.' >&2; return 1
  fi
}

probe() {
  check_gpu
  if ((!runtime_only)); then
    /usr/local/cuda-13.2/bin/nvcc --version
    cmake --version | head -1
    g++ --version | head -1
  fi
  local python
  python=$(command -v python3 || true)
  [[ -n $python ]] || python=/root/miniconda3/bin/python
  "$python" - <<'PY'
import ctypes, json
import yaml
cuda = ctypes.CDLL('/usr/local/cuda-13.2/lib64/libcudart.so.13')
version = ctypes.c_int()
if cuda.cudaRuntimeGetVersion(ctypes.byref(version)) != 0:
    raise SystemExit('Cannot query CUDA runtime')
if version.value // 10 != 1302:
    raise SystemExit('Expected CUDA 13.2 runtime, got ' + str(version.value))
cudnn = ctypes.CDLL('libcudnn.so.9')
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
cudnn_version = cudnn.cudnnGetVersion()
if cudnn_version < 92400:
    raise SystemExit('cuDNN must be at least 9.24.0')
for name in ('libcublas.so.13', 'libcublasLt.so.13', 'libnvrtc.so.13', 'libzip.so.4', 'libz.so.1'):
    ctypes.CDLL(name)
print(json.dumps(dict(cudaRuntime=version.value, cudnnRuntime=cudnn_version, pyYaml=yaml.__version__)))
PY
}
export PATH="/usr/local/cuda-13.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ((check_only)); then probe; exit 0; fi
((EUID == 0)) || { echo 'Run with sudo (or as root).' >&2; exit 2; }
if ((install_driver)); then
  if [[ -e /.dockerenv || -e /run/.containerenv ]] || grep -qi microsoft /proc/sys/kernel/osrelease; then
    echo 'Driver installation is for physical hosts only; use the host driver in containers/WSL.' >&2; exit 2
  fi
else
  check_gpu
fi
mkdir -p "$prefix"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$prefix/setup-$stamp.log"
exec > >(tee -a "$log") 2>&1
trap 'echo "Setup failed at line $LINENO. Log: $log. Fix the error and run the script again." >&2' ERR
echo 'Installing CUDA 13.2 and cuDNN; the existing driver is retained unless --install-driver was given.'
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg python3 python3-yaml python3-venv zlib1g libbz2-1.0
keyring="$prefix/cuda-keyring_1.1-1_all.deb"
if [[ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]]; then
  curl --fail --location --retry 3 "$url/cuda-keyring_1.1-1_all.deb" -o "$keyring"
  dpkg -i "$keyring"
fi
# Some cloud images retain cuda-keyring in dpkg but remove its APT source file.
if ! grep -Rqs "$url" /etc/apt/sources.list /etc/apt/sources.list.d; then
  printf 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] %s /\n' "$url" > "/etc/apt/sources.list.d/cuda-$repo-x86_64.list"
fi
apt-get update
if [[ $VERSION_ID == 24.04 ]]; then zip_package=libzip4t64; else zip_package=libzip4; fi
apt-get install -y --no-install-recommends "$zip_package"

# Select one coherent cuDNN version for runtime and headers. Never silently downgrade.
installed=$(dpkg-query -W -f='${Version}' libcudnn9-cuda-13 2>/dev/null || true)
if [[ -n $installed ]] && dpkg --compare-versions "$installed" ge 9.24.0; then
  cudnn_version=$installed
else
  cudnn_version=$(apt-cache madison libcudnn9-cuda-13 | awk '$3 ~ /^9\.24\./ {print $3}' | sort -V | tail -1)
fi
[[ -n $cudnn_version ]] || { echo 'NVIDIA repository has no cuDNN 9.24 package; no fallback version was selected.'; exit 1; }
packages=("libcudnn9-cuda-13=$cudnn_version")
if ((runtime_only)); then
  packages+=(cuda-cudart-13-2 cuda-nvrtc-13-2 libcublas-13-2)
else
  packages+=(cuda-compiler-13-2 cuda-libraries-dev-13-2 cuda-nvtx-13-2
    "libcudnn9-dev-cuda-13=$cudnn_version" "libcudnn9-headers-cuda-13=$cudnn_version"
    build-essential cmake ninja-build git zlib1g-dev libzip-dev)
fi
# Cloud images sometimes hold an older cuDNN. Only lift holds for the three
# explicitly requested cuDNN packages, and restore them even after a failure.
held_cudnn=()
holds=$(apt-mark showhold)
for package in libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13; do
  if printf '%s\n' "$holds" | grep -Fxq "$package"; then held_cudnn+=("$package"); fi
done
restore_holds() {
  if ((${#held_cudnn[@]})); then apt-mark hold "${held_cudnn[@]}"; fi
}
trap restore_holds EXIT
if ((${#held_cudnn[@]})); then apt-mark unhold "${held_cudnn[@]}"; fi
apt-get install -y --no-install-recommends "${packages[@]}"
restore_holds
held_cudnn=()
ldconfig
cat > "$prefix/env.sh" <<'ENV'
# Source this file before compiling, starting KataGo, or starting iKataGo.
export CUDA_PATH=/usr/local/cuda-13.2
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="/usr/local/cuda-13.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENV
if ((install_driver)); then
  apt-get install -y --no-install-recommends "linux-headers-$(uname -r)" nvidia-open
  echo 'Driver packages installed. Reboot yourself, complete Secure Boot/MOK enrollment if requested, then run this script again without --install-driver.'
  exit 10
fi
probe | tee "$prefix/verification-$stamp.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' 'cuda-*13-2*' 'libcublas*13-2*' 'libcudnn9*-cuda-13' "$zip_package" > "$prefix/packages-$stamp.txt"
echo "Environment ready. Before use: source $prefix/env.sh"
echo 'If using the binary package, run ./run-katago.sh version; see README.md to load a model.'
