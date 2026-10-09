#!/usr/bin/env python3
"""Build the SM120 edition on a CUDA-supported Windows or Linux toolchain.

Does not install packages, choose a model, run a GPU workload, or deploy iKataGo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone


class BuildError(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv, env, logfile=None):
    print("+ " + subprocess.list2cmdline([str(x) for x in argv]), flush=True)
    proc = subprocess.Popen([str(x) for x in argv], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines = []
    with (logfile.open("w", encoding="utf-8") if logfile else open(os.devnull, "w")) as log:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
    if proc.wait() != 0:
        raise BuildError("Command failed; inspect " + str(logfile or "the output above"))
    return "".join(lines)


def read_cache(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    return values


def require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise BuildError(description + " not found: " + str(path))
    return path


def extract_source(bundle, destination):
    archive = require_file(bundle / "katago-rtx50-source.tar.gz", "Source archive")
    hash_file = require_file(bundle / "source-sha256.txt", "Source checksum")
    fields = hash_file.read_text(encoding="utf-8-sig").strip().split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise BuildError("source-sha256.txt must start with the archive's SHA256")
    expected = fields[0].lower()
    if sha256(archive) != expected:
        raise BuildError("Source archive SHA256 mismatch; obtain the complete matching release package")
    # Never silently reuse a partially extracted or user-modified source tree.
    if destination.exists():
        raise BuildError("Source extraction directory already exists: " + str(destination) +
                         ". Use --source with that tree to resume deliberately, or a new --build-dir.")
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name
                    or not path.parts or path.parts[0] != "KataGo"
                    or not (member.isfile() or member.isdir())):
                raise BuildError("Unexpected source archive entry: " + member.name)
        # The checked package contains only regular files/directories below KataGo.
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(member.mode & 0o777)
    return destination / "KataGo", expected


def cudnn_header_version(include):
    header = require_file(include / "cudnn_version.h", "Selected cuDNN version header")
    text = header.read_text(encoding="utf-8", errors="replace")
    parts = []
    for key in ("MAJOR", "MINOR", "PATCHLEVEL"):
        match = re.search(r"^\s*#define\s+CUDNN_" + key + r"\s+(\d+)", text, re.M)
        if not match:
            raise BuildError("Cannot parse cuDNN version in " + str(header))
        parts.append(int(match[1]))
    if tuple(parts) < (9, 24, 0):
        raise BuildError("Selected cuDNN headers require >= 9.24.0; got " + ".".join(map(str, parts)))
    return parts


def runtime_library(args, link_library, include):
    if args.cudnn_runtime:
        return require_file(args.cudnn_runtime, "cuDNN runtime")
    if os.name != "nt":
        return link_library
    candidates = []
    # A .lib is an import library, not a loadable DLL. Search nearby SDK layouts,
    # then the active environment; --cudnn-runtime removes ambiguity.
    for root in (link_library.parent, link_library.parent.parent,
                 link_library.parent.parent.parent, include.parent):
        for subdir in ("bin", "bin/x64", "bin/12.0", "bin/13.0"):
            candidates.extend((root / subdir).glob("cudnn64_*.dll"))
    found = shutil.which("cudnn64_9.dll")
    if found:
        candidates.append(Path(found))
    unique = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(unique) != 1:
        raise BuildError("Cannot unambiguously select cuDNN runtime DLL. Pass --cudnn-runtime PATH/cudnn64_9.dll")
    return unique[0]


def prepend_env(env, variable, paths):
    values = [str(p) for p in paths if Path(p).is_dir()]
    if env.get(variable):
        values.append(env[variable])
    env[variable] = os.pathsep.join(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="KataGo source tree (or its cpp directory); otherwise verify/extract bundled archive")
    parser.add_argument("--build-dir", required=True, type=Path, help="Build and log directory")
    parser.add_argument("--cuda-root", type=Path, help="CUDA Toolkit root; otherwise CUDA_PATH/CUDA_HOME or nvcc on PATH")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--generator", default="Ninja", help="CMake generator (default: Ninja)")
    parser.add_argument("--cudnn-include", type=Path)
    parser.add_argument("--cudnn-library", type=Path, help="Linux libcudnn.so or Windows cudnn.lib")
    parser.add_argument("--cudnn-runtime", type=Path, help="Actual cuDNN .so/.dll, useful for separate SDK/runtime installs")
    parser.add_argument("--zlib-include", type=Path)
    parser.add_argument("--zlib-library", type=Path)
    parser.add_argument("--cmake-arg", action="append", default=[], help="Extra CMake argument; e.g. --cmake-arg=-DCMAKE_TOOLCHAIN_FILE=...")
    args = parser.parse_args()
    if args.jobs < 1:
        raise BuildError("--jobs must be positive")
    cmake = shutil.which("cmake")
    if not cmake:
        raise BuildError("CMake is missing. Install CMake >= 3.18.2 and put it on PATH")
    if "Ninja" in args.generator and not shutil.which("ninja"):
        raise BuildError("Ninja is missing. Install Ninja or select an installed CMake generator with --generator")
    if os.name == "nt" and "Ninja" in args.generator and not shutil.which("cl"):
        raise BuildError("MSVC cl.exe is not on PATH. Run from an x64 Visual Studio developer terminal with a CUDA-supported MSVC toolset")
    env = os.environ.copy()
    cuda_root = args.cuda_root or env.get("CUDA_PATH") or env.get("CUDA_HOME")
    if cuda_root:
        cuda_root = Path(cuda_root).expanduser().resolve()
        nvcc = require_file(cuda_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"), "CUDA compiler")
    else:
        found = shutil.which("nvcc")
        if not found:
            raise BuildError("nvcc is missing. Pass --cuda-root for an installed CUDA Toolkit >= 13.2")
        nvcc = Path(found).resolve()
        cuda_root = nvcc.parent.parent
    prepend_env(env, "PATH", [cuda_root / "bin"])
    version_output = run([nvcc, "--version"], env)
    match = re.search(r"release\s+(\d+)\.(\d+)", version_output)
    if not match or tuple(map(int, match.groups())) < (13, 2):
        raise BuildError("Selected nvcc must be CUDA >= 13.2")
    build = args.build_dir.expanduser().resolve()
    build.mkdir(parents=True, exist_ok=True)
    archive_sha = None
    if args.source:
        source = args.source.expanduser().resolve()
    else:
        source, archive_sha = extract_source(Path(__file__).resolve().parent, build / "source")
    cpp = source if source.name == "cpp" else source / "cpp"
    require_file(cpp / "CMakeLists.txt", "KataGo CMake project")
    for header in ("cutlass/include/cutlass/cutlass.h", "cudnn-frontend/include/cudnn_frontend.h"):
        require_file(cpp / "external" / header, "Vendored CUDA dependency")
    configure = [cmake, "-S", cpp, "-B", build, "-G", args.generator]
    if os.name == "nt" and args.generator.startswith("Visual Studio"):
        configure += ["-A", "x64", "-T", "cuda=" + str(cuda_root)]
    configure += args.cmake_arg
    configure += ["-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release", "-DNO_GIT_REVISION=ON",
                  "-DCMAKE_CUDA_ARCHITECTURES=120", "-DKATAGO_SM120_ONLY=ON",
                  "-DBUILD_DISTRIBUTED=OFF", "-DKATAGO_AUTO_FETCH_DEPS=OFF",
                  "-DCMAKE_CUDA_COMPILER=" + str(nvcc), "-DCUDAToolkit_ROOT=" + str(cuda_root)]
    for option, value in (("CUDNN_INCLUDE_DIR", args.cudnn_include), ("CUDNN_LIBRARY", args.cudnn_library),
                          ("ZLIB_INCLUDE_DIR", args.zlib_include), ("ZLIB_LIBRARY", args.zlib_library)):
        if value:
            configure.append("-D" + option + "=" + str(value.expanduser().resolve()))
    configure_output = run(configure, env, build / "configure.log")
    cache = read_cache(build / "CMakeCache.txt")
    if cache.get("CMAKE_CUDA_ARCHITECTURES") != "120" or cache.get("KATAGO_SM120_ONLY") not in ("ON", "1", "TRUE"):
        raise BuildError("CMake did not select the requested SM120 build; inspect CMakeCache.txt")
    if "Manually-specified variables were not used by the project" in configure_output and "KATAGO_SM120_ONLY" in configure_output.split("Manually-specified variables were not used by the project")[-1]:
        raise BuildError("This source tree does not implement KATAGO_SM120_ONLY")
    include = Path(cache.get("CUDNN_INCLUDE_DIR", ""))
    header_version = cudnn_header_version(include)
    library = require_file(cache.get("CUDNN_LIBRARY", ""), "Selected cuDNN link library")
    runtime = runtime_library(args, library, include)
    runtime_paths = [runtime.parent, cuda_root / "bin", cuda_root / "lib64", cuda_root / "lib", cuda_root / "targets/x86_64-linux/lib"]
    prepend_env(env, "PATH" if os.name == "nt" else "LD_LIBRARY_PATH", runtime_paths)
    # Run in a new process so the platform loader receives the runtime paths before
    # startup. On Windows Python >= 3.8 also requires add_dll_directory for ctypes.
    probe = ("import ctypes,json,os,sys; "
             "dirs=[os.add_dll_directory(p) for p in json.loads(sys.argv[2]) if os.path.isdir(p)] if os.name=='nt' else []; "
             "lib=ctypes.CDLL(sys.argv[1]); lib.cudnnGetVersion.restype=ctypes.c_size_t; "
             "print(json.dumps({'runtime':sys.argv[1],'version':lib.cudnnGetVersion()}))")
    runtime_output = run([sys.executable, "-c", probe, runtime, json.dumps([str(p) for p in runtime_paths])], env, build / "cudnn-runtime.json")
    runtime_version = json.loads(runtime_output)["version"]
    expected = header_version[0] * 10000 + header_version[1] * 100 + header_version[2]
    if runtime_version != expected:
        raise BuildError("Selected cuDNN header/runtime versions differ: " + str(expected) + " versus " + str(runtime_version))
    run([cmake, "--build", build, "--config", "Release", "--parallel", str(args.jobs)], env, build / "build.log")
    basename = "katago.exe" if os.name == "nt" else "katago"
    candidates = [build / "Release" / basename, build / basename]
    binary = next((p for p in candidates if p.is_file()), None)
    if not binary:
        raise BuildError("Build finished but executable was not found in " + str(build))
    if sys.platform.startswith("linux"):
        ldd = shutil.which("ldd")
        if not ldd:
            raise BuildError("ldd is needed to check the Linux executable's runtime libraries")
        linked = run([ldd, binary], env, build / "linked-libraries.txt")
        if "not found" in linked:
            raise BuildError("The executable has unresolved runtime libraries; inspect linked-libraries.txt")
        selected = re.search(r"libcudnn\.so\.\d+\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)", linked)
        if not selected or Path(selected[1]).resolve() != runtime.resolve():
            raise BuildError("The executable resolves a different cuDNN runtime than the one checked; inspect linked-libraries.txt")
    if os.name == "nt":
        shadow = binary.parent / runtime.name
        if shadow.is_file() and sha256(shadow) != sha256(runtime):
            raise BuildError("A different cuDNN DLL beside katago.exe shadows the selected runtime: " + str(shadow))
    # Dynamic dependencies are tested even though this does not initialize a GPU.
    run([binary, "version"], env, build / "version.txt")
    manifest = {"completedUtc": datetime.now(timezone.utc).isoformat(), "source": str(source),
                "sourceArchiveSha256": archive_sha, "modelRestriction": None,
                "cudaRoot": str(cuda_root), "cudaCompiler": str(nvcc), "nvccVersion": version_output,
                "cudaArchitectures": "120", "sm120Only": True,
                "cudnnInclude": str(include.resolve()), "cudnnLibrary": str(library),
                "cudnnRuntime": str(runtime), "cudnnVersion": runtime_version,
                "cutlassEnabled": "enabling the fused FFN kernel" in configure_output,
                "runtimeSearchDirectories": [str(p) for p in runtime_paths if p.is_dir()],
                "binary": str(binary), "binarySha256": sha256(binary)}
    (build / "build-result.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nBuild complete: " + str(binary))
    print("Record: " + str(build / "build-result.json"))
    print("GPU inference and model correctness have not been tested by this build-only script.")
    if not manifest["cutlassEnabled"]:
        print("CUTLASS paths were unavailable on this toolchain; the CUDA fallback was built.")


if __name__ == "__main__":
    try:
        main()
    except (BuildError, OSError, tarfile.TarError) as error:
        print("Build stopped: " + str(error), file=sys.stderr)
        sys.exit(1)
