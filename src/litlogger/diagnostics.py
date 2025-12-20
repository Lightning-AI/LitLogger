# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import distributions
from typing import List

import psutil

import litlogger


def get_os_info() -> str:
    """Return a pretty name of operating system and its version: 'Ubuntu 20.04.6 LTS' or 'MacOS 15.0.1'.

    If it's not possible, then the more verbose name is returned:
    'Linux-5.15.0-1070-aws-x86_64-with-glibc2.31' or 'macOS-15.0.1-arm64-arm-64bit'.
    """
    os_name = platform.system()
    default_os_name = platform.platform()

    if os_name == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME"):
                        return line.strip().split("=")[1].strip('"')
            return default_os_name
        except FileNotFoundError:
            return default_os_name

    if os_name == "Darwin":
        try:
            version = platform.mac_ver()[0]
            return f"MacOS {version}"
        except Exception:
            return default_os_name

    return default_os_name


def get_cpu_name() -> str:
    """Return the name of the CPU (e.g. Apple M3 Pro, Intel(R) Xeon(R) Platinum 8488C)."""
    system = platform.system()
    try:
        if system == "Linux":
            cpu_info = subprocess.check_output("grep 'model name' /proc/cpuinfo | head -n 1", shell=True, text=True)
            return cpu_info.split(":")[1].strip()

        if system == "Darwin":
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            return cpu_info.strip()

        return ""
    except Exception:
        return ""


def get_gpu_info() -> dict:
    """Return Nvidia GPU info as a dict with keys: name, count, and memory_gb.

    The memory_gb value is a rounded integer number of gigabytes. If no NVIDIA GPUs are present
    or the 'nvidia-smi' command is unavailable, returns zeros and an empty name.
    Example: {"name": "NVIDIA T4", "count": 4, "memory_gb": 16}
    """
    default_output = {
        "name": "",
        "count": 0,
        "memory_gb": 0,
    }

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
        ).strip()

        if not output:
            return default_output

        gpu_list = output.split("\n")
        gpu_name, gpu_memory_mib = re.split(r",\s+", gpu_list[0])
        gpu_memory_gb = round((int(gpu_memory_mib) / 1e6) * 1024)

        return {"name": gpu_name, "count": len(gpu_list), "memory_gb": gpu_memory_gb}

    except Exception:
        return default_output


def get_cuda_version() -> str:
    """Return the CUDA version installed on the system."""
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        match = re.search(r"release (\d+\.\d+)", output)
        return match.group(1) if match else ""
    except Exception:
        return ""


def get_cudnn_version() -> str:
    """Return the cuDNN version installed on the system."""
    cudnn_path = "/usr/include/cudnn_version.h"
    if os.path.isfile(cudnn_path):
        # Read the file content and find cuDNN version
        with open(cudnn_path, encoding="utf-8") as fin:
            content = fin.read()
            # Find the major, minor, and patch versions of cuDNN
            major = re.search(r"#define CUDNN_MAJOR (\d+)", content)
            minor = re.search(r"#define CUDNN_MINOR (\d+)", content)
            patch = re.search(r"#define CUDNN_PATCHLEVEL (\d+)", content)
            if major and minor and patch:
                return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return ""


def get_cli_args() -> str:
    """Get command-line arguments as a space-separated string.

    Returns the arguments passed to the script (sys.argv[1:]) joined by spaces.

    Example:
        For `python script.py -a --verbose --config=foo.yaml`,
        returns: `-a --verbose --config=foo.yaml`
    """
    return " ".join(sys.argv[1:])


def collect_system_info() -> dict:
    """Collect git, system, hardware, environment and CLI information.

    Returns a dictionary including (when available):
    - git_repo_name, git_branch, git_commit_hash
    - os_name, python_version, litlogger_version, installed_packages
    - cpu_name, cpu_count_logical, cpu_count_physical, system_memory_gb
    - gpu_name, gpu_count, gpu_memory_gb, cuda_version, cudnn_version
    - execution_command, cli_args, hostname

    TODO: Add get_system_metrics() function to track ongoing resource usage:
          - CPU usage % (psutil.cpu_percent())
          - Memory usage % (psutil.virtual_memory().percent)
          - GPU utilization % (nvidia-smi --query-gpu=utilization.gpu)
          - GPU memory usage (nvidia-smi --query-gpu=memory.used)
          - Disk I/O (psutil.disk_io_counters())
          - Network I/O (psutil.net_io_counters())
          And add experiment.log_system_metrics() to log these metrics periodically.
    """

    # git
    def get_git_info(command: List[str], default_message: str) -> str:
        try:
            return subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            return default_message

    # in case git is not installed
    try:
        git_repo_name = get_git_info(["git", "rev-parse", "--show-toplevel"], "Not a git repository").split("/")[-1]
        git_branch = get_git_info(["git", "rev-parse", "--abbrev-ref", "HEAD"], "No branch found")
        git_commit_hash = get_git_info(["git", "rev-parse", "HEAD"], "No commit hash found")
    except FileNotFoundError:
        git_repo_name = None
        git_branch = None
        git_commit_hash = None

    # software
    litlogger_version = litlogger.__version__
    installed_packages = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))

    # hardware
    system_memory_gb = psutil.virtual_memory().total // 1024**3
    gpu_info = get_gpu_info()

    # args - capture python interpreter and script path (cli_args has the arguments separately)
    script_path = os.path.abspath(sys.argv[0])
    execution_command = f"{sys.executable} {script_path}"
    cli_args = get_cli_args()

    return {
        "git_repo_name": git_repo_name,
        "git_branch": git_branch,
        "git_commit_hash": git_commit_hash,
        "os_name": get_os_info(),
        "python_version": platform.python_version(),
        "litlogger_version": litlogger_version,
        "installed_packages": installed_packages,
        "cpu_name": get_cpu_name(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "system_memory_gb": system_memory_gb,
        "gpu_name": gpu_info["name"],
        "gpu_count": gpu_info["count"],
        "gpu_memory_gb": gpu_info["memory_gb"],
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
        "execution_command": execution_command,
        "cli_args": cli_args,
        "hostname": platform.node(),
    }
