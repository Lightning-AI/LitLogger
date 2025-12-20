import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest
from litlogger.diagnostics import (
    collect_system_info,
    get_cli_args,
    get_cpu_name,
    get_cuda_version,
    get_cudnn_version,
    get_gpu_info,
    get_os_info,
)


class TestSystemInfo:
    @patch("platform.system")
    @patch("platform.platform")
    @patch("builtins.open", new_callable=mock_open, read_data='PRETTY_NAME="Ubuntu 20.04.6 LTS"\n')
    def test_get_os_info_linux(self, mock_open, mock_platform_platform, mock_platform_system):
        mock_platform_system.return_value = "Linux"
        mock_platform_platform.return_value = "Linux-5.15.0-1070-aws-x86_64-with-glibc2.31"

        result = get_os_info()
        assert result == "Ubuntu 20.04.6 LTS"

    @patch("platform.platform")
    @patch("builtins.open")
    def test_get_os_info_linux_fallback(self, mock_open, mock_platform_platform):
        mock_platform_platform.return_value = "Linux-5.15.0-1070-aws-x86_64-with-glibc2.31"
        mock_open.side_effect = FileNotFoundError
        result = get_os_info()
        assert result == "Linux-5.15.0-1070-aws-x86_64-with-glibc2.31"

    @patch("platform.system")
    @patch("platform.platform")
    @patch("platform.mac_ver")
    def test_get_os_info_darwin(self, mock_mac_ver, mock_platform_platform, mock_platform_system):
        mock_platform_system.return_value = "Darwin"
        mock_platform_platform.return_value = "macOS-15.0.1-arm64-arm-64bit"
        mock_mac_ver.return_value = ("15.0.1", ("", "", ""), "")
        result = get_os_info()
        assert result == "MacOS 15.0.1"

    @patch("platform.system")
    @patch("platform.platform")
    @patch("platform.mac_ver")
    def test_get_os_info_darwin_fallback(self, mock_mac_ver, mock_platform_platform, mock_platform_system):
        mock_platform_system.return_value = "Darwin"
        mock_platform_platform.return_value = "macOS-15.0.1-arm64-arm-64bit"
        mock_mac_ver.side_effect = Exception("Error getting macOS version")
        result = get_os_info()
        assert result == "macOS-15.0.1-arm64-arm-64bit"

    @pytest.mark.parametrize(
        ("system", "mock_output", "expected_output"),
        [
            (
                "Linux",
                "model name\t: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz\n",
                "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz",
            ),
            ("Darwin", "Apple M1\n", "Apple M1"),
            ("Not supported OS", "", ""),
        ],
    )
    @patch("platform.system")
    @patch("subprocess.check_output")
    def test_get_cpu_name(self, mock_subprocess, mock_platform, system, mock_output, expected_output):
        mock_platform.return_value = system
        mock_subprocess.return_value = mock_output

        assert get_cpu_name() == expected_output

    @patch("subprocess.check_output")
    def test_get_gpu_info(self, mock_subprocess):
        mock_subprocess.return_value = "Nvidia T4, 15360\nNvidia T4, 15360"
        result = get_gpu_info()
        assert result == {"name": "Nvidia T4", "count": 2, "memory_gb": 16}

    @patch("subprocess.check_output")
    def test_get_gpu_info_fallback(self, mock_subprocess):
        mock_subprocess.side_effect = Exception("nvidia-smi not found")
        result = get_gpu_info()
        assert result == {"name": "", "count": 0, "memory_gb": 0}

    @pytest.mark.parametrize(
        ("nvcc_output", "expected_version"),
        [
            (
                "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on "
                "Mon_Apr__3_17:16:06_PDT_2023\nCuda compilation tools, release 12.1, V12.1.105\nBuild "
                "cuda_12.1.r12.1/compiler.32688072_0\n",
                "12.1",
            ),
            ("Cuda compilation tools, release 11.2, V11.2.152", "11.2"),
            ("Cuda compilation tools, release 10.1, V10.1.243", "10.1"),
            ("Some unexpected output", ""),
            ("", ""),
        ],
    )
    def test_get_cuda_version(self, nvcc_output, expected_version):
        with patch("subprocess.check_output", return_value=nvcc_output):
            assert get_cuda_version() == expected_version

    @pytest.mark.parametrize(
        ("side_effect", "expected_version"),
        [
            (FileNotFoundError, ""),  # Case for nvcc not installed
            (subprocess.CalledProcessError(1, "nvcc"), ""),  # Case for nvcc error
        ],
    )
    def test_get_cuda_version_exceptions(self, side_effect, expected_version):
        with patch("subprocess.check_output", side_effect=side_effect):
            assert get_cuda_version() == expected_version

    @pytest.mark.parametrize(
        ("file_content", "expected_version"),
        [
            ("#define CUDNN_MAJOR 8\n#define CUDNN_MINOR 1\n#define CUDNN_PATCHLEVEL 0", "8.1.0"),
            ("#define CUDNN_MAJOR 7\n#define CUDNN_MINOR 6\n#define CUDNN_PATCHLEVEL 5", "7.6.5"),
            ("#define CUDNN_MAJOR 8\n#define CUDNN_MINOR 1", ""),  # Incomplete definitions
            ("", ""),  # Empty file content
        ],
    )
    def test_get_cudnn_version(self, file_content, expected_version):
        with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=file_content)):
            assert get_cudnn_version() == expected_version

    def test_get_cudnn_version_without_cudnn_file(self):
        with patch("os.path.isfile", return_value=False):
            assert get_cudnn_version() == ""

    def test_get_cli_args(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--lr",
                "0.01",
                "--use_gpu",
                "--prompt",
                '"Hello world"',
                "--use_compilation=True",
                "--context",
                "be a good AI",
            ],
        ):
            result = get_cli_args()
            # get_cli_args now returns space-separated args as-is
            expected_output = '--lr 0.01 --use_gpu --prompt "Hello world" --use_compilation=True --context be a good AI'
            assert result == expected_output

    @patch("litlogger.diagnostics.get_cudnn_version")
    @patch("litlogger.diagnostics.get_cuda_version")
    @patch("litlogger.diagnostics.get_gpu_info")
    @patch("litlogger.diagnostics.get_cpu_name")
    @patch("litlogger.diagnostics.get_os_info")
    @patch("subprocess.check_output")
    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("platform.python_version")
    @patch("platform.node")
    @patch("litlogger.__version__", "1.0.0")
    def test_collect_system_info(
        self,
        mock_hostname,
        mock_python_version,
        mock_virtual_memory,
        mock_cpu_count,
        mock_subprocess,
        mock_os_info,
        mock_cpu_name,
        mock_gpu_info,
        mock_cuda_version,
        mock_cudnn_version,
    ):
        mock_hostname.return_value = "test-host"
        mock_python_version.return_value = "3.11.7"
        mock_virtual_memory.return_value = MagicMock(total=38_654_705_664)
        mock_cpu_count.side_effect = (8, 4)
        mock_subprocess.side_effect = [
            "/home/user/test_repo",  # git rev-parse --show-toplevel
            "main",  # git rev-parse --abbrev-ref HEAD
            "abcdef123456",  # git rev-parse HEAD
        ]
        mock_os_info.return_value = "Operating System: Ubuntu 20.04.6 LTS"
        mock_cpu_name.return_value = "Intel Core i7-9700K CPU @ 3.60GHz"
        mock_gpu_info.return_value = {"name": "Nvidia T4", "count": 2, "memory_gb": 16}
        mock_cuda_version.return_value = "11.2"
        mock_cudnn_version.return_value = "8.1.0"

        # get_cudnn_version
        with (
            patch("os.path.isfile", return_value=True),
            patch(
                "builtins.open",
                mock_open(read_data="#define CUDNN_MAJOR 8\n#define CUDNN_MINOR 1\n#define CUDNN_PATCHLEVEL 0"),
            ),
        ):
            result = collect_system_info()

        assert result["git_repo_name"] == "test_repo"
        assert result["git_branch"] == "main"
        assert result["git_commit_hash"] == "abcdef123456"
        assert result["os_name"] == "Operating System: Ubuntu 20.04.6 LTS"
        assert result["python_version"] == "3.11.7"
        assert result["litlogger_version"] == "1.0.0"
        assert result["cpu_name"] == "Intel Core i7-9700K CPU @ 3.60GHz"
        assert result["cpu_count_physical"] == 4
        assert result["cpu_count_logical"] == 8
        assert result["system_memory_gb"] == 36
        assert result["gpu_name"] == "Nvidia T4"
        assert result["gpu_count"] == 2
        assert result["gpu_memory_gb"] == 16
        assert result["cuda_version"] == "11.2"
        assert result["cudnn_version"] == "8.1.0"
        assert result["hostname"] == "test-host"

    def test_collect_system_info_sanity_check(self):
        result = collect_system_info()
        assert result["git_repo_name"] != ""
        assert result["git_branch"] != ""
        assert result["git_commit_hash"] != ""
        assert result["os_name"] != ""
        assert result["python_version"] != ""
        assert result["litlogger_version"] != ""
        assert result["cpu_count_physical"] >= 1
        assert result["cpu_count_logical"] >= 1
        assert result["system_memory_gb"] >= 1
        assert result["hostname"] != ""

        assert result["gpu_count"] >= 0
        if result["gpu_count"] > 0:
            assert result["gpu_name"] != ""
            assert result["gpu_memory_gb"] > 0
            assert result["cuda_version"] != ""
            assert result["cudnn_version"] != ""
        else:
            assert result["gpu_name"] == ""
            assert result["gpu_memory_gb"] == 0
            assert result["cuda_version"] == ""
            assert result["cudnn_version"] == ""
