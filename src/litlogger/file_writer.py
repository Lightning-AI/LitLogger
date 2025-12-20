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
"""Binary file writer for efficient on-disk metric buffering and upload."""

import json
import os
import struct
import tarfile
from typing import Dict, List

import numpy as np

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.client import LitRestClient
from litlogger.types import Metrics, MetricsTracker, MetricValue


def _sanitize_version_for_path(version: str) -> str:
    """Sanitize version string for use in file paths.

    Windows doesn't allow colons in filenames, so replace them with hyphens.
    """
    return version.replace(":", "-")


class BinaryFileWriter:
    """Write metrics to per-series .litbin files and upload them as a single archive.

    The file format starts with a JSON header (size-prefixed, big-endian uint32) followed by
    packed binary records depending on the selected storage mode (values only, values+steps,
    or values+relative_time+steps).
    """

    def __init__(
        self,
        log_dir: str,
        version: str,
        store_step: bool,
        store_created_at: bool,
        teamspace_id: str,
        metrics_store_id: str,
        cloud_account: str,
        client: LitRestClient | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.version = version
        self.store_step = store_step
        self.store_created_at = store_created_at

        self.teamspace_id = teamspace_id
        self.metrics_store_id = metrics_store_id
        self.cloud_account = cloud_account
        self._artifacts_api = ArtifactsApi(client=client or LitRestClient(max_retries=5))
        self.files = {}

    def store(self, metrics: Dict[str, Metrics], trackers: Dict[str, MetricsTracker] | None = None) -> None:
        """Append metric values to per-series binary files, creating them if needed.

        Writes a small header once per file, then appends:
        - value (float32) when neither step nor timestamp is stored
        - step (uint64) + value (float32) when only steps are stored
        - step (uint64) + relative_time_seconds (float32) + value (float32) when timestamps are stored
        """
        for k, v in metrics.items():
            if k not in self.files:
                filepath = os.path.join(self.log_dir, f"{k}.litbin")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                self.files[k] = open(filepath, "wb")  # type: ignore # noqa: SIM115

                # Convert datetime to ISO string for header if present
                created_at_str = None
                if v.values[0].created_at:
                    created_at_str = v.values[0].created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

                header = {
                    "version": 1,
                    "store_created_at": self.store_created_at,
                    "store_step": self.store_step,
                    "created_at": created_at_str,
                }

                header_in_bytes = json.dumps(header).encode("utf-8")
                header_size_bytes = np.asarray(len(header_in_bytes), dtype=">u4").tobytes()
                self.files[k].write(header_size_bytes)
                self.files[k].write(header_in_bytes)
                self.files[k].flush()

            if not self.store_step and not self.store_created_at:
                self._write_only_values(k, [v.value for v in v.values])
            elif self.store_step and not self.store_created_at:
                self._write_values_steps(k, v.values)
            else:
                assert trackers
                self._write_all(k, v.values, trackers)

    def _write_only_values(self, k: str, values: List[float]) -> None:
        """Append raw float32 values for series k."""
        buf = b""
        for value in values:
            buf += struct.pack(">f", value)

        self.files[k].write(buf)
        self.files[k].flush()

    def _write_values_steps(self, k: str, values: List[MetricValue]) -> None:
        """Append step (uint64) and value (float32) pairs for series k."""
        buf = b""
        for value in values:
            buf += struct.pack(">Q", value.step)  # big endian unsigned long long
            buf += struct.pack(">f", value.value)  # big endian float

        self.files[k].write(buf)
        self.files[k].flush()

    def _write_all(self, k: str, values: List[MetricValue], trackers: Dict[str, MetricsTracker]) -> None:
        """Append step, relative time (since series start), and value records for series k."""
        buf = b""
        for value in values:
            # Handle None timestamps (e.g., when store_created_at=False)
            if value.created_at is not None and trackers[k].started_at is not None:
                relative_time = value.created_at.timestamp() - trackers[k].started_at.timestamp()
            else:
                relative_time = 0.0

            step = value.step if value.step is not None else 0
            buf += struct.pack(">Q", step)  # big endian unsigned long long
            buf += struct.pack(">f", relative_time)  # big endian float
            buf += struct.pack(">f", value.value)  # big endian float

        self.files[k].write(buf)
        self.files[k].flush()

    def upload(self) -> None:
        """Close all files, archive them into a single tar.gz, and upload to remote storage."""
        # flush & close all files
        for f in self.files.values():
            f.flush()
            f.close()

        filenames = [fn for fn in os.listdir(self.log_dir) if fn.endswith(".litbin")]

        # Create and upload tar ball
        safe_version = _sanitize_version_for_path(self.version)
        file_path = os.path.join(self.log_dir, f"{safe_version}.tar.gz")
        with tarfile.open(file_path, "w:gz") as tar:
            for filename in filenames:
                tar.add(
                    os.path.join(self.log_dir, filename),
                    arcname=f"{self.teamspace_id}/{self.metrics_store_id}/{filename}",
                )
                os.remove(os.path.join(self.log_dir, filename))

        remote_path = f"/litlogger/{self.metrics_store_id}.tar.gz"
        self._artifacts_api.upload_metrics_binary(
            teamspace_id=self.teamspace_id,
            cloud_account=self.cloud_account,
            file_path=file_path,
            remote_path=remote_path,
        )
