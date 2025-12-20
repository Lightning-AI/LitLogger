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
"""Terminal output utilities for litlogger.

This module provides styled console output for experiment lifecycle events,
making it easy to see what's happening during training and experiment tracking.
"""

import math
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import click
from rich.console import Console

# Sparkline characters for visualizing metrics history
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Lightning logo ASCII art - original full size

LIGHTNING_LOGO_FULL = """\
                    ####
                ###########
             ####################
         ############################
    #####################################
##############################################
#########################  ###################
#######################    ###################
####################      ####################
##################       #####################
################        ######################
#####################        #################
######################     ###################
#####################    #####################
####################   #######################
###################  #########################
##############################################
    #####################################
         ############################
             ####################
                  ##########
                     ####"""

LIGHTNING_LOGO_SMALL = """\
           #
       #########
   #################
############# #########
##########   ##########
########    ###########
###########    ########
##########   ##########
#########  ############
   #################
       #########
           #"""


def sparkify(series: List[float]) -> str:
    """Convert a series of numbers to a sparkline string.

    Example:
        >>> sparkify([0.5, 1.2, 3.5, 7.3, 8.0, 12.5])
        '▁▁▂▄▅█'
    """
    if not series:
        return ""

    # Filter out non-finite values
    finite_series = [x for x in series if math.isfinite(x)]
    if not finite_series:
        return ""

    minimum = min(finite_series)
    maximum = max(finite_series)
    data_range = maximum - minimum

    if data_range == 0.0:
        return SPARK_CHARS[0] * len(series)

    coefficient = (len(SPARK_CHARS) - 1.0) / data_range
    return "".join(SPARK_CHARS[int(round((x - minimum) * coefficient))] if math.isfinite(x) else " " for x in series)


def _is_unicode_safe() -> bool:
    """Check if the terminal supports unicode."""
    try:
        encoding = getattr(sys.stderr, "encoding", None) or "ascii"
        return encoding.lower() in ("utf-8", "utf8")
    except Exception:
        return False


def _rich_to_str(*renderables: Any, force_terminal: bool = False) -> str:
    """Convert Rich renderables/markup to an ANSI string."""
    import os

    with open(os.devnull, "w") as f:
        console = Console(file=f, record=True, force_terminal=force_terminal)
        console.print(*renderables)
    return console.export_text(styles=True)


@dataclass
class RunStats:
    """Statistics tracked during a run for the summary output."""

    metrics_logged: int = 0
    artifacts_logged: int = 0
    models_logged: int = 0
    # Store recent values for sparklines (metric_name -> list of values)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value for history tracking."""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        # Keep max 100 recent values per metric
        if len(self.metrics_history[name]) < 100:
            self.metrics_history[name].append(value)
        self.metrics_logged += 1


class Printer:
    """Handles styled terminal output for litlogger.

    Provides consistent formatting
    for experiment information, progress updates, and summaries.

    Example output on init:
        litlogger: Experiment initialized
                   Name:      my-experiment
                   Teamspace: my-teamspace
                   View at:   https://lightning.ai/...

    Example output on finalize:
        litlogger: Run complete
                   Metrics logged: 1,234
                   Artifacts:      5

                   Metrics history:
                         loss ▇▆▅▄▃▂▁▁
                     accuracy ▁▂▃▄▅▆▇█
    """

    # Styled prefix for all messages
    PREFIX = click.style("litlogger", fg="magenta", bold=True)
    WARN_PREFIX = click.style("WARNING", fg="yellow")
    ERROR_PREFIX = click.style("ERROR", fg="red", bold=True)

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the printer.

        Args:
            verbose: If True, print styled console output. Defaults to True.
        """
        self.verbose = verbose
        self._supports_unicode = _is_unicode_safe()
        self._supports_emoji = platform.system() != "Windows" and self._supports_unicode

    def _echo(self, message: str, prefix: bool = True, **kwargs: Any) -> None:
        """Print a message to stderr with optional prefix."""
        if not self.verbose:
            return

        if prefix:
            message = f"{self.PREFIX}: {message}"

        click.echo(message, file=sys.stderr, **kwargs)

    # =========================================================================
    # Text Styling Methods
    # =========================================================================

    def name(self, text: str) -> str:
        """Style text as a run/experiment name (yellow)."""
        return click.style(text, fg="yellow")

    def code(self, text: str) -> str:
        """Style text as code (bold)."""
        return click.style(text, bold=True)

    def files(self, text: str) -> str:
        """Style text as a file path (magenta, bold)."""
        return click.style(text, fg="magenta", bold=True)

    def link(self, url: str, text: str | None = None) -> str:
        """Style text as a clickable link (blue, underline).

        Uses Rich for OSC 8 hyperlinks when outputting to a real terminal (TTY).

        Args:
            url: The URL to link to.
            text: Optional display text. If None, shows the URL itself.
        """
        display = text or url
        if sys.stderr.isatty():
            display = _rich_to_str(f"[link={url}]{display}[/link]", force_terminal=True).rstrip("\n")
        return click.style(display, fg="blue", underline=True)

    def success(self, text: str) -> str:
        """Style text as success (green)."""
        return click.style(text, fg="green")

    def error(self, text: str) -> str:
        """Style text as error (red)."""
        return click.style(text, fg="red")

    def secondary(self, text: str) -> str:
        """Style text as secondary/dimmed (white/gray)."""
        return click.style(text, fg="white")

    def emoji(self, name: str) -> str:
        """Return an emoji by name, or empty string if not supported."""
        if not self._supports_emoji:
            return ""

        emojis = {
            "rocket": "🚀",
            "star": "⭐",
            "check": "✓",
            "cross": "✗",
            "arrow": "→",
            "box": "📦",
            "chart": "📊",
            "folder": "📁",
            "lightning": "⚡",
            "warning": "⚠️ ",
            "link": "🔗",
            "sparkles": "✨",
        }
        return emojis.get(name, "")

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log(self, message: str, prefix: bool = True) -> None:
        """Log an informational message."""
        self._echo(message, prefix=prefix)

    def warn(self, message: str) -> None:
        """Log a warning message."""
        self._echo(f"{self.WARN_PREFIX} {message}")

    def error_msg(self, message: str) -> None:
        """Log an error message."""
        self._echo(f"{self.ERROR_PREFIX} {message}")

    # =========================================================================
    # Experiment Lifecycle Output
    # =========================================================================

    def experiment_start(
        self,
        name: str,
        teamspace: str,
        url: str,
        version: str | None = None,
        metadata: Dict[str, str] | None = None,
        show_logo: bool = True,
    ) -> None:
        """Print experiment start header with key information.

        Example output:
               ##
             ######
           ##########        litlogger: Experiment initialized
          #####  #####                  Name: my-experiment
         ####    #####             Teamspace: my-teamspace
         #####    ####               View at: https://lightning.ai/...
         #####   #####
          #####  ####
           ##########
             ######
               ##
        """
        if not self.verbose:
            return

        rocket = self.emoji("rocket")
        prefix_space = " " if rocket else ""

        # Show logo with info alongside it
        if show_logo and self._supports_unicode:
            logo_lines = LIGHTNING_LOGO_SMALL.split("\n")
            # Color the logo purple (bright magenta)
            # Lightning purple: #792EE5 = RGB(121, 46, 229)
            colored_logo = [click.style(line, fg=(121, 46, 229)) for line in logo_lines]

            # Build info lines to display next to logo
            info_lines = [
                f"{self.PREFIX}: {rocket}{prefix_space}Experiment initialized",
                f"                    Name: {self.name(name)}",
                f"               Teamspace: {self.code(teamspace)}",
            ]

            if version:
                info_lines.append(f"                 Version: {self.secondary(version)}")

            if metadata:
                items = list(metadata.items())[:3]
                meta_str = ", ".join(f"{k}={v}" for k, v in items)
                if len(metadata) > 3:
                    meta_str += f" (+{len(metadata) - 3} more)"
                info_lines.append(f"                Metadata: {self.secondary(meta_str)}")

            link_emoji = self.emoji("link")
            link_prefix = f"{link_emoji} " if link_emoji else ""
            info_lines.append(f"                 View at: {link_prefix}{self.link(url)}")

            # Calculate vertical centering offset
            logo_height = len(logo_lines)
            info_height = len(info_lines)
            vertical_offset = max(0, (logo_height - info_height) // 2)

            # Print logo and info side by side, with info vertically centered
            max_lines = max(logo_height, info_height + vertical_offset)
            logo_width = max(len(line) for line in logo_lines)
            for i in range(max_lines):
                logo_part = colored_logo[i] if i < len(colored_logo) else ""
                # Pad logo to consistent width
                logo_padding = " " * (logo_width - len(logo_lines[i])) if i < len(logo_lines) else " " * logo_width

                # Get info line with vertical offset for centering
                info_idx = i - vertical_offset
                info_part = info_lines[info_idx] if 0 <= info_idx < len(info_lines) else ""
                self._echo(f"{logo_part}{logo_padding}    {info_part}", prefix=False)

            click.echo("", file=sys.stderr)  # Blank line without prefix
        else:
            # Fallback to simple output without logo
            self._echo(f"{rocket}{prefix_space}Experiment initialized")
            self._echo(f"           Name: {self.name(name)}", prefix=False)
            self._echo(f"      Teamspace: {self.code(teamspace)}", prefix=False)

            if version:
                self._echo(f"        Version: {self.secondary(version)}", prefix=False)

            if metadata:
                items = list(metadata.items())[:3]
                meta_str = ", ".join(f"{k}={v}" for k, v in items)
                if len(metadata) > 3:
                    meta_str += f" (+{len(metadata) - 3} more)"
                self._echo(f"       Metadata: {self.secondary(meta_str)}", prefix=False)

            link_emoji = self.emoji("link")
            link_prefix = f"{link_emoji} " if link_emoji else ""
            self._echo(f"        View at: {link_prefix}{self.link(url)}", prefix=False)
            click.echo("", file=sys.stderr)  # Blank line without prefix

    def experiment_complete(
        self,
        name: str,
        stats: RunStats | None = None,
        url: str | None = None,
    ) -> None:
        """Print experiment completion with summary statistics.

        Example output:
            litlogger: ✨ Run complete!
                       Metrics logged: 1,234
                       Artifacts:      5

                       Metrics history:
                             loss ▇▆▅▄▃▂▁▁
                         accuracy ▁▂▃▄▅▆▇█
        """
        if not self.verbose:
            return

        sparkles = self.emoji("sparkles")
        prefix_space = " " if sparkles else ""

        self._echo(f"{sparkles}{prefix_space}Run {self.success('complete')}")

        if stats:
            self._print_run_stats(stats)

        if url:
            link_emoji = self.emoji("link")
            link_prefix = f"{link_emoji} " if link_emoji else ""
            self._echo(f"        View at: {link_prefix}{self.link(url)}", prefix=False)

        click.echo("", file=sys.stderr)  # Blank line without prefix

    def _print_run_stats(self, stats: RunStats) -> None:
        """Print the run summary statistics."""
        # Build summary rows with right-aligned labels
        rows = []
        if stats.metrics_logged > 0:
            rows.append(("Metrics logged", f"{stats.metrics_logged:,}"))
        if stats.artifacts_logged > 0:
            rows.append(("Artifacts", f"{stats.artifacts_logged:,}"))
        if stats.models_logged > 0:
            rows.append(("Models", f"{stats.models_logged:,}"))

        if rows:
            max_label_len = max(len(label) for label, _ in rows)
            for label, value in rows:
                self._echo(
                    f"   {label:>{max_label_len}}: {value}",
                    prefix=False,
                )

        # Print metrics history with sparklines
        if stats.metrics_history and self._supports_unicode:
            click.echo("", file=sys.stderr)  # Blank line
            self._echo("   Metrics history:", prefix=False)

            # Get max metric name length for alignment
            max_name_len = max(len(name) for name in stats.metrics_history)
            max_name_len = min(max_name_len, 20)  # Cap at 20 chars

            # Show up to 5 metrics
            for metric_name, values in list(stats.metrics_history.items())[:5]:
                # Truncate metric name if too long
                display_name = metric_name[:20] if len(metric_name) > 20 else metric_name

                # Downsample to ~40 points if needed
                if len(values) > 40:
                    step = len(values) / 40
                    sampled = [values[int(i * step)] for i in range(40)]
                else:
                    sampled = values

                sparkline = sparkify(sampled)
                self._echo(
                    f"     {display_name:>{max_name_len}} {sparkline}",
                    prefix=False,
                )

            if len(stats.metrics_history) > 5:
                remaining = len(stats.metrics_history) - 5
                self._echo(
                    f"     {self.secondary(f'... and {remaining} more metrics')}",
                    prefix=False,
                )

    def metric_logged(self, name: str, value: float, step: int | None = None) -> None:
        """Print a single metric log event (used in verbose mode).

        Example:
            litlogger: → loss: 0.0123 (step 100)
        """
        if not self.verbose:
            return

        arrow = self.emoji("arrow") or "→"
        step_str = f" (step {step})" if step is not None else ""
        self._echo(f"{arrow} {name}: {value:.4g}{step_str}")

    def artifact_logged(self, path: str, remote_path: str | None = None) -> None:
        """Print artifact upload confirmation.

        Example:
            litlogger: ✓ Logged model.pt
        """
        if not self.verbose:
            return

        check = self.emoji("check") or "✓"
        display_path = remote_path or path
        self._echo(f"{self.success(check)} Logged {self.files(display_path)}")

    def artifact_retrieved(self, path: str) -> None:
        """Print artifact download confirmation.

        Example:
            litlogger: ✓ Retrieved model.pt
        """
        if not self.verbose:
            return

        check = self.emoji("check") or "✓"
        self._echo(f"{self.success(check)} Retrieved {self.files(path)}")

    def artifact_failed(self, path: str, error: str) -> None:
        """Print artifact upload/download failure.

        Example:
            litlogger: ✗ Failed to upload model.pt: Connection error
        """
        if not self.verbose:
            return

        cross = self.emoji("cross") or "✗"
        self._echo(f"{self.error(cross)} Failed {self.files(path)}: {error}")

    # =========================================================================
    # Generic Success/Failure Methods
    # =========================================================================

    def print_success(self, message: str, detail: str | None = None) -> None:
        """Print a success message.

        Args:
            message: The main success message.
            detail: Optional additional detail (shown in secondary style).

        Example:
            litlogger: ✓ Uploaded metrics batch
            litlogger: ✓ Connected to server (latency: 50ms)
        """
        if not self.verbose:
            return

        check = self.emoji("check") or "✓"
        msg = f"{self.success(check)} {message}"
        if detail:
            msg += f" {self.secondary(f'({detail})')}"
        self._echo(msg)

    def print_failure(self, message: str, error: str | None = None) -> None:
        """Print a failure message.

        Args:
            message: The main failure message.
            error: Optional error detail.

        Example:
            litlogger: ✗ Failed to upload metrics
            litlogger: ✗ Connection failed: timeout
        """
        if not self.verbose:
            return

        cross = self.emoji("cross") or "✗"
        msg = f"{self.error(cross)} {message}"
        if error:
            msg += f": {error}"
        self._echo(msg)

    def print_progress(self, message: str, current: int | None = None, total: int | None = None) -> None:
        """Print a progress message.

        Args:
            message: The progress message.
            current: Current progress count.
            total: Total count.

        Example:
            litlogger: → Uploading metrics...
            litlogger: → Uploading files (5/10)...
        """
        if not self.verbose:
            return

        arrow = self.emoji("arrow") or "→"
        if current is not None and total is not None:
            msg = f"{arrow} {message} ({current}/{total})..."
        elif current is not None:
            msg = f"{arrow} {message} ({current})..."
        else:
            msg = f"{arrow} {message}..."
        self._echo(msg)

    def print_info(self, message: str, detail: str | None = None) -> None:
        """Print an informational message.

        Args:
            message: The info message.
            detail: Optional additional detail.

        Example:
            litlogger: i Using default teamspace
            litlogger: i Reconnecting (attempt 2)
        """
        if not self.verbose:
            return

        info = "ℹ" if self._supports_unicode else "i"  # noqa: RUF001
        msg = f"{info} {message}"
        if detail:
            msg += f" {self.secondary(f'({detail})')}"
        self._echo(msg)
