"""Download the FMA (Free Music Archive) dataset to Google Drive.

Handles downloading the FMA-large subset (~93 GB, 106,574 tracks) and
verifying download integrity. Designed to run once in Colab with Drive
mounted, so subsequent notebooks can skip this step.
"""

from pathlib import Path
from typing import Optional


def download_fma_large(drive_path: str) -> None:
    """Download the FMA-large subset to the specified Google Drive path.

    Args:
        drive_path: Destination directory on mounted Google Drive
                    (e.g. '/content/drive/MyDrive/fma').
    """
    ...


def verify_download(drive_path: str) -> bool:
    """Verify the integrity of a previously downloaded FMA dataset.

    Args:
        drive_path: Directory containing the downloaded FMA files.

    Returns:
        True if all expected files are present and checksums match.
    """
    ...
