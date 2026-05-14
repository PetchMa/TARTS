"""Process-wide defaults applied before timm / huggingface_hub are imported.

Summit and USDF often run with no internet and with ``HOME`` pointing at paths
that are not writable or do not exist. timm backbones with ``pretrained=True``
download via the Hugging Face Hub into ``HF_HOME`` (or defaults derived from
``HOME``). Setting ``HF_HOME`` (or this package's alias) early avoids failed
``makedirs`` under e.g. ``/home/saluser/.cache/...`` on RA hosts.
"""

from __future__ import annotations

import os
from pathlib import Path


def apply_tarts_runtime_env() -> None:
    """Apply environment tweaks that must run before Hub-backed models load.

    ``TARTS_HF_HOME``
        If set and ``HF_HOME`` is unset, ``HF_HOME`` is set to this path
        (``~`` expanded). Operators may instead set ``HF_HOME`` directly; see
        Hugging Face Hub documentation.
    """
    tarts_hf = os.environ.get("TARTS_HF_HOME")
    if tarts_hf and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.fspath(Path(tarts_hf).expanduser())


apply_tarts_runtime_env()
