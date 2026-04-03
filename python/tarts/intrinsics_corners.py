"""LSSTCam corner intrinsics: Z4–Z28 vs field angle (deg), nearest-neighbor lookup per band/detector.

Parquet layout (see ``intrinsics_corners.parquet``): columns ``x``, ``y`` (field angles, degrees),
``Z4``…``Z28`` (coefficients in **micrometers**, as stored), plus ``band``, ``detector``.

Used only for on-sky LSSTCam processing; simulation / training typically does not apply this.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


class IntrinsicsCornersTables:
    """Nearest-neighbor Zernike intrinsics in microns, keyed by (band_letter, detector_id)."""

    def __init__(self) -> None:
        """Create an empty container; populate via :meth:`from_parquet`."""
        self._trees: Dict[Tuple[str, int], Any] = {}
        self._z_microns: Dict[Tuple[str, int], np.ndarray] = {}
        self._n_modes = 0
        self.noll_min = 4
        self.noll_max = 28

    @property
    def loaded(self) -> bool:
        """True if at least one (band, detector) KD-tree table is present."""
        return bool(self._trees)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        noll_indices: Sequence[int] | None = None,
    ) -> IntrinsicsCornersTables | None:
        """Load parquet and build cKDTree per (band, detector).

        Z columns are interpreted as **micrometers** (float16 in file → float64 here; no m/µm
        conversion). Add directly to WaveNet outputs that are in microns after the usual ``/1000``.

        Parameters
        ----------
        path
            Path to ``intrinsics_corners.parquet``.
        noll_indices
            Noll indices the model predicts (e.g. from YAML ``noll_zk``). Intrinsics rows are
            ``Z4``…``Z28``; we select columns ``Z{n}`` for each ``n`` in this list (must be in 4…28).
        """
        if pd is None or cKDTree is None:
            logger.warning(
                "IntrinsicsCornersTables: need pandas and scipy (optional: pyarrow for parquet); "
                "skipping intrinsics load."
            )
            return None
        path = Path(path)
        if not path.is_file():
            logger.warning("IntrinsicsCornersTables: file not found: %s", path)
            return None

        try:
            master_df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("IntrinsicsCornersTables: failed to read parquet %s: %s", path, e)
            return None

        required = {"band", "detector", "x", "y"}
        if not required.issubset(master_df.columns):
            logger.warning("IntrinsicsCornersTables: missing columns (need %s)", required)
            return None

        if noll_indices is None:
            noll_list: List[int] = list(range(4, 29))
        else:
            noll_list = [int(n) for n in noll_indices]
            for n in noll_list:
                if n < 4 or n > 28:
                    logger.warning("IntrinsicsCornersTables: noll %s out of supported Z4–Z28 range", n)
                    return None

        z_cols = [f"Z{n}" for n in noll_list]
        for c in z_cols:
            if c not in master_df.columns:
                logger.warning("IntrinsicsCornersTables: missing column %s in parquet", c)
                return None

        self = cls()
        self._n_modes = len(noll_list)
        n_groups = 0
        for (band, detector), group_df in master_df.groupby(["band", "detector"]):
            clean = group_df.drop(columns=["band", "detector"], errors="ignore")
            x = clean["x"].to_numpy(dtype=np.float64, copy=True)
            y = clean["y"].to_numpy(dtype=np.float64, copy=True)
            z = np.stack([clean[c].to_numpy(dtype=np.float64, copy=True) for c in z_cols], axis=1)
            key = (str(band), int(detector))
            pts = np.column_stack([x, y])
            self._trees[key] = cKDTree(pts)
            self._z_microns[key] = z
            n_groups += 1

        logger.info(
            "IntrinsicsCornersTables: loaded %d (band, detector) tables from %s (%d Noll modes)",
            n_groups,
            path,
            len(noll_list),
        )
        return self

    def lookup_microns_batch(
        self,
        band_letter: str,
        detector_id: int,
        fx_deg: np.ndarray,
        fy_deg: np.ndarray,
    ) -> np.ndarray:
        """Return intrinsics in microns, shape (N, n_modes), nearest neighbor in (x,y).

        Parameters
        ----------
        band_letter
            Single character ``u``, ``g``, ``r``, ``i``, ``z``, or ``y``.
        detector_id
            LSST detector number (same convention as ``MAP_DETECTOR_TO_NUMBER``).
        fx_deg, fy_deg
            1-D arrays of field angles in degrees (same units as table ``x``, ``y``).
        """
        key = (band_letter.lower(), int(detector_id))
        tree = self._trees.get(key)
        z_all = self._z_microns.get(key)
        if tree is None or z_all is None:
            n = int(np.asarray(fx_deg).reshape(-1).shape[0])
            return np.zeros((n, self._n_modes), dtype=np.float64)

        q = np.column_stack([fx_deg.reshape(-1), fy_deg.reshape(-1)])
        _, idx = tree.query(q, k=1)
        return z_all[idx].astype(np.float64, copy=False)
