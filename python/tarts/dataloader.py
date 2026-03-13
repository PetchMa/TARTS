"""Pytorch DataSet for the AOS simulations."""

# Standard library imports
import glob
import hashlib
import logging
import os
import pickle
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        """Fallback when tqdm is not installed: return iterable unchanged."""
        return iterable


# Local/application imports
from .constants import DEFAULT_NOLL_ZK, DEFAULT_TRAIN_FRACTION
from .utils import shift_offcenter, transform_inputs

logger = logging.getLogger(__name__)


def _plot_seqnum_histogram(selected_seqnums: np.ndarray, save_path: str) -> None:
    """Plot histogram of SEQNUM counts in the selected training set.

    Used to verify uniform coverage across perturbations (one bin per SEQNUM).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping SEQNUM histogram.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    # Exclude sentinel -1 if present
    vals = selected_seqnums[selected_seqnums >= 0]
    if vals.size == 0:
        logger.warning("No valid SEQNUMs for histogram.")
        return
    ax.hist(vals, bins=min(100, max(50, int(np.ptp(vals)) + 1)), color="steelblue", edgecolor="white")
    ax.set_xlabel("SEQNUM")
    ax.set_ylabel("Sample count")
    ax.set_title("SEQNUM distribution in training set (should be roughly uniform per bin)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"SEQNUM histogram saved to {save_path}")


class Donuts(Dataset):
    """PRETRAINING DATASET Batoid Sims.

    A PyTorch Dataset class for loading AOS snippets of donuts
    and corresponding Zernike coefficients and offcenter
    augmented shifts from simulations.

    This class loads simulated donut images and associated
    metadata from a specified directory. The dataset
    can be used for training, validation, or testing, and
    allows for optional transformations to be applied
    to the inputs.

    Parameters
    ----------
    mode : str, optional, default="train"
        Specifies which subset of the data to load.
        Options include "train", "val" (validation), or "test".
    transform : bool, optional, default=True
        Whether to apply transformations (e.g., normalization) to the inputs.
    adjustment_factor : float, optional, default=0
        RADIAL factor used to shift the image
        (e.g., shifting the donut image) during loading.
    data_dir : str, optional, default="aos_sims"
        Path to the directory containing the simulated
        data (images, Zernikes, etc.).
    kwargs : Any, optional
        Additional keyword arguments for customization.

    Attributes
    ----------
    settings : dict
        A dictionary storing the configuration options
        (mode, transform, data_dir).
    observations : Table
        Table containing metadata for each observation.
    obs_ids : dict
        Dictionary containing the indices for the train, validation, and test splits.
    image_files : dict
        Dictionary of lists containing the file paths to the
        images for each mode (train, val, test).
    adjustment_factor : float
        Adjustment factor for shifting the donut images.
    """

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        adjustment_factor=0,
        data_dir: str = "aos_sims",
        **kwargs: Any,
    ) -> None:
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transform_inputs from ml_aos.utils.
        adjustment_factor: float, default=0
            RADIAL factor used to shift the image during loading.
        data_dir: str, default=aos_sims
            Location of the data directory.
        """
        # save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
            "data_dir": data_dir,
        }

        # get a list of all the observations
        all_image_files = glob.glob(f"{data_dir}/images/*")
        obs_ids = list(set([int(file.split("/")[-1].split(".")[1][3:]) for file in all_image_files]))

        # get the table of metadata for each observation
        observations = Table.read(f"{data_dir}/opSimTable.parquet")
        observations = observations[obs_ids]
        self.observations = observations

        # now split the observations between train, test, val
        train_ids = []
        val_ids = []
        test_ids = []

        # we don't have enough u band, so let's put 2 in test and rest in train
        group = observations[observations["lsstFilter"] == "u"]
        test_ids.extend(group["observationId"][:2])
        train_ids.extend(group["observationId"][2:])

        # for the rest of the bands, let's put 2 each in test/val, and rest in train
        for band in "grizy":
            group = observations[observations["lsstFilter"] == band]
            test_ids.extend(group["observationId"][:2])
            val_ids.extend(group["observationId"][2:4])
            train_ids.extend(group["observationId"][4:])

        self.obs_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        self.adjustment_factor = adjustment_factor
        # partition the image files
        self.image_files = {
            mode: [file for file in all_image_files if int(file.split("/")[-1].split(".")[1][3:]) in ids]
            for mode, ids in self.obs_ids.items()
        }

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.image_files[self.settings["mode"]])  # type: ignore

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors
                image: donut image, shape=(256, 256)
                offset: radial offset amount in units of pixels
                offset_vec: vector offset amount in units of pixels
                field_x, field_y: the field angle in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                band: LSST band indicated by index in string "ugrizy" (e.g. 2 = "r")
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                dof: the telescope perturbations corresponding to the zernikes
                pntId: the pointing ID
                obsID: the observation ID
                objID: the object ID
        """
        # get the image file
        img_file = self.image_files[self.settings["mode"]][idx]  # type: ignore

        # load the image
        img = np.load(img_file, allow_pickle=True)

        # crop out the central 160x160
        img = img[5:-5, 5:-5]

        # get the IDs
        pntId, obsId, objId = img_file.split("/")[-1].split(".")[:3]

        # get the catalog for this observation
        catalog = Table.read(f"{self.settings['data_dir']}/catalogs/{pntId}.catalog.parquet")

        # get the row for this source
        row = catalog[catalog["objectId"] == int(objId[3:])][0]

        # get the donut locations
        fx, fy = row["xField"], row["yField"]

        # get the intra/extra flag
        intra = "SW1" in row["detector"]

        # get the observed band
        obs_row = self.observations[self.observations["observationId"] == int(obsId[3:])]
        band = "ugrizy".index(obs_row["lsstFilter"].item())

        # load the zernikes
        zernikes = np.load(
            (
                f"{self.settings['data_dir']}/zernikes/"
                f"{pntId}.{obsId}.detector{row['detector'][:3]}.zernikes.npy"
            ),
            allow_pickle=True,
        )

        # load the degrees of freedom
        dof = np.load(f"{self.settings['data_dir']}/dof/{pntId}.dofs.npy", allow_pickle=True)

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            img, fx, fy, intra, band = transform_inputs(  # type: ignore
                img,
                fx,
                fy,
                intra,
                band,
            )

        # convert everything to tensors
        img = torch.from_numpy(img).float()
        # shift the image
        img_adjusted, offset_amount = shift_offcenter(img, adjust=self.adjustment_factor, return_offset=True)
        # track the offset vector and renormalise the vector amount
        offset_vec = np.array(np.array(offset_amount).astype(np.float32)) / self.adjustment_factor
        # compute the radial offset factor (vector norm)
        offset_r = np.sqrt(offset_amount[0] ** 2 + offset_amount[1] ** 2)
        offset_r = np.array(offset_r.astype(np.float32))[None] / self.adjustment_factor

        # record the meta data
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        band = torch.FloatTensor([band])
        zernikes = torch.from_numpy(zernikes).float()
        dof = torch.from_numpy(dof).float()

        output = {
            "image": img_adjusted,
            "offset": offset_amount,
            "offset_vec": offset_vec,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
            "zernikes": zernikes,
            "dof": dof,
            "pntId": int(pntId[3:]),
            "obsId": int(obsId[3:]),
            "objId": int(objId[3:]),
        }
        return output


class Donuts_Fullframe(Dataset):
    """FINETUNE DATASET realistic ImSim after running the generate_finetune_data.

    Full frame images. (REQUIRES LSST SCI PIPELINE!)
    The training and validation split is by splitting the list
    of the directories by 80% and 20% of the list

    Parameters
    ----------
    mode : str, optional, default="train"
        Specifies which subset of the data to load.
        Options include "train", "val" (validation), or "test".
    transform : bool, optional, default=True
        Whether to apply transformations (e.g., normalization) to the inputs.
    data_dir : str, optional, default="aos_sims"
        Path to the directory containing the simulated
        data (images, Zernikes, etc.).
    kwargs : Any, optional
        Additional keyword arguments for customization.

    Attributes
    ----------
    settings : dict
        A dictionary storing the configuration options
        (mode, transform, data_dir).
    observations : Table
        Table containing metadata for each observation.
    obs_ids : dict
        Dictionary containing the indices for the train, validation, and test splits.
    image_files : dict
        Dictionary of lists containing the file paths to the
        images for each mode (train, val, test).

    """

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        adjustment_factor=0,
        data_dir: str = "/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/simulation_pretrain/",
        noll_zk: Optional[List[int]] = None,
        coral_filepath: str = "/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/coral/",
        coral_mode: bool = False,
        mask_mode: bool = False,
        rotate: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Load the simulated ImSim donuts and zernikes in a Pytorch Dataset.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transform_inputs from ml_aos.utils.
        adjustment_factor: float, default=0
            RADIAL factor used to shift the image during loading.
        data_dir: str, default=aos_sims
            Location of the data directory.
        noll_zk: Optional[List[int]], default=None
            List of Noll Zernike indices to include in the dataset.
            If None, defaults to [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28].
        coral_filepath: str, default="/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/coral/"
            Path to the coral dataset directory.
        coral_mode: bool, default=False
            Whether to enable coral mode for domain adaptation.
        mask_mode: bool, default=False
            Whether to use mask mode for zernike extraction.
        rotate: bool, default=False
            Whether to use rotated Zernike targets from the dataset when available.
        train_fraction: float, optional
            Fraction of stamps per SEQNUM (0.0-1.0). Always uses full coverage of all
            unique SEQNUMs (perturbations); this controls how many stamps per SEQNUM.
            For train, if None, uses DEFAULT_TRAIN_FRACTION. For val, if None, uses full data.
        min_seqnums: int, default=10001
            Unused; kept for API compatibility. Full coverage uses all unique SEQNUMs
            found in the directory (number is logged).
        rotation_direction: Optional[str], default=None
            Required when rotate=True. Must be \"pos\" or \"neg\" to select
            `zk_true_camera_pos` or `zk_true_camera_neg` from the dataset.
        """
        train_fraction = kwargs.pop("train_fraction", None)
        rotation_direction = kwargs.pop("rotation_direction", None)
        _ = kwargs.pop("min_seqnums", 10001)  # Unused; kept for API compatibility
        plot_seqnum_hist = kwargs.pop("plot_seqnum_hist", None)
        # Optional: disable SEQNUM-balanced sampling and just use all images.
        seqnum_balanced = kwargs.pop("seqnum_balanced", True)
        # Ignore deprecated augmentation params for backward compatibility
        kwargs.pop("augment", None)
        kwargs.pop("augment_scale", None)
        kwargs.pop("augment_kmin", None)
        self.settings = {
            "mode": mode,
            "transform": transform,
            "data_dir": data_dir,
        }
        if self.settings["mode"] == "train":
            self.image_dir = data_dir + "/train"
        if self.settings["mode"] == "val":
            self.image_dir = data_dir + "/val"
        self.mask_mode = mask_mode
        self.image_files = []
        # Loop through all files and subdirectories
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.image_files.append(file_path)
        logger.debug(f"Image directory: {self.image_dir}")
        self.coral_filepath = coral_filepath
        self.coral_mode = coral_mode
        self.rotate = rotate
        # Enforce explicit rotation direction when rotate=True to avoid silent mismatches
        if self.rotate and rotation_direction not in ("pos", "neg"):
            raise ValueError(
                f"rotation_direction must be 'pos' or 'neg' when rotate=True "
                f"(got {rotation_direction!r})."
            )
        self.rotation_direction = rotation_direction

        if coral_mode:
            self.coral_image_files = []
            coral_data_path = coral_filepath
            if self.settings["mode"] == "train":
                coral_data_path += "/train"
            if self.settings["mode"] == "val":
                coral_data_path += "/val"
            for root, _, files in os.walk(coral_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.coral_image_files.append(file_path)

        self.used_indices = list(range(len(self.image_files)))
        if not seqnum_balanced:
            # Unbalanced path: optional subsample by train_fraction (no SEQNUM balancing).
            frac = train_fraction if train_fraction is not None else DEFAULT_TRAIN_FRACTION
            n_total = len(self.image_files)
            n_keep = max(1, int(frac * n_total))
            if n_keep < n_total:
                chosen = np.random.choice(n_total, size=n_keep, replace=False)
                self.used_indices = np.random.permutation(chosen).tolist()
                logger.info(
                    f"Subsampled to {len(self.used_indices)} / {n_total} samples (train_fraction={frac})."
                )
        else:
            # SEQNUM-based balanced sampling: always full coverage of all perturbations.
            # train_fraction controls stamps per SEQNUM (at least 1 per SEQNUM).
            frac = train_fraction if train_fraction is not None else DEFAULT_TRAIN_FRACTION
            # Store SEQNUM cache outside the data directory so it is never read as data.
            _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tarts", "seqnum")
            os.makedirs(_cache_dir, exist_ok=True)
            _path_hash = hashlib.sha256(os.path.abspath(self.image_dir).encode()).hexdigest()[:16]
            cache_path = os.path.join(_cache_dir, f"{_path_hash}.pkl")
            seqnum_per_index: Optional[List[int]] = None
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cache = pickle.load(f)
                    path_to_seqnum = cache.get("path_to_seqnum")
                    if path_to_seqnum is not None and set(path_to_seqnum.keys()) == set(self.image_files):
                        seqnum_per_index = [path_to_seqnum[fp] for fp in self.image_files]
                        logger.info(
                            f"Loaded SEQNUM cache from {cache_path} ({len(seqnum_per_index)} entries)."
                        )
                    elif cache.get("image_files") == self.image_files:
                        # Legacy format: same order required
                        seqnum_per_index = cache.get("seqnum_per_index")
                        if seqnum_per_index is not None:
                            logger.info(
                                f"Loaded SEQNUM cache from {cache_path} ({len(seqnum_per_index)} entries)."
                            )
                except (pickle.PickleError, OSError) as e:
                    logger.warning(f"Could not load SEQNUM cache: {e}")
            if seqnum_per_index is None:
                seqnum_per_index = []
                for i, fp in enumerate(tqdm(self.image_files, desc="Building SEQNUM index")):
                    try:
                        data = np.load(fp, allow_pickle=True)
                        try:
                            seqnum = int(data["SEQNUM"].item()) if "SEQNUM" in data else -1
                        finally:
                            if hasattr(data, "close"):
                                data.close()
                    except (KeyError, ValueError, OSError) as e:
                        logger.warning(f"Failed to load SEQNUM from {fp}: {e}")
                        seqnum = -1
                    seqnum_per_index.append(seqnum)
                try:
                    path_to_seqnum = {fp: seqnum_per_index[i] for i, fp in enumerate(self.image_files)}
                    with open(cache_path, "wb") as f:
                        pickle.dump({"path_to_seqnum": path_to_seqnum}, f)
                    logger.info(f"Saved SEQNUM cache to {cache_path}.")
                except OSError as e:
                    logger.warning(f"Could not save SEQNUM cache: {e}")
            seqnum_to_indices: Dict[int, List[int]] = {}
            for i, seqnum in enumerate(seqnum_per_index):
                seqnum_to_indices.setdefault(seqnum, []).append(i)
            num_unique = len(seqnum_to_indices) - (1 if -1 in seqnum_to_indices else 0)
            logger.info(f"Found {num_unique} unique SEQNUMs in dataset.")
            selected: List[Tuple[int, int]] = []  # (file_index, seqnum)
            for seqnum, indices in seqnum_to_indices.items():
                n_take = max(1, min(len(indices), int(frac * len(indices))))
                chosen = np.random.choice(len(indices), size=n_take, replace=False)
                for j in chosen:
                    selected.append((indices[int(j)], seqnum))
            np.random.shuffle(selected)
            self.used_indices = [s[0] for s in selected]
            selected_seqnums = np.array([s[1] for s in selected])
            if len(self.used_indices) < num_unique:
                logger.warning(
                    f"Selected {len(self.used_indices)} samples < {num_unique} unique SEQNUMs; "
                    "some perturbation bins may be unrepresented."
                )
            # Optional: plot SEQNUM distribution to verify uniform coverage
            if plot_seqnum_hist and self.settings["mode"] == "train":
                _plot_seqnum_histogram(
                    selected_seqnums,
                    plot_seqnum_hist if isinstance(plot_seqnum_hist, str) else "seqnum_histogram.png",
                )

        if noll_zk is None:
            noll_zk = DEFAULT_NOLL_ZK
        self.noll_zk = np.array(noll_zk) - 4
        self.adjustment_factor = adjustment_factor
        self.rotate = rotate

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.used_indices)

    def sample_coral(self) -> Dict[str, Any]:
        """Sample a coral image randomly."""
        # Check if coral files are available
        if not self.coral_image_files or len(self.coral_image_files) == 0:
            raise RuntimeError("No coral image files available for sampling.")

        # Randomly sample from coral image files with retry for corrupted files
        max_retries = 10
        corrupted_files = []  # Track files to remove

        for attempt in range(max_retries):
            try:
                # Re-check availability in case files were deleted
                if not self.coral_image_files:
                    raise RuntimeError("All coral image files have been removed due to corruption.")

                idx = np.random.randint(0, len(self.coral_image_files))
                img_file = self.coral_image_files[idx]

                # Skip already identified corrupted files
                if img_file in corrupted_files:
                    continue

                # Load coral file (avoid 'with' for NumPy 2.x / dict return)
                loaded_state = np.load(img_file, allow_pickle=True)
                try:
                    state = {key: loaded_state[key].copy() for key in loaded_state.keys()}
                finally:
                    if hasattr(loaded_state, "close"):
                        loaded_state.close()
                break  # Successfully loaded, exit retry loop
            except (EOFError, IOError, OSError) as e:
                # File is corrupted, truncated, or missing - delete it
                logger.warning(f"Corrupted coral file detected: {img_file}. Error: {e}. Deleting...")
                try:
                    if os.path.exists(img_file):
                        os.remove(img_file)
                        logger.info(f"Deleted corrupted file: {img_file}")
                except (OSError, PermissionError) as delete_error:
                    logger.warning(f"Failed to delete {img_file}: {delete_error}")

                # Remove from list to avoid trying again
                if img_file in self.coral_image_files:
                    self.coral_image_files.remove(img_file)
                corrupted_files.append(img_file)

                if attempt == max_retries - 1:
                    # Last attempt failed, raise the error
                    raise RuntimeError(
                        f"Failed to load coral file after {max_retries} attempts. "
                        f"Last error: {e}. All coral files may be corrupted."
                    )
                # Try another random file
                continue

        # get the donut locations
        fx, fy = (
            torch.tensor(state["field_x"]) * np.pi / 180,
            torch.tensor(state["field_y"]) * np.pi / 180,
        )

        # get the intra/extra flag
        intra = torch.tensor(state["intra"]).int()

        band_tensor = torch.tensor(state["band"]).int()
        band = band_tensor.item()
        img = torch.tensor(state["image_aligned"])

        # Get zernikes using noll_zk indexing
        # Convert to numpy first to handle object dtype from npz files
        zernikes = torch.zeros((1, len(self.noll_zk)))

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            # Convert tensors to numpy for transform_inputs
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
            fx_val = float(fx.item() if isinstance(fx, torch.Tensor) else fx)
            fy_val = float(fy.item() if isinstance(fy, torch.Tensor) else fy)
            intra_val = bool(intra.item() if isinstance(intra, torch.Tensor) else intra)
            band_val = int(band_tensor.item() if isinstance(band_tensor, torch.Tensor) else band)
            img_out, fx_out, fy_out, intra_out, band_out = transform_inputs(
                img_np,
                fx_val,
                fy_val,
                intra_val,
                band_val,
            )
            # convert everything to tensors
            img = torch.from_numpy(img_out).float() if isinstance(img_out, np.ndarray) else img_out.float()
            # get meta data
            fx_tensor = torch.FloatTensor([float(fx_out)])
            fy_tensor = torch.FloatTensor([float(fy_out)])
            intra_tensor = torch.FloatTensor([float(intra_out)])
            band_tensor = torch.FloatTensor([float(band_out)])
        else:
            # convert everything to tensors if not transformed
            img = img.float()
            fx_tensor = torch.FloatTensor([float(fx.item() if isinstance(fx, torch.Tensor) else fx)])
            fy_tensor = torch.FloatTensor([float(fy.item() if isinstance(fy, torch.Tensor) else fy)])
            intra_tensor = torch.FloatTensor(
                [float(intra.item() if isinstance(intra, torch.Tensor) else intra)]
            )
            band_tensor = torch.FloatTensor([float(band)])
        zernikes = zernikes.float()[0, :]
        coral_output = {
            "coral_image": img,
            "coral_field_x": fx_tensor,
            "coral_field_y": fy_tensor,
            "coral_intrafocal": intra_tensor,
            "coral_band": band_tensor,
            "coral_zernikes": zernikes,
        }
        return coral_output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors
                image: donut image, shape=(256, 256)
                field_x, field_y: the field angle in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                band: LSST band indicated by index in string "ugrizy" (e.g. 2 = "r")
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                dof: the telescope perturbations corresponding to the zernikes
                pntId: the pointing ID
                obsID: the observation ID
                objID: the object ID
        """
        # get the image file (use index mapping for SEQNUM-balanced sampling)
        img_file = self.image_files[self.used_indices[idx]]

        # Load npz; avoid 'with' so we support np.load returning dict (e.g. NumPy 2.x / .npy)
        state = np.load(img_file, allow_pickle=True)
        try:
            # get the donut locations
            fx, fy = (
                torch.tensor(state["field_x"]) * np.pi / 180,
                torch.tensor(state["field_y"]) * np.pi / 180,
            )

            # get the intra/extra flag
            intra = torch.tensor(state["intra"]).int()

            # Choose which Zernike target to use based on rotation settings
            zk_true_key = None
            is_rotated = False
            if self.rotate:
                # Prefer new explicit positive/negative rotation fields when available
                if self.rotation_direction == "pos" and "zk_true_camera_pos" in state:
                    zk_true_key = "zk_true_camera_pos"
                    is_rotated = True
                elif self.rotation_direction == "neg" and "zk_true_camera_neg" in state:
                    zk_true_key = "zk_true_camera_neg"
                    is_rotated = True
                # Legacy single-rotation field
                elif "zk_true_camera" in state:
                    zk_true_key = "zk_true_camera"
                    is_rotated = True
                # Fall back to unrotated if nothing else is present
                elif "zk_true" in state:
                    zk_true_key = "zk_true"
                    is_rotated = False
                else:
                    raise KeyError(
                        f"No suitable rotated Zernikes found in npz file when rotate=True. File: {img_file}"
                    )
            else:
                if "zk_true" in state:
                    zk_true_key = "zk_true"
                    is_rotated = False
                else:
                    raise KeyError(f"'zk_true' not found in npz file. File: {img_file}")

            if self.mask_mode:
                zernikes = torch.tensor(state[zk_true_key])
                # zk_true_camera has 1 less dimension, so no need to index [:, 0]
                if not is_rotated:
                    zernikes = zernikes[:, 0]
                zernikes = zernikes[None, :]
            else:
                zernikes = torch.tensor(state[zk_true_key])
                # zk_true_camera has 1 less dimension, so indexing is different
                if is_rotated:
                    zernikes = zernikes[self.noll_zk]
                else:
                    zernikes = zernikes[:, self.noll_zk]

            band_tensor = torch.tensor(state["band"]).int()
            band = band_tensor.item()
            img = torch.tensor(state["image_aligned"])
        finally:
            if hasattr(state, "close"):
                state.close()

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            # Convert tensors to numpy for transform_inputs
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
            fx_val = float(fx.item() if isinstance(fx, torch.Tensor) else fx)
            fy_val = float(fy.item() if isinstance(fy, torch.Tensor) else fy)
            intra_val = bool(intra.item() if isinstance(intra, torch.Tensor) else intra)
            band_val = int(band_tensor.item() if isinstance(band_tensor, torch.Tensor) else band)
            img_out, fx_out, fy_out, intra_out, band_out = transform_inputs(
                img_np,
                fx_val,
                fy_val,
                intra_val,
                band_val,
            )
            # convert everything to tensors
            img = torch.from_numpy(img_out).float() if isinstance(img_out, np.ndarray) else img_out.float()
            # get meta data
            fx_tensor = torch.FloatTensor([float(fx_out)])
            fy_tensor = torch.FloatTensor([float(fy_out)])
            intra_tensor = torch.FloatTensor([float(intra_out)])
            band_tensor = torch.FloatTensor([float(band_out)])
        else:
            # convert everything to tensors if not transformed
            img = img.float()
            fx_tensor = torch.FloatTensor([float(fx.item() if isinstance(fx, torch.Tensor) else fx)])
            fy_tensor = torch.FloatTensor([float(fy.item() if isinstance(fy, torch.Tensor) else fy)])
            intra_tensor = torch.FloatTensor(
                [float(intra.item() if isinstance(intra, torch.Tensor) else intra)]
            )
            band_tensor = torch.FloatTensor([float(band)])

        # Apply image shifting only in train mode and when adjustment_factor > 0
        if self.settings["mode"] == "train" and self.adjustment_factor > 0:
            img, offset_amount = shift_offcenter(img, adjust=self.adjustment_factor, return_offset=True)
            # Add offset information to output
            offset_vec = np.array(np.array(offset_amount).astype(np.float32)) / self.adjustment_factor
            offset_r = np.sqrt(offset_amount[0] ** 2 + offset_amount[1] ** 2)
            offset_r = np.array(offset_r.astype(np.float32)) / self.adjustment_factor
        else:
            offset_amount = [0, 0]
            offset_vec = np.array([0.0, 0.0])
            offset_r = np.array(0.0)
        # zk_true_camera is already 1D, so no need to index [0, :]
        if zernikes.dim() > 1:
            zernikes = zernikes.float()[0, :]
        else:
            zernikes = zernikes.float()

        output = {
            "image": img,
            "field_x": fx_tensor,
            "field_y": fy_tensor,
            "intrafocal": intra_tensor,
            "band": band_tensor,
            "zernikes": zernikes,
            "offset": offset_amount,
            "offset_vec": offset_vec,
            "offset_r": offset_r,
        }
        if self.coral_mode:
            coral_output = self.sample_coral()
            output.update(coral_output)

        return output


class zernikeDataset(Dataset):
    """AGGREGATORNET DATASET.

    A PyTorch Dataset for training AGGREGATORNET
    loading and processing multiple Zernike coefficient data for LSST simulations.


    Parameters
    ----------
    seq_length : int
        The maximum sequence length for each sample. Sequences longer than
        this length will be truncated, and shorter sequences will be padded with zeros.
    train : bool, optional, default=True
        Whether to load the training dataset (`True`) or the
        testing dataset (`False`). The training set
        corresponds to the first 80% of the data, while the
        test set corresponds to the remaining 20%.
    data_dir : str, optional, default='.../LSST_FULL_FRAME/aggregator/'
        The root directory containing the dataset files.
        The files should be structured in subdirectories
        under this directory.
    alpha : float, optional, default=1e-3
        A parameter used for adjusting Zernike coefficients during processing.
    return_true : bool, optional, default=False
        Whether to return the true Zernike coefficients (`True`)
        or the estimated coefficients (`False`).
    coral_mode : bool, optional, default=False
        Whether to enable coral mode for sampling real data alongside simulations.
    coral_filepath : str, optional, default='.../LSST_FULL_FRAME/aggregator_real/'
        Path to the directory containing real/coral aggregator data files.

    Attributes
    ----------
    max_seq_length : int
        The maximum sequence length for each sample, as specified during initialization.
    filename : list
        A list of file paths to the dataset files.
        Files are loaded recursively from the specified data directory.
    num_samples : int
        The total number of samples in the dataset (based on the mode: train/test).
    alpha : float
        The alpha parameter used for Zernike coefficient adjustments.
    return_true : bool
        Whether to return the true Zernike coefficients or the estimated ones.
    device : torch.device
        The device (CUDA or CPU) where tensors will be allocated for processing.

    Methods
    -------
    __len__ : int
        Returns the total number of samples in the dataset.
    __getitem__ : tuple
        Loads and processes a sample from the dataset at the given index. Returns a tuple of:
        - x_total (torch.Tensor) : Input data tensor with
            Zernike coefficients, field positions, and SNR data.
        - mean (torch.Tensor) : Mean Zernike coefficient.
        - y (torch.Tensor) : True Zernike coefficients (if `return_true` is `True`).
    """

    def __init__(
        self,
        seq_length,
        train=True,
        data_dir="/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/aggregator/",
        alpha=1e-3,
        rotate=False,
        rotation_direction: str | None = None,
        return_true=False,
        coral_mode=False,
        coral_filepath="/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/aggregator_real/",
    ):
        r"""Initialize the zernikeDataset.

        Parameters
        ----------
        seq_length : int
            The maximum sequence length for each sample.
        train : bool, optional, default=True
            Whether to load the training dataset or testing dataset.
        data_dir : str, optional
            The root directory containing the dataset files.
        alpha : float, optional, default=1e-3
            Parameter used for adjusting Zernike coefficients during processing.
        return_true : bool, optional, default=False
            Whether to return the true Zernike coefficients or estimated coefficients.
        rotate : bool, optional, default=False
            Whether to use rotated Zernikes in the camera frame when available.
        rotation_direction : {"pos", "neg"}, optional
            When rotate=True, selects which camera-frame Zernikes to use:
            - \"pos\" → zk_true_camera_pos (positive RTP rotation)
            - \"neg\" → zk_true_camera_neg (negative RTP rotation)
        coral_mode : bool, optional, default=False
            Whether to enable coral mode for real data sampling.
        coral_filepath : str, optional
            Path to the real/coral aggregator dataset directory.
        """
        self.max_seq_length = seq_length
        self.coral_mode = coral_mode
        self.coral_filepath = coral_filepath

        # Loop through all files and subdirectories
        if train:
            self.image_dir = data_dir + "/train"
        else:
            self.image_dir = data_dir + "/val"

        self.filename = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.filename.append(file_path)

        # Use all files in the directory (no additional train/val split)
        # The split is already handled by using different directories (train/val)
        self.num_samples = len(self.filename)
        self.alpha = alpha
        self.return_true = return_true

        # Load coral files if coral_mode is enabled
        if self.coral_mode:
            self.coral_files = []
            coral_data_path = coral_filepath
            if train:
                coral_data_path += "/train"
            else:
                coral_data_path += "/val"

            for root, _, files in os.walk(coral_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.coral_files.append(file_path)

            logger.info(f"Loaded {len(self.coral_files)} coral aggregator files from {coral_data_path}")

        # Always use CPU in dataset - PyTorch Lightning handles GPU transfer
        # This avoids CUDA reinitialization issues with num_workers > 0
        self.device = torch.device("cpu")
        self.rotate = rotate
        # Enforce explicit rotation direction when using rotated Zernikes
        if self.rotate and rotation_direction not in ("pos", "neg"):
            raise ValueError(
                f"rotation_direction must be 'pos' or 'neg' when rotate=True "
                f"(got {rotation_direction!r})."
            )
        self.rotation_direction = rotation_direction

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def sample_coral(self) -> Dict[str, Any]:
        """Sample a coral (real) aggregator data sample randomly.

        Returns
        -------
        dict
            Dictionary containing:
            - coral_x_total: Input features tensor (seq_length, features)
            - coral_mean: Mean Zernike coefficients
            - coral_filter: Filter name
            - coral_raftbay: Raft bay sensor name
        """
        # Check if coral files are available
        if not self.coral_files or len(self.coral_files) == 0:
            raise RuntimeError("No coral aggregator files available for sampling.")

        # Randomly sample from coral files with retry for corrupted files
        max_retries = 10
        corrupted_files = []

        for attempt in range(max_retries):
            try:
                # Re-check availability in case files were deleted
                if not self.coral_files:
                    raise RuntimeError("All coral aggregator files have been removed due to corruption.")

                idx = np.random.randint(0, len(self.coral_files))
                coral_file = self.coral_files[idx]

                # Skip already identified corrupted files
                if coral_file in corrupted_files:
                    continue

                # Load the coral data from npz
                npz_data = np.load(coral_file, allow_pickle=True)

                # Reconstruct dictionary - split stacked arrays back into lists
                loaded_data = {
                    "estimated_zk": [torch.from_numpy(arr) for arr in npz_data["estimated_zk"]],
                    "zk_mean": torch.from_numpy(npz_data["zk_mean"]),
                    "field_x": [torch.from_numpy(arr) for arr in npz_data["field_x"]],
                    "field_y": [torch.from_numpy(arr) for arr in npz_data["field_y"]],
                    "snr": npz_data["snr"].tolist(),
                    "header": json.loads(str(npz_data["header_json"])),
                }
                break  # Successfully loaded, exit retry loop
            except (pickle.UnpicklingError, EOFError, IOError, OSError) as e:
                # File is corrupted, truncated, or missing - delete it
                logger.warning(f"Corrupted coral file detected: {coral_file}. Error: {e}. Deleting...")
                try:
                    if os.path.exists(coral_file):
                        os.remove(coral_file)
                        logger.info(f"Deleted corrupted file: {coral_file}")
                except (OSError, PermissionError) as delete_error:
                    logger.warning(f"Failed to delete {coral_file}: {delete_error}")

                # Remove from list to avoid trying again
                if coral_file in self.coral_files:
                    self.coral_files.remove(coral_file)
                corrupted_files.append(coral_file)

                if attempt == max_retries - 1:
                    # Last attempt failed, raise the error
                    raise RuntimeError(
                        f"Failed to load coral file after {max_retries} attempts. "
                        f"Last error: {e}. All coral files may be corrupted."
                    )
                # Try another random file
                continue

        # Process coral data similar to __getitem__
        x = torch.stack(loaded_data["estimated_zk"]).to(self.device) / 1000
        mean = loaded_data["zk_mean"].to(self.device)

        # Track the field x/y in degrees
        field_x = torch.stack(loaded_data["field_x"])
        field_y = torch.stack(loaded_data["field_y"])

        # Load the SNR values + normalize
        snr = (
            torch.tensor(loaded_data["snr"]).to(self.device)[..., None]
            / torch.tensor(loaded_data["snr"]).max()
        )

        # Combine the field x/y
        position = torch.concatenate([field_x, field_y], dim=-1).to(self.device)

        # Combine all into one array as an embedding
        x = x.squeeze(1)  # [seq_length, features]
        position = position.squeeze(1)  # [seq_length, 2]
        x_total = torch.cat([x, position, snr], dim=1)

        # Control padding the sequence
        idx_tensor = torch.randperm(x_total.size(0))
        x_total = x_total[idx_tensor]

        if x_total.shape[0] > self.max_seq_length:
            x_total = x_total[: self.max_seq_length, :]
        else:
            padding = torch.zeros((self.max_seq_length - x_total.shape[0], x_total.shape[1])).to(self.device)
            x_total = torch.cat([x_total, padding], dim=0).to(self.device).float()

        # Extract filter and raftbay info
        filter_name = loaded_data["header"].get("FILTER", "unknown")
        if isinstance(filter_name, str):
            filter_name = filter_name.split("_")[0]

        raftbay = loaded_data["header"].get("RAFTBAY", "UNKNOWN") + "_SW0"

        coral_output = {
            "coral_x_total": x_total,
            "coral_mean": mean[None, ...],
            "coral_filter": filter_name,
            "coral_raftbay": raftbay,
        }

        return coral_output

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, torch.Tensor, torch.Tensor, str, str],
    ]:
        """Retrieve and process a single sample from the dataset at the specified index.

        This method loads a data sample from the file at the given index,
        processes various features (including
        Zernike coefficients, field positions, and SNR values), and
        formats them into a tensor suitable
        for input into a neural network. It also handles sequence length
        padding by truncating or padding
        the data to the specified maximum sequence length.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - x_total (torch.Tensor) : A tensor of input features,
              including the Zernike coefficients, field
              positions (x, y), and SNR values, shaped as `(seq_length, features)`.
            - mean (torch.Tensor) : A tensor containing the mean
              Zernike coefficient.
            - y (torch.Tensor) : A tensor of true Zernike
              coefficients (ground truth), shaped as `(N,)`.
            - filter_name (str) : Filter name extracted from header.
            - raftbay (str) : Raft bay sensor name from header.

        Notes
        -----
        - Zernike coefficients are converted using `convert_zernikes`
          function and normalized by dividing by 1000.
        - Field positions (x, y) are stacked and concatenated to form the position tensor.
        - SNR values are normalized to the range [0, 1].
        - If the resulting data exceeds `max_seq_length`, it is truncated;
          otherwise, it is padded with zeros
          to match the specified sequence length.
        """
        # Load dictionary from npz file
        try:
            npz_data = np.load(self.filename[idx], allow_pickle=True)

            # Reconstruct dictionary - split stacked arrays back into lists
            loaded_data = {
                "estimated_zk": [torch.from_numpy(arr) for arr in npz_data["estimated_zk"]],
                "zk_mean": torch.from_numpy(npz_data["zk_mean"]),
                "field_x": [torch.from_numpy(arr) for arr in npz_data["field_x"]],
                "field_y": [torch.from_numpy(arr) for arr in npz_data["field_y"]],
                "snr": npz_data["snr"].tolist(),
                "header": json.loads(str(npz_data["header_json"])),
            }
            # Add conditional fields if they exist
            # When rotate=True, use camera-frame Zernikes; enforce that the correct
            # explicit positive/negative field is present for a hard guarantee.
            if self.rotate:
                if self.rotation_direction == "neg":
                    if "zk_true_camera_neg" not in npz_data:
                        raise KeyError(
                            "Expected 'zk_true_camera_neg' in npz file when "
                            "rotate=True and rotation_direction='neg'."
                        )
                    loaded_data["zk_true"] = torch.from_numpy(npz_data["zk_true_camera_neg"])
                elif self.rotation_direction == "pos":
                    if "zk_true_camera_pos" not in npz_data:
                        raise KeyError(
                            "Expected 'zk_true_camera_pos' in npz file when "
                            "rotate=True and rotation_direction='pos'."
                        )
                    loaded_data["zk_true"] = torch.from_numpy(npz_data["zk_true_camera_pos"])
            else:
                if "zk_true" in npz_data:
                    loaded_data["zk_true"] = torch.from_numpy(npz_data["zk_true"])
                else:
                    raise KeyError("'zk_true' not found in npz file when rotate=False")
        except (IOError, OSError, RuntimeError, KeyError) as e:
            logger.error(
                f"Error loading file {self.filename[idx] if idx < len(self.filename) else 'unknown'}: {e}"
            )
            raise
        # convert zernikes microns
        x = torch.stack(loaded_data["estimated_zk"]).to(self.device) / 1000

        mean = loaded_data["zk_mean"].to(self.device)

        # track the field x/y in degrees
        field_x = torch.stack(loaded_data["field_x"])
        field_y = torch.stack(loaded_data["field_y"])
        # load the SNR values + normalise
        snr = (
            torch.tensor(loaded_data["snr"]).to(self.device)[..., None]
            / torch.tensor(loaded_data["snr"]).max()
        )
        # combine the field x/y
        position = torch.concatenate([field_x, field_y], dim=-1).to(self.device)
        # combine all into one array as an embedding
        # Remove singleton dimension for correct concatenation
        x = x.squeeze(1)  # [seq_length, 25]
        position = position.squeeze(1)  # [seq_length, 2]
        x_total = torch.cat([x, position, snr], dim=1)
        # control padding the sequence
        idx_tensor = torch.randperm(x_total.size(0))
        x_total = x_total[idx_tensor]
        if x_total.shape[0] > self.max_seq_length:
            x_total = x_total[: self.max_seq_length, :]
        else:
            padding = torch.zeros((self.max_seq_length - x_total.shape[0], x_total.shape[1])).to(self.device)
            x_total = torch.cat([x_total, padding], dim=0).to(self.device).float()
        y = loaded_data["zk_true"]

        # Prepare output dictionary
        output = {
            "x_total": x_total,
            "mean": mean[None, ...],
            "y": y,
            "filter": loaded_data["header"]["FILTER"].split("_")[0],
            "raftbay": loaded_data["header"]["RAFTBAY"] + "_SW0",
        }

        # Sample coral data if coral_mode is enabled
        if self.coral_mode:
            try:
                coral_output = self.sample_coral()
                output.update(coral_output)
            except RuntimeError as e:
                logger.warning(f"Failed to sample coral data: {e}")
                # Continue without coral data

        # return the stack of embedings, mean zernike estimate and the true zernike in PSF
        # Return as tuple for backward compatibility
        if self.coral_mode and "coral_x_total" in output:
            return (
                output["x_total"],
                output["mean"],
                output["y"],
                output["filter"],
                output["raftbay"],
                output["coral_x_total"],
                output["coral_mean"],
                output["coral_filter"],
                output["coral_raftbay"],
            )
        else:
            return (
                output["x_total"],
                output["mean"],
                output["y"],
                output["filter"],
                output["raftbay"],
            )


# Collate function for padding sequences
def zk_collate_fn(batch):
    """Custom collate function for batching samples in a DataLoader.

    Parameters
    ----------
    batch : list of tuples
        A list where each element is a tuple containing:
        - x (torch.Tensor) : Input features for the
            sample, shaped as `(seq_length, features)`.
        - x_mean (torch.Tensor) : Mean Zernike coefficients for
            the sample, shaped as `(1, features)`.
        - y (torch.Tensor) : True Zernike coefficients (target values)
            for the sample, shaped as `(1, features)`.
        - filter (str) : Filter name.
        - raftbay (str) : Raft bay sensor name.
        And optionally (if coral_mode=True):
        - coral_x_total (torch.Tensor) : Coral input features.
        - coral_mean (torch.Tensor) : Coral mean Zernike coefficients.
        - coral_filter (str) : Coral filter name.
        - coral_raftbay (str) : Coral raft bay sensor name.

    Returns
    -------
    tuple
        A tuple containing:
        - (x_total, x_mean_total, filter_total, chipid_total,
           [coral_x_total, coral_x_mean_total, coral_filter_total, coral_chipid_total]) :
            - x_total (torch.Tensor) : A tensor of input features
                for the entire batch, shaped
              as `(batch_size, seq_length, features)`.
            - x_mean_total (torch.Tensor) : A tensor of mean
                Zernike coefficients for the entire batch,
              shaped as `(batch_size, features)`.
            - filter_total (list) : List of filter names.
            - chipid_total (list) : List of raft bay sensor names.
            - [coral_*_total] : Optional coral data if available.
        - y_total (torch.Tensor) :
            - y_total (torch.Tensor) : A tensor of true Zernike
                coefficients (targets) for the entire batch,
              shaped as `(batch_size, features)`.

    Notes
    -----
    - The resulting tensors (`x_total`, `x_mean_total`, and `y_total`)
        are returned in a format suitable for training a model.
    - If coral_mode is enabled, coral data tensors are also included.
    """
    # Check if batch contains coral data (9 elements) or not (5 elements)
    has_coral = len(batch[0]) == 9

    if has_coral:
        (
            x_batch,
            x_mean_batch,
            y_batch,
            filter_batch,
            chipid_batch,
            coral_x_batch,
            coral_mean_batch,
            coral_filter_batch,
            coral_chipid_batch,
        ) = zip(*batch)
    else:
        x_batch, x_mean_batch, y_batch, filter_batch, chipid_batch = zip(*batch)

    x_total = torch.zeros((len(x_batch), x_batch[0].shape[0], x_batch[0].shape[1]))
    y_total = torch.zeros((len(y_batch), y_batch[0].shape[1]))
    x_mean_total = torch.zeros((len(x_mean_batch), x_mean_batch[0].shape[-1]))

    # match the parallel arrays together to get the values
    filter_total = []
    chipid_total = []

    for i, (x, x_mean, y, f, s) in enumerate(zip(x_batch, x_mean_batch, y_batch, filter_batch, chipid_batch)):
        x_total[i, :, :] = x
        y_total[i, :] = y[0, :]
        x_mean_total[i, :] = x_mean[0, 0, :]  # <-- fix here
        filter_total.append(f)
        chipid_total.append(s)

    # Process coral data if available
    if has_coral:
        coral_x_total = torch.zeros(
            (len(coral_x_batch), coral_x_batch[0].shape[0], coral_x_batch[0].shape[1])
        )
        coral_x_mean_total = torch.zeros((len(coral_mean_batch), coral_mean_batch[0].shape[-1]))
        coral_filter_total = []
        coral_chipid_total = []

        for i, (cx, cm, cf, cs) in enumerate(
            zip(coral_x_batch, coral_mean_batch, coral_filter_batch, coral_chipid_batch)
        ):
            coral_x_total[i, :, :] = cx
            coral_x_mean_total[i, :] = cm[0, 0, :]
            coral_filter_total.append(cf)
            coral_chipid_total.append(cs)

        return (
            x_total,
            x_mean_total,
            filter_total,
            chipid_total,
            coral_x_total,
            coral_x_mean_total,
            coral_filter_total,
            coral_chipid_total,
        ), y_total
    else:
        return (x_total, x_mean_total, filter_total, chipid_total), y_total
