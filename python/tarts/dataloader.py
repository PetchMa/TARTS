"""Pytorch DataSet for the AOS simulations."""

import glob
from typing import Any, Dict
import pickle
import numpy as np
import torch
from astropy.table import Table
from .utils import transform_inputs, shift_offcenter
from torch.utils.data import Dataset
import os


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
        intra = torch.FloatTensor([intra])  # type: ignore
        band = torch.FloatTensor([band])  # type: ignore
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
        noll_zk: list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28],
        coral_filepath: str = "/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/coral/",
        coral_mode: bool = False,
        mask_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load the simulated ImSim donuts and zernikes in a Pytorch Dataset.

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
        noll_zk: list, default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
            List of Noll Zernike indices to include in the dataset.
        coral_filepath: str, default="/media/peterma/mnt2/peterma/research/LSST_FULL_FRAME/coral/"
            Path to the coral dataset directory.
        coral_mode: bool, default=False
            Whether to enable coral mode for domain adaptation.
        mask_mode: bool, default=False
            Whether to use mask mode for zernike extraction.
        """
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
        print(self.image_dir)
        self.coral_filepath = coral_filepath
        self.coral_mode = coral_mode
        if coral_mode:
            self.coral_image_files = []
            # Use a temporary variable to avoid modifying the original path
            coral_data_path = coral_filepath
            if self.settings["mode"] == "train":
                coral_data_path += "/train"
            if self.settings["mode"] == "val":
                coral_data_path += "/val"
            for root, _, files in os.walk(coral_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.coral_image_files.append(file_path)

        if self.settings["mode"] == "train":
            # self.image_files = self.image_files[: int(0.1 * len(self.image_files))]

            self.image_files = self.image_files[: int(0.5 * len(self.image_files))]

        self.noll_zk = np.array(noll_zk) - 4
        self.adjustment_factor = adjustment_factor

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.image_files)  # type: ignore

    def sample_coral(self) -> Dict[str, torch.Tensor]:
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
                if len(self.coral_image_files) == 0:
                    raise RuntimeError("All coral image files have been removed due to corruption.")

                idx = np.random.randint(0, len(self.coral_image_files))
                img_file = self.coral_image_files[idx]

                # Skip already identified corrupted files
                if img_file in corrupted_files:
                    continue

                state = np.load(img_file, allow_pickle=True)
                break  # Successfully loaded, exit retry loop
            except (EOFError, IOError, OSError) as e:
                # File is corrupted, truncated, or missing - delete it
                print(f"⚠️  Corrupted coral file detected: {img_file}. Error: {e}. Deleting...")
                try:
                    if os.path.exists(img_file):
                        os.remove(img_file)
                        print(f"✓ Deleted corrupted file: {img_file}")
                except Exception as delete_error:
                    print(f"⚠️  Failed to delete {img_file}: {delete_error}")

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

        band = torch.tensor(state["band"]).int().item()
        img = torch.tensor(state["image_aligned"])

        # Get zernikes using noll_zk indexing
        # Convert to numpy first to handle object dtype from npz files
        zernikes = torch.zeros((1, len(self.noll_zk)))

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
        img = img.float()
        # get meta data
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])  # type: ignore
        band = torch.FloatTensor([band])  # type: ignore
        zernikes = zernikes.float()[0, :]
        coral_output = {
            "coral_image": img,
            "coral_field_x": fx,
            "coral_field_y": fy,
            "coral_intrafocal": intra,
            "coral_band": band,
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
        # get the image file
        img_file = self.image_files[idx]

        state = np.load(img_file, allow_pickle=True)

        # get the donut locations
        fx, fy = (
            torch.tensor(state["field_x"]) * np.pi / 180,
            torch.tensor(state["field_y"]) * np.pi / 180,
        )

        # get the intra/extra flag
        intra = torch.tensor(state["intra"]).int()

        if self.mask_mode:
            zernikes = torch.tensor(state["zk_true"])
            zernikes = zernikes[:, 0]
            zernikes = zernikes[None, :]
        else:
            zernikes = torch.tensor(state["zk_true"])[:, self.noll_zk]

        band = torch.tensor(state["band"]).int().item()
        img = torch.tensor(state["image_aligned"])

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
        img = img.float()

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

        # get meta data
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])  # type: ignore
        band = torch.FloatTensor([band])  # type: ignore
        zernikes = zernikes.float()[0, :]

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
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
        return_true=False,
    ):
        """Initialize the zernikeDataset.

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
        """
        self.max_seq_length = seq_length
        # Loop through all files and subdirectories
        if train:
            self.image_dir = data_dir + "/train"
        else:
            self.image_dir = data_dir + "/train"

        self.filename = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.filename.append(file_path)

        if train:
            self.filename = self.filename[: int(0.8 * len(self.filename))]
        else:
            self.filename = self.filename[int(0.8 * len(self.filename)) :]

        self.num_samples = len(self.filename)
        self.alpha = alpha
        self.return_true = return_true
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
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
        # Load dictionary from file
        try:
            with open(self.filename[idx], "rb") as file:
                loaded_data = pickle.load(file)
        except Exception:
            print(file)
            print("error")
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
        position = torch.concatenate([field_x, field_y], axis=-1).to(self.device)
        # combine all into one array as an embedding
        # Remove singleton dimension for correct concatenation
        x = x.squeeze(1)  # [seq_length, 25]
        position = position.squeeze(1)  # [seq_length, 2]
        x_total = torch.cat([x, position, snr], dim=1)
        # control padding the sequence
        idx = torch.randperm(x_total.size(0))
        x_total = x_total[idx]
        if x_total.shape[0] > self.max_seq_length:
            x_total = x_total[: self.max_seq_length, :]
        else:
            padding = torch.zeros((self.max_seq_length - x_total.shape[0], x_total.shape[1])).to(self.device)
            x_total = torch.cat([x_total, padding], axis=0).to(self.device).float()
        y = loaded_data["zk_true"]
        # return the stack of embedings, mean zernike estimate and the true zernike in PSF
        return x_total, mean[None, ...], y


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

    Returns
    -------
    tuple
        A tuple containing:
        - (x_total, x_mean_total) :
            - x_total (torch.Tensor) : A tensor of input features
                for the entire batch, shaped
              as `(batch_size, seq_length, features)`.
            - x_mean_total (torch.Tensor) : A tensor of mean
                Zernike coefficients for the entire batch,
              shaped as `(batch_size, features)`.
        - y_total (torch.Tensor) :
            - y_total (torch.Tensor) : A tensor of true Zernike
                coefficients (targets) for the entire batch,
              shaped as `(batch_size, features)`.

    Notes
    -----
    - The resulting tensors (`x_total`, `x_mean_total`, and `y_total`)
        are returned in a format suitable for training a model.
    """
    x_batch, x_mean_batch, y_batch = zip(*batch)
    x_total = torch.zeros((len(x_batch), x_batch[0].shape[0], x_batch[0].shape[1]))
    y_total = torch.zeros((len(y_batch), y_batch[0].shape[1]))
    x_mean_total = torch.zeros((len(x_mean_batch), x_mean_batch[0].shape[-1]))
    # match the parallel arrays together to get the values
    for i, (x, x_mean, y) in enumerate(zip(x_batch, x_mean_batch, y_batch)):
        x_total[i, :, :] = x
        y_total[i, :] = y[0, :]
        x_mean_total[i, :] = x_mean[0, 0, :]  # <-- fix here
    return (x_total, x_mean_total), y_total
