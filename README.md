# TARTS

TARTS stands for (T)riple-stage (A)lignment and (R)econstruction using (T)ransformer (S)ystems for Active Optics. It is a modular PyTorch/PyTorch Lightning package for estimating Zernike wavefront coefficients from LSST defocused images. The triple-stage design consists of: (1) AlignNet for donut alignment/centering and field metadata normalization, (2) WaveNet for per-donut Zernike regression, and (3) AggregatorNet (transformer-based) for sequence-level fusion of multiple donut predictions. The package includes utilities (`tarts.utils`), datasets/dataloaders, and the high-level `NeuralActiveOpticsSys` orchestrator. Core configuration (e.g., `noll_zk`, crop size, sequence length) is provided in `TARTS/python/tarts/dataset_params.yaml`.

## Features

- Triple-stage active optics pipeline:
  - AlignNet: robust donut centering and field metadata normalization
  - WaveNet: per-donut Zernike regression with CNN feature extractor
  - AggregatorNet: transformer-based fusion across multiple donuts
- Utilities for Zernike conversion, plotting, cropping, SNR filtering, and dataset helpers
- PyTorch Lightning modules for training, validation, and inference
- YAML-configurable parameters (`dataset_params.yaml`)

## Requirements

- Python 3.9+
- PyTorch, PyTorch Lightning
- NumPy, PyYAML, matplotlib
- (Optional, for LSST integrations) Rubin/LSST Science Pipelines

## Installation

Install only the TARTS package (recommended for library usage):

```bash
pip install -e TARTS/python
```

Or add the package path dynamically in your scripts:

```python
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / 'TARTS' / 'python'))
```

## Configuration

Project parameters are centralized in:

```
TARTS/python/tarts/dataset_params.yaml
```

Common entries include:
- `noll_zk`: list of Noll indices used by downstream pipelines
- `CROP_SIZE`, `max_seq_len`, `deg_per_pix`, `mm_pix`, `alpha`
- AggregatorNet configuration under `aggregator_model`

Load safely in Python:

```python
from tarts.utils import safe_yaml_load
params = safe_yaml_load('TARTS/python/tarts/dataset_params.yaml')
noll_zk = params['noll_zk']
```

## Modules

- `tarts.utils`
  - Zernike conversions: `convert_zernikes`, `convert_zernikes_deploy`
  - Image helpers: `batched_crop`, `get_centers`, `single_conv`, `filter_SNR`
  - Misc: `count_parameters`, `printOnce`, `safe_yaml_load`
- `tarts.dataloader`
  - `Donuts` and `Donuts_Fullframe` datasets for simulations/ImSim-style data
  - Collate function `zk_collate_fn` for batching sequences
- `tarts.lightning_alignnet`
  - `AlignNetSystem` and `DonutLoader` Lightning modules
- `tarts.lightning_wavenet`
  - `WaveNetSystem`, `DonutLoader`, and `DonutLoader_Fullframe`
- `tarts.NeuralActiveOpticsSys`
  - High-level orchestrator that wires AlignNet, WaveNet, and AggregatorNet for inference

## Minimal usage (NeuralActiveOpticsSys)

```python
from tarts.NeuralActiveOpticsSys import NeuralActiveOpticsSys

model = NeuralActiveOpticsSys(dataset_params='TARTS/python/tarts/dataset_params.yaml')
# Use forward/forward_align/forward_shifts, or deploy_run for LSST exposures
```

## Notes

- When integrating with Rubin/LSST data, ensure the LSST science stack is available; this package itself does not enforce LSST dependencies by default.
- For training with your own data, adapt the datasets under `tarts.dataloader` and keep `noll_zk` consistent across stages.

## License

TBD.
