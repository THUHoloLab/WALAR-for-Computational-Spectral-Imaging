# Learning Spectral Images One Wavelength at a Time

**Xinyu Liu, Yunhui Gao, and [Liangcai Cao](https://scholar.google.com/citations?user=FYYb_-wAAAAJ&hl=en)**

Affiliation: *[HoloLab](http://www.holoddd.com/), State Key Laboratory of
Precision Measurement Technology and Instruments, Department of Precision
Instrument, Tsinghua University, Beijing, China*

## WALAR

Wavelength-guided adaptive learning and reconstruction (WALAR) learns from
physically recorded Bayer measurements and synchronized, response-corrected
single-band references. Samples at different wavelengths train one shared
wavelength-conditioned network, TASSIR-Net, without requiring a complete
spectral reference for each training scene. At inference, the same measurement
is queried at 26 wavelengths from 450 to 700 nm; the predicted bands form a
spectral datacube.

## Data

Data are split by acquisition day (7 training, 1 validation, 1 test). The final
configuration uses 550,000 of the 555,614 retained training pairs, with 7,800
validation pairs and 13,000 test pairs.

Each split is stored in `data/train`, `data/validation`, or `data/test`:

```text
data/train/
|-- data.mdb
`-- lock.mdb
```

`data.mdb` stores samples under eight-digit keys (`00000000`, `00000001`, ...),
with `__len__` and `__meta__` metadata. Each sample is a serialized dictionary:

| Field | Content |
|---|---|
| `measurement` | Single-channel Bayer raw array |
| `truth` | Synchronized, response-corrected single-band reference |
| `wavelength` | Scalar `(wavelength_nm - 450) / 250` |
| `wavelength_nm` | Target wavelength in nanometers |
| `filename` | Original image name |

Both images are `float32` tensors of shape `(1, H, W)`, scaled by `1/255`.
LMDB reduces small-file I/O for the large training set. The read-only loader
requires `data.mdb`; `lock.mdb` is managed by LMDB.

For conversion, arrange registered, temporally matched image pairs by wavelength,
with matching relative filenames. The references must already be response-corrected.
Store both images as 8-bit single-channel files, retaining the Bayer array
without demosaicing or color interpolation. The converter applies scaling,
not calibration.

```text
raw_session/450/measurement/frame.png
raw_session/450/truth/frame.png
raw_session/460/measurement/frame.png
raw_session/460/truth/frame.png
```

The data and pretrained weights are not included in this repository.

## Usage

Dependencies: PyTorch, LMDB (`lmdb`), NumPy, Pillow, and tqdm.

```bash
python tools/build_lmdb.py --source ../raw/day01 ../raw/day02 --output ./data/train
python train.py --train-size 550000
python evaluate.py --checkpoint ./outputs/best.pt --data-dir ./data/test
```

Prepare the validation and test LMDBs in their respective directories before
training and evaluation. `--source` accepts multiple acquisition sessions.

Defaults: 490 x 490 inputs, 16 x 16 patches, stride 12, batch size 32,
60 epochs, and AdamW with learning rate and weight decay of `1e-4`.
The loss is `MSE - 0.1 x SSIM`. Validation PSNR selects `outputs/best.pt`.

`--train-size` takes a prefix of a fixed random permutation (seed 2026);
omit it to use all training records. Resume with `--resume ./outputs/last.pt`,
keeping the same subset settings.

Evaluation writes overall and wavelength-resolved metrics to
`evaluation/metrics.json`. PSNR is calculated per image before averaging;
SSIM uses an 11 x 11 Gaussian window with sigma 1.5 and data range 1.
The test set is not used for model selection.

The reported experiments used an NVIDIA GeForce RTX 4090 GPU and a 12th Gen
Intel(R) Core(TM) i9-12900K CPU.
