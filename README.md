# Wavelength-Guided-Adaptive-Learning-for-Dynamic-Scene-Computational-Spectral-Imaging

Training and evaluation code for wavelength-guided adaptive learning and
reconstruction (WALAR). TASSIR-Net takes an encoded measurement and a target
wavelength as input and reconstructs the corresponding single-band image.

## Files

```text
train.py                 training and validation
evaluate.py              held-out test evaluation
tools/build_lmdb.py      image pairs to LMDB
walar/model.py           TASSIR-Net
walar/data.py            LMDB dataset reader
walar/losses.py          MSE and single-band SSIM loss
walar/engine.py          training and evaluation loops
data/{train,validation,test}/
```

## Dependencies

PyTorch, LMDB (`lmdb`), NumPy, Pillow, and tqdm.

## Data

Each split is stored as an LMDB directory:

```text
data/train/
|-- data.mdb
`-- lock.mdb
```

`data.mdb` contains the indexed samples and two metadata records, `__len__` and
`__meta__`. Each sample is a serialized dictionary with the following fields:

| Field | Content |
|---|---|
| `measurement` | Encoded image, `float32`, shape `(1, H, W)`, range `[0, 1]` |
| `truth` | Synchronized target-band image with the same shape and range |
| `wavelength` | Scalar wavelength input normalized to `[0, 1]` |
| `wavelength_nm` | Target wavelength in nanometers |
| `filename` | Original image name |

The reported wavelength input is

```text
(wavelength_nm - 450) / (700 - 450).
```

LMDB is used because the training set contains hundreds of thousands of image
pairs. Keeping the samples in one indexed database avoids repeated directory
scans and reduces the cost of opening large numbers of small image files, which
improves DataLoader throughput during training.

Before conversion, paired images are arranged by wavelength. Filenames in the
`measurement` and `truth` directories must match.

```text
raw_session/
|-- 450/
|   |-- measurement/
|   `-- truth/
|-- 460/
|   |-- measurement/
|   `-- truth/
`-- ...
```

Several acquisition sessions can be combined into one split:

```text
python tools/build_lmdb.py --source ../raw/day01 ../raw/day02 --output ./data/train
```

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| Traffic scenes | 555,614 | 7,800 | 13,000 |
| OLED screen | 114,720 | 5,200 | 5,200 |

Traffic data were divided by acquisition day (7/1/1 days). The OLED subsets are
temporally disjoint; the test set contains animation sequences not used for
training or validation.

## Training

```text
python train.py --train-dir ./data/train --validation-dir ./data/validation --output-dir ./outputs
```

The default configuration matches the reported traffic experiment: 490 x 490
inputs, 16 x 16 patches with a stride of 12, batch size 32, 60 epochs, and AdamW
with a fixed learning rate and weight decay of `1e-4`. The loss is
`MSE - 0.1 x SSIM`. Validation PSNR selects `best.pt`; `last.pt` can be used to
resume an interrupted run.

## Evaluation

```text
python evaluate.py --checkpoint ./outputs/best.pt --data-dir ./data/test
```

The evaluation report contains overall and wavelength-resolved MSE, PSNR, and
SSIM. Test data are not used for checkpoint selection.

The reported experiments used an NVIDIA GeForce RTX 4090 GPU and a 12th Gen
Intel(R) Core(TM) i9-12900K CPU.
