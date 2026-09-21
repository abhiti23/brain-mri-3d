# Setup

## Starting from zero

1. Go to: https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing
2. Download **OpenBHB-Training Data** (Size: 42.15 GB). You should have a `train.zip` file at the end of this.
3. Download **OpenBHB-Validation Data** (Size: 9.91 GB). You should have a `val.zip` file at the end of this.

## For training set

Unzip the file:

```bash
unzip train.zip
```

If it triggers a zip bomb error, do this and select `[A]`:

```bash
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip train.zip
```

Remove unnecessary files:

```bash
rm train_fsl.zip train_quasiraw.zip train_roi.zip
```

Unzip VBM data:

```bash
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip train_vbm.zip
```

Unzip labels:

```bash
unzip train_labels.zip
ls *.csv 2>/dev/null || ls *.tsv 2>/dev/null || ls *label* 2>/dev/null
```

Repeat for validation set. Then feature extraction (GS or FDA) can begin.
