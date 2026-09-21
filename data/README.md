# Data

Each model reads a CSV manifest with one row per image, volume or MRI session. The same format is used by `code/train.py` (training and evaluation) and `code/predict.py` (age estimation with a released checkpoint). The example files in this folder contain made-up rows that show the format only.

## Retinal Age and Chest Age (2D images)

| Column | Description |
|---|---|
| `subject_id` | Identifier of the row (for example the image file name). |
| `image_path` | Path to the image. It must open with `PIL.Image.open(path).convert("RGB")`. |
| `biomarker_value` | Chronological age in years at imaging. Optional for `predict.py`. |

Example: [`example_2d.csv`](example_2d.csv).

- Retinal Age: colour fundus photographs, cropped around the fundus and padded to a square (as in RETFound). In the paper, only images graded Good or Usable by AutoMorph were used.
- Chest Age: frontal (PA or AP) chest radiographs, as PNG or JPEG.
- Images are resized to 256 pixels and centre-cropped to 224 × 224 at evaluation. No other preprocessing is needed.

## Abdominal Age (CT)

| Column | Description |
|---|---|
| `subject_id` | Identifier of the volume. |
| `image_path` | Path to the CT volume as NIfTI (`.nii` or `.nii.gz`) in Hounsfield units. |
| `biomarker_value` | Chronological age in years at imaging. Optional for `predict.py`. |

Example: [`example_ct.csv`](example_ct.csv).

Each volume is processed on the fly: resampled to 1.5 mm isotropic spacing, the body is segmented at −950 HU and cropped with a 15 mm margin, and 32 axial slices are sampled. Nothing derived from the volume is written to disk. The model was trained on the Merlin abdominal and pelvic CT studies.

## Brain Age (T1-weighted MRI)

| Column | Description |
|---|---|
| `subject_id` | Identifier of the patient. |
| `session_id` | Identifier of the imaging session. Must be unique. |
| `archive_path` | Either an OASIS-3 session ZIP archive (every `*T1w*.nii.gz` member is one run) or a single T1w NIfTI file. `predict.py` also accepts this column as `image_path`. |
| `biomarker_value` | Chronological age in years at the session. Optional for `predict.py`. |

Example: [`example_mri.csv`](example_mri.csv).

Each T1w run is reoriented to RAS, intensity-clipped to the 0.5th and 99.5th percentiles, cropped to the head with a 5 mm margin, and 32 axial slices are sampled. When a session has several T1w runs, training picks one at random and evaluation averages the predictions of all runs.

## Splits used in the paper

In the paper, AlzEye, ChestX-ray14 and OASIS-3 were split at the patient level. Merlin releases no patient identifiers, so its official split was kept. The age models were trained on healthy patients only. The split files are not released, because they are built from clinical linkage and from data covered by data use agreements. The dataset sources are listed in the main [README](../README.md#2-get-model-weights-and-data).
