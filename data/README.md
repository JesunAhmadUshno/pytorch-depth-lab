# data/

No dataset files live in this repo, and none are committed.

To reproduce the study, fetch EuroSAT RGB yourself:

- File: EuroSAT_RGB.zip (94.7 MB), Zenodo record 7711810,
  Version v2. Record md5 f46e308c4d50d4bf32fedad2d3b.
- URL: https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
- Licence: MIT (official repository, phelber/EuroSAT; see
  ../LICENSES.md).
- Extract so that this directory contains EuroSAT_RGB/ with one
  folder per class (AnnualCrop, Forest, HerbaceousVegetation,
  Highway, Industrial, Pasture, PermanentCrop, Residential,
  River, SeaLake), 27,000 JPEG images in total.
- Verify: `python -m src.data --verify` recomputes the zip
  SHA-256 (compare results/dataset.json), counts images per
  class, and regenerates the frozen split (seed 20261107,
  21,600 / 2,700 / 2,700) into results/split.json.

Local layout used by the code (all ignored by git):

- data/EuroSAT_RGB.zip (the downloaded archive)
- data/EuroSAT_RGB/ (extracted images)
