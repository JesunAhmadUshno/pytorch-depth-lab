# LICENSES

## This repo's code

Apache-2.0. See LICENSE.

## EuroSAT dataset

- Source checked before any download: the dataset's official
  repository, phelber/EuroSAT on GitHub, License section, read
  2026-11-07: "The dataset is licensed under the MIT license. In
  general, Sentinel data is free and open to the public under EU
  law."
- The verdict pack carried two contradictory rows (CC BY-NC-SA
  4.0 in E19, an unverified MIT line in E20). The official
  repository settles it: MIT. The CC BY-NC-SA figure is not used
  anywhere in this repo.
- Download used: EuroSAT_RGB.zip, Zenodo record 7711810,
  Version v2 (the authors' current hosting). Record md5:
  f46e308c4d50d4bf32fedad2d3b. SHA-256 of the file as downloaded
  is in results/dataset.json.
- Copernicus Sentinel data terms apply to the underlying imagery
  (Sentinel data is free and open under EU law; see the
  Copernicus Sentinel Data Terms and Conditions).
- No dataset files are committed to this repo. Only derived
  statistics (split indices, metrics, counts) are stored.

## ImageNet weights

ResNet-18 IMAGENET1K_V1 weights are fetched from PyTorch's
official download host at run time by torchvision and are used
under the torchvision model-zoo terms. The weights file itself is
not committed here.

## Libraries

PyTorch and TorchInductor: BSD-style licence (pytorch/pytorch).
PyTorch Lightning: Apache-2.0. torchvision: BSD. Exact installed
versions: results/versions.json.
