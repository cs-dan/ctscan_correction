# Iterative Algorithm for Artifact Reduction

### :dart: Objective

---

Correct artifacts in CT scans.

To do so by a more lightweight approach, backpropagating the error map to function as a denoiser from a classifier.

### :construction: Installation

---

Unpack.

Note: Use Python 3.11.10

Install dependencies on environment from requirements.txt.

Replace '(environment name)\Lib\site-packages\getcatsim\reconstruction\pyfiles\recon.py' with provided '\libs\recon.py'.

### :heavy_exclamation_mark: Use

---

Run demo_compare notebook file for comparison.

Run demo_single notebook file for no comparing, inference.

### :arrow_down_small: Inputs

---

Data folder contains some separated sample data to try the program on.

The 'with_artifacts' folders contains artifact samples. The 'without_artifacts' folder contains the clean versions of those samples.

There are two types of anatomy for input:

* 'o' for body

* 'h' for head.

Copy path to corresponding text inputs in the demo script.

The same applies to the weights .pth file in the 'models' folder.

### :clipboard: Notes

---

Currently, sometimes, results vary depending on sample.

To develop more consistent results, a larger or more varied dataset may be considered.

Alternatively, the classifier architecture could be scaled up, however that would defeat the purpose of this project.

The simple method, metes out results fulfilling its purpose however.

### :copyright: Disclaimers

---

AAPMProj, AAPMRecon scripts belong to its developer. I have only retrofited some functionality in them for the purposes of this project.
