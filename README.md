# Iterative Algorithm for Artifact Reduction

### Objective

---

Correct artifacts in CT scans.

### Installation

---

Unpack.

Note: Use Python 3.11.10

Install dependencies on environment from requirements.txt.

Replace '(environment name)\Lib\site-packages\getcatsim\reconstruction\pyfiles\recon.py' with provided '\libs\recon.py'.

### Use

---

Run demo_compare notebook file for comparison.

Run demo_single notebook file for no comparing.

### :arrow_down_small: Inputs

---

Data folder contains some separated sample data to try the program on.

The 'with_artifacts' folders contains artifact samples. The 'without_artifacts' folder contains the clean versions of those samples.

There are two types of anatomy for input. 'o' for body, and 'h' for head.

Copy path to corresponding text inputs in the demo script.

The same applies to the weights .pth file in the 'models' folder.

### Notes

---

Currently, sometimes, results vary depending on sample.

AAPMProj, AAPMRecon scripts belong to its developer.
