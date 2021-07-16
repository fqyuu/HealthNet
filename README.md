# DepHIM

## Installation

Install Python 3.5.

Install tensorflow 1.12.2.

## Data

We provide a processed CS dataset for training/evaluation. Please click [MIMIC](https://mimic.mit.edu/docs/gettingstarted/) to get the instructions for getting access to MIMIC-III. Researchers seeking to use the database must:

- Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
- Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
- Follow the tutorials for direct cloud access (recommended), or download the data locally.

Put the files 'ADMISSIONS.csv', 'DIAGNOSES_ICD.csv', 'PATIENTS.csv', 'PRESCRIPTIONS.csv' and 'PROCEDURES_ICD.csv' in the folder 'data' before proceeding. 

## Usage

### 1. Generate Candidates

To filter POIs and reduce the search space.
```bash
$ python train.py
```
To train and evaluate Encoder 1 and Filter, we split each dataset into a training set, a validation set and a test set, here. Encoder 1 and filtering layers form a reasonable filter capable of reducing search space, i.e., reducing the number of candidates from which recommended POIs are selected finally.

Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.

### 2. Rank POI

To sort the POIs in the candidate set.

```bash
$ python train_rankpoi.py
```
Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.
