# DepHIM

## Installation

Install Python 3.5.

Install tensorflow 1.12.2.

## Data

We provide a processed CS dataset for training/evaluation. Please click [here](https://mimic.mit.edu/docs/gettingstarted/) to get the instructions for getting access to MIMIC-III. Researchers seeking to use the database must:

- Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
- Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
- Follow the tutorials for direct cloud access (recommended), or download the data locally.

After you have completed all the steps and gained access to MIMIC data, place the files 'ADMISSIONS.csv', 'DIAGNOSES_ICD.csv', 'PATIENTS.csv', 'PRESCRIPTIONS.csv' and 'PROCEDURES_ICD.csv' under the folder 'data'.

## Usage

### 1. Build Graph and Learn Node Embedding

```bash
$ python bulid_CPmatrix.py
```
The functions constructing conditional probability matrices, building multi-dependency graph and learning the node embedding are encapsulated in the same script. Note that the embedding size can be configured through util.py

### 2. Perform Healthcare Tasks Using Hierarchical Sequence Model

Use DepHIM to make diagnosis prediction and mortality prediction.

```bash
$ python train.py
```
To train and evaluate our DepHIM, we split each dataset in 7:1:2 for training, validation and testing. Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test. The data path and the healthcare tasks can be configured through util.py.
