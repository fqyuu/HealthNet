# DepHIM

## Installation

Install Python 3.7.9, tensorflow 2.0.0.

If you plan to use GPU computation, install CUDA.

## Data

We provide the links to obtain the file or tools needed in data preprocessing:

- [ccs_multi_dx_tool_2015.csv](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip): Multi-level CCS (Clinical Classification Software) for ICD9.

- [MEDI_11242015.csv](https://www.vumc.org/cpm/sites/vumc.org.cpm/files/public_files/MEDI_11242015.csv): MEDI (MEDication Indication) is an ensemble medication indication resource for primary and secondary uses of electronic medical record (EMR) data.

- [ndc2atc_level4.csv, ndc_rxnorm_mapping.txt](https://github.com/sjy1203/GAMENet/tree/master/data): mapping files for drug code transformation.

Due to the sensitivity and confidentiality of EHR data, please click [here](https://mimic.mit.edu/docs/gettingstarted/) to get the instructions for getting access to MIMIC-III. Researchers seeking to use the database must:

- Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
- Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
- Follow the tutorials for direct cloud access (recommended), or download the data locally.

## Usage

### 1. Data Preprocessing
After you have completed all the steps and gained access to MIMIC data, place the files 'ADMISSIONS.csv', 'DIAGNOSES_ICD.csv', 'PATIENTS.csv', 'PRESCRIPTIONS.csv' and 'PROCEDURES_ICD.csv' under the folder 'data'.
All the functions of data preprocessing are encapsulated and executed in step 2. 

### 2. Build Graph and Learn Node Embedding

```bash
$ python bulid_CPmatrix.py
```
The functions constructing conditional probability matrices, building multi-dependency graph and learning the node embedding are encapsulated in the same script. Note that the embedding size of node can be configured through util.py.

### 3. Perform Healthcare Tasks Using Hierarchical Sequence Model

Use DepHIM to make diagnosis prediction and mortality prediction.

```bash
$ python train.py
```
To train and evaluate our DepHIM, we split each dataset in 7:1:2 for training, validation and testing. Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test. The data path and the healthcare tasks can be configured through util.py.
