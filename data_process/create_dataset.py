import pandas as pd
import os
from data_process.util import dataset_name,dataset_path
def createdatset(path_source,save_path):
    ad=pd.read_csv(path_source+'ADMISSIONS.csv',usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','EDREGTIME','HOSPITAL_EXPIRE_FLAG'])
    ccs=pd.read_csv(path_source+'ccs_multi_dx_tool_2015.csv',quotechar='\'')
    patients=pd.read_csv(path_source+'PATIENTS.csv')
    diag_icd=pd.read_csv(path_source+'DIAGNOSES_ICD.csv',usecols=['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE'])
    pro_icd = pd.read_csv(path_source + 'PROCEDURES_ICD.csv',usecols=['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE'])

    ad_cs = ad[ad['SUBJECT_ID'].isin(diag_icd[diag_icd['ICD9_CODE'].isin(
        ccs[ccs['CCS LVL 1 LABEL'] == '"Diseases of the circulatory system"']['ICD-9-CM CODE'])]['SUBJECT_ID'])]
    ad_cs.columns = ad_cs.columns.map(lambda x: x.lower())
    ad_cs=ad_cs.reset_index(drop=True)
    ad_cs.to_csv(save_path + 'admissions.csv', index=False)

    patients.columns = patients.columns.map(lambda x: x.lower())
    patient_cs=patients[patients['subject_id'].isin(ad_cs['subject_id'].drop_duplicates())]
    patient_cs=patient_cs[['subject_id', 'gender','dob']].reset_index(drop=True)
    patient_cs.to_csv(save_path + 'patients.csv', index=False)

    med_atc=ndc2atc4(ad_cs,path_source+'ndc2rxnorm_mapping.txt',path_source+'ndc2atc_level4.csv')
    med_atc.to_csv(save_path+'prescriptions.csv',index=False)

    diag_icd.columns = diag_icd.columns.map(lambda x: x.lower())
    diag_cs=diag_icd[diag_icd['subject_id'].isin(ad_cs['subject_id'].drop_duplicates())]
    diag_cs=diag_cs.reset_index(drop=True)
    diag_cs.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    diag_cs=diag_cs.reset_index(drop=True)
    diag_cs.to_csv(save_path + 'diagnoses_icd.csv', index=False)

    medi_med=read_medi(path_source+"MEDI_11242015.csv")
    med2diag=merge_med_icd(diag_cs,med_atc,medi_med)
    med2diag.to_csv(save_path+'med2diag.csv',index=False)

    pro_icd.columns = pro_icd.columns.map(lambda x: x.lower())
    pro_cs=pro_icd[pro_icd['subject_id'].isin(ad_cs['subject_id'].drop_duplicates())]
    pro_cs=pro_cs.reset_index(drop=True)
    pro_cs.to_csv(save_path + 'procedures_icd.csv', index=False)

def read_medi(medi_path):
    medi_med=pd.read_csv(medi_path)
    medi_med.drop(columns=['RXCUI_IN',"STR","CUI","NDDF","HSP"],axis=1,inplace=True)
    medi_med.dropna(inplace=True)

    # ATC-5 to ATC-4
    medi_med['ATC']=medi_med['ATC'].map(lambda x:x[0:5])
    # Remove points in code
    medi_med['CODE'] = medi_med['CODE'].map(lambda x: ''.join(filter(str.isalnum,x)))

    medi_med.drop_duplicates(inplace=True)
    medi_med = medi_med.reset_index(drop=True)
    return medi_med

def ndc2atc4(ad_cs,ndc2rx_path,rx2atc_path):
    med = pd.read_csv(path_source + 'PRESCRIPTIONS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'NDC'],dtype={'NDC':str})
    med.columns = med.columns.map(lambda x: x.lower())
    med_cs=med[med['subject_id'].isin(ad_cs['subject_id'].drop_duplicates())]
    med_ndc=med_cs.reset_index(drop=True)

    med_ndc['ndc']=med_ndc['ndc'].astype('category')
    # med_ndc.drop(index=med_ndc[med_ndc['ndc'] == '0'].index, axis=0, inplace=True)
    med_ndc.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    med_ndc=med_ndc.reset_index(drop=True)

    with open(ndc2rx_path, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_ndc['RXCUI'] = med_ndc['ndc'].map(ndc2rxnorm)

    rxnorm2atc = pd.read_csv(rx2atc_path)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_ndc.drop(index=med_ndc[med_ndc['RXCUI'].isin([''])].index, axis=0, inplace=True)
    med_ndc.dropna(subset=['RXCUI'],inplace=True)

    med_ndc['RXCUI'] = med_ndc['RXCUI'].astype('int64')
    med_ndc = med_ndc.reset_index(drop=True)
    med_atc = med_ndc.merge(rxnorm2atc, on=['RXCUI'])

    # med_atc.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_atc.drop(columns=['RXCUI', 'ndc'], axis=1, inplace=True)
    med_atc = med_atc.drop_duplicates()
    med_atc.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    med_atc = med_atc.reset_index(drop=True)
    return med_atc

def merge_med_icd(diag_pd,med_atc,medi_med):
    diag_pd.drop(columns=['subject_id','seq_num'],axis=1,inplace=True)
    med_atc.drop(columns=['subject_id','startdate'],axis=1,inplace=True)
    med2diag_nf=med_atc.merge(diag_pd,on='hadm_id',how='left')
    med2diag_nf.drop(columns=['hadm_id'],axis=1,inplace=True)
    med2diag_nf = med2diag_nf.drop_duplicates()
    medi_med = medi_med.rename(columns={'ATC':'ATC4','CODE':'icd9_code'})
    # print(med2diag_nf['ATC4'].unique().shape)
    # print(med2diag_nf['icd9_code'].unique().shape)
    # filter
    med2diag=pd.merge(med2diag_nf,medi_med,how='inner')

    return med2diag

if __name__ == '__main__':
    path_source=dataset_path
    save_path = dataset_path+'{}/processed/'.format(dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    createdatset(path_source,save_path)