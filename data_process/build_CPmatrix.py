import pandas as pd
from collections import defaultdict
from operator import itemgetter
# from data_process.node2embedding import node2emb
import pickle
import os
from data_process.util import dataset_name,embedding_size,dataset_path

def divide_patients(patients_path):
    patients_pd=pd.read_csv(patients_path)
    patients_pd['dob']=pd.to_datetime(patients_pd['dob'], format='%Y-%m-%d %H:%M:%S')
    patients_pd['division'] = 0
    patients_pd.loc[int(0.7*patients_pd.shape[0]):int(0.8*patients_pd.shape[0]), 'division'] = 1
    patients_pd.loc[int(0.8*patients_pd.shape[0]):patients_pd.shape[0], 'division'] = 2
    # valid=patients_pd.sample(int(0.1*patients_pd.shape[0]))
    # patients_pd.loc[valid.index, 'division'] = 1
    # test=patients_pd[patients_pd['division']==0].sample(int(0.2*patients_pd.shape[0]))
    # patients_pd.loc[test.index, 'division'] = 2
    return patients_pd

def read_data(patients_pd,adm_path,diag_path,pres_path,proced_path,med2diag_path):
    adm_pd=pd.read_csv(adm_path)
    gb=adm_pd.groupby(['subject_id'])['hadm_id'].count().reset_index(name='count')
    # 添加列is_readmission，标志patient是否有再入院，用于生成数据集
    patients_pd['is_readmission']=0
    patients_pd.loc[patients_pd[patients_pd['subject_id'].isin(gb[gb['count']>1]['subject_id'])].index,'is_readmission']=1

    # is_lasted:是否patient的最后一次admission。is_edreg:是否急诊留观。age,hospital_expire_flag。
    adm_pd['admittime']=pd.to_datetime(adm_pd['admittime'], format='%Y-%m-%d %H:%M:%S')
    adm_pd.sort_values(by=['subject_id', 'admittime'], inplace=True)
    adm_pd.reset_index(drop=True,inplace=True)
    adm_pd['is_lasted']=1
    adm_pd.loc[adm_pd[(adm_pd.groupby(['subject_id']).cumcount(ascending=False)>0)|(adm_pd['subject_id'].isin(gb[gb['count']==1]['subject_id']))].index,'is_lasted']=0
    adm_pd['is_edreg']='edreg_0'
    adm_pd.loc[adm_pd[~adm_pd['edregtime'].isna()].index,'is_edreg']='edreg_1'
    adm_pd = adm_pd.merge(patients_pd.drop(columns='is_readmission'), on='subject_id', how='left')
    adm_pd['age'] = 0
    adm_pd['age']=pd.DataFrame(adm_pd['admittime'].dt.year-adm_pd['dob'].dt.year)
    adm_pd['age']='age_'+adm_pd['age'].map(lambda x: int(x/10) if 0<=x<90 else int(9)).map(str)
    adm_pd['hospital_expire_flag']= 'end_'+adm_pd['hospital_expire_flag'] .map(str)
    adm_pd.drop(columns=['admittime','edregtime','dob'],inplace=True)

    diag_pd=pd.read_csv(diag_path).sort_values(by=['subject_id', 'seq_num'])
    diag_pd['icd9_code'] = 'diag_' + diag_pd['icd9_code'].map(str)

    pres_pd=pd.read_csv(pres_path)
    pres_pd['startdate'] = pd.to_datetime(pres_pd['startdate'], format='%Y-%m-%d %H:%M:%S')
    pres_pd.sort_values(by=['subject_id', 'startdate'], inplace=True)
    pres_pd.reset_index(drop=True, inplace=True)
    pres_pd['ATC4'] = 'med_' + pres_pd['ATC4'].map(str)

    proced_pd = pd.read_csv(proced_path).sort_values(by=['subject_id', 'seq_num'])
    proced_pd['icd9_code'] ='pro_'+proced_pd['icd9_code'].map(str)

    match_med2diag=pd.read_csv(med2diag_path)
    match_med2diag['ATC4'] = 'med_' + match_med2diag['ATC4'].map(str)
    match_med2diag['icd9_code'] = 'diag_' + match_med2diag['icd9_code'].map(str)

    return adm_pd,diag_pd,pres_pd,proced_pd,match_med2diag

def matrix2graph(adm_pd,diag_pd,pres_pd,proced_pd,match_med2diag,graph_save_path):
    # 删除test中所有拥有多次encounter的patient的最后一次encounter
    adm_delete=adm_pd.drop(index=adm_pd[(adm_pd['is_lasted']==1)&(adm_pd['division']==2)].index)
    # 删除出现次数小于5次的diagnoses code、med code、procedure code
    diag_delete=diag_pd.drop(index=diag_pd[~diag_pd['hadm_id'].isin(adm_delete['hadm_id'])].index)
    # temp=diag_delete.groupby(['icd9_code'])['hadm_id'].count().reset_index(name='count')
    # diag_delete=diag_delete[diag_delete['icd9_code'].isin(temp[temp['count']<5]['icd9_code'])].sort_values(by=['subject_id', 'seq_num'])
    # print(diag_pd['icd9_code'].unique().shape)
    # print(diag_delete['icd9_code'].unique().shape)

    # print(diag_pd['hadm_id'].unique().shape)
    # print(diag_delete['hadm_id'].unique().shape)
    #
    # s=diag_pd.drop(index=diag_pd[diag_pd['hadm_id'].isin(adm_delete['hadm_id'])].index)
    # print(s['hadm_id'].unique().shape)
    # s=s.drop(index=s[~s['icd9_code'].isin(diag_delete['icd9_code'])].index)
    # print(s['hadm_id'].unique().shape)
    pres_delete=pres_pd.drop(index=pres_pd[~pres_pd['hadm_id'].isin(adm_delete['hadm_id'])].index)
    temp=pres_delete.groupby(['ATC4'])['hadm_id'].count().reset_index(name='count')
    pres_delete=pres_delete[pres_delete['ATC4'].isin(temp[temp['count']<5]['ATC4'])]

    proced_delete=proced_pd.drop(index=proced_pd[~proced_pd['hadm_id'].isin(adm_delete['hadm_id'])].index)
    temp=proced_delete.groupby(['icd9_code'])['hadm_id'].count().reset_index(name='count')
    proced_delete=proced_delete[proced_delete['icd9_code'].isin(temp[temp['count']<5]['icd9_code'])]

    # 条件概率矩阵diag2diag,diag2pro....
    #main:保留主diagnoses code。other:保留其他code。合并后去除主code出现在icd9_code_y中的情况。
    diag_main = diag_delete.drop(index=diag_delete[diag_delete['seq_num']!=1].index).drop(columns=['subject_id','seq_num'], axis=1)
    diag_other = diag_delete.drop(index=diag_delete[diag_delete['seq_num']==1].index).drop(columns=['subject_id','seq_num'], axis=1)
    temp=diag_main.merge(diag_other, on='hadm_id',how='left')
    temp.drop(index=temp[temp['icd9_code_y'].isin(temp['icd9_code_x'])].index,inplace=True)
    diag2diag=pd.crosstab(temp['icd9_code_x'], temp['icd9_code_y'], normalize='index')

    temp=diag_delete.drop(columns=['subject_id','seq_num'], axis=1).merge(pres_delete.drop(columns=['subject_id','startdate'], axis=1),on='hadm_id',how='left')
    diag2med = pd.crosstab(temp['icd9_code'], temp['ATC4'], normalize='index')
    # 已匹配的icd和atc不需要出现在概率矩阵中（相当于他们所对应的其他code都置0）
    match_med2diag=match_med2diag.merge(temp[['icd9_code','ATC4']].drop_duplicates(),how='inner')
    dif_columns = list(set(diag2med.columns).difference(set(match_med2diag['ATC4'])))
    diag2med.drop(index=diag2med[diag2med.index.isin(match_med2diag['icd9_code'].drop_duplicates())].index,columns=dif_columns,inplace=True)

    temp=diag_delete.drop(columns=['subject_id','seq_num'], axis=1).merge(proced_delete.drop(columns=['subject_id','seq_num'], axis=1),on='hadm_id',how='left')
    diag2pro=pd.crosstab(temp['icd9_code_x'], temp['icd9_code_y'], normalize='index')

    # test的相关admission不能参与计算,is_readm代表当前admission后是否会再次入院
    # 条件概率矩阵code-result & covariant-result
    temp=adm_delete.drop(index=adm_delete[(adm_delete.groupby(['subject_id']).cumcount(ascending=False)==0)&(adm_delete['division']==2)].index)
    temp['is_readm']=0
    temp.loc[:,'is_readm'] = 1-temp['is_lasted']
    temp['is_readm']='readm_'+temp['is_readm'].astype(str)
    temp_diag2readm=temp.drop(columns=['subject_id','is_lasted','hospital_expire_flag','is_edreg','gender','division','age'])\
        .merge(diag_delete.drop(columns=['subject_id','seq_num'], axis=1),on='hadm_id',how='left')
    diag2readm=pd.crosstab(temp_diag2readm['icd9_code'], temp_diag2readm['is_readm'],normalize='index')

    temp_med2readm= temp.drop(columns=['subject_id', 'is_lasted','hospital_expire_flag','is_edreg','gender','division','age'])\
        .merge(pres_delete.drop(columns=['subject_id','startdate'], axis=1), on='hadm_id', how='left')
    med2readm=pd.crosstab(temp_med2readm['ATC4'], temp_med2readm['is_readm'],normalize='index')

    temp_pro2readm = temp.drop(columns=['subject_id',  'is_lasted','hospital_expire_flag','is_edreg','gender','division','age'])\
        .merge(proced_delete.drop(columns=['subject_id','seq_num'], axis=1), on='hadm_id', how='left')
    pro2readm=pd.crosstab(temp_pro2readm['icd9_code'], temp_pro2readm['is_readm'],normalize='index')

    cov_age2readm=pd.crosstab(temp['age'], temp['is_readm'],normalize='index')
    cov_gen2readm = pd.crosstab(temp['gender'], temp['is_readm'], normalize='index')
    cov_edreg2readm = pd.crosstab(temp['is_edreg'], temp['is_readm'], normalize='index')

    temp_diag2end=adm_delete.drop(columns=['subject_id','is_lasted','is_edreg','gender','division','age'])\
        .merge(diag_delete.drop(columns=['subject_id','seq_num'], axis=1),on='hadm_id',how='left')
    diag2end=pd.crosstab(temp_diag2end['icd9_code'], temp_diag2end['hospital_expire_flag'],normalize='index')

    temp_med2end=adm_delete.drop(columns=['subject_id','is_lasted','is_edreg','gender','division','age']) \
        .merge(pres_delete.drop(columns=['subject_id', 'startdate'], axis=1), on='hadm_id', how='left')
    med2end=pd.crosstab(temp_med2end['ATC4'], temp_med2end['hospital_expire_flag'],normalize='index')

    temp_pro2end=adm_delete.drop(columns=['subject_id','is_lasted','is_edreg','gender','division','age']) \
        .merge(proced_delete.drop(columns=['subject_id', 'seq_num'], axis=1), on='hadm_id', how='left')
    pro2end=pd.crosstab(temp_pro2end['icd9_code'], temp_pro2end['hospital_expire_flag'],normalize='index')

    cov_age2end=pd.crosstab(adm_delete['age'], adm_delete['hospital_expire_flag'],normalize='index')
    cov_gen2end = pd.crosstab(adm_delete['gender'], adm_delete['hospital_expire_flag'], normalize='index')
    cov_edreg2end = pd.crosstab(adm_delete['is_edreg'], adm_delete['hospital_expire_flag'], normalize='index')

    # build graph
    print("build graph")
    matrix_list=[]
    # graph_edge_all=[]
    matrix_list.append(diag2diag)
    matrix_list.append(diag2med)
    matrix_list.append(diag2pro)
    matrix_list.append(diag2readm)
    matrix_list.append(med2readm)
    matrix_list.append(pro2readm)
    matrix_list.append(cov_age2readm)
    matrix_list.append(cov_gen2readm)
    matrix_list.append(cov_edreg2readm)
    matrix_list.append(diag2end)
    matrix_list.append(med2end)
    matrix_list.append(pro2end)
    matrix_list.append(cov_age2end)
    matrix_list.append(cov_gen2end)
    matrix_list.append(cov_edreg2end)
    graph_edge_all=matrix2edge(matrix_list)
    graph_edge_all=graph_edge_all.reset_index(drop=True)
    graph_edge_all.columns=['code1','code2','weight']
    graph_edge_all.to_csv(graph_save_path, index=False)

    return adm_pd,diag_pd,diag_delete,pres_delete,proced_delete,graph_edge_all

def matrix2edge(matrix_list):
    graph_edge_all=pd.DataFrame()
    # j=0
    for matrix in matrix_list:
        # j += 1
        # print(j)
        edge = []
        for index, row in matrix.iterrows():
            for i in range(len(row)):
                if row[i] > 0:
                    edge.append([index, matrix.columns[i], row[i]])
        edge=pd.DataFrame(edge)
        graph_edge_all=graph_edge_all.append(edge)
    return graph_edge_all

# create dataset: training，validation，test
def create_input(adm_pd,diag_pd,diag_delete,pres_delete,proced_delete,graph,input_save_path):
    node_code=pd.DataFrame(graph['code1'].drop_duplicates()).rename(columns={'code1':'code'}).merge(pd.DataFrame(graph['code2'].drop_duplicates()).rename(columns={'code2':'code'}),on='code',how='outer')
    list1= node_code[node_code['code'].str.contains('diag_')]['code'].to_list()
    list2 = node_code[node_code['code'].str.contains('med_')]['code'].to_list()
    list3 = node_code[node_code['code'].str.contains('pro_')]['code'].to_list()
    dict_diag=dict(zip(list1,list(range(0,len(list1)))))
    dict_med=dict(zip(list2,list(range(0,len(list2)))))
    dict_pro=dict(zip(list3,list(range(0,len(list3)))))
    dict_age={'age_0': 0, 'age_1': 1, 'age_2': 2, 'age_3': 3, 'age_4': 4, 'age_5': 5, 'age_6': 6, 'age_7': 7, 'age_8': 8, 'age_9': 9}
    dict_gender={'F':0,'M':1}
    dict_edreg={'edreg_0':0,'edreg_1':1}
    dict_readm = {'readm_0': 0, 'readm_1': 1}
    dict_end = {'end_0': 0, 'end_1': 1}

    # training set######################################################################################################
    ####################################################################################################################
    adm_train=adm_pd[(adm_pd['division']==0)]
    subject_train=defaultdict(list)
    diag_train=defaultdict(list)
    pres_train = defaultdict(list)
    pro_train = defaultdict(list)
    age_train=defaultdict(list)
    gender_train = defaultdict(list)
    is_edreg_train = defaultdict(list)

    for key,value in adm_train[adm_train['is_lasted'] == 0].groupby(['subject_id']):
        subject_train[key].append(value['hadm_id'].to_list())
    temp=adm_train[adm_train['is_lasted'] == 0]['hadm_id'].drop_duplicates()
    diag_temp=diag_delete[diag_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pres_temp = pres_delete[pres_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pro_temp = proced_delete[proced_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    for key,value in diag_temp:
        diag_train[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))
    for key,value in pres_temp:
        pres_train[key].append(itemgetter(*value['ATC4'].to_list())(dict_med))
    for key,value in pro_temp:
        pro_train[key].append(itemgetter(*value['icd9_code'].to_list())(dict_pro))

    temp=adm_train[adm_train['is_lasted'] == 0].groupby(['hadm_id'])
    for key,value in temp:
        age_train[key].append(itemgetter(*value['age'].to_list())(dict_age))
    for key,value in temp:
        gender_train[key].append(itemgetter(*value['gender'].to_list())(dict_gender))
    for key,value in temp:
        is_edreg_train[key].append(itemgetter(*value['is_edreg'].to_list())(dict_edreg))

    result_train_readm=defaultdict(list)
    temp=adm_train.groupby(['subject_id'])['hadm_id'].count().reset_index(name='count')
    for index,row in temp.iterrows():
        if row['count']==1:
            result_train_readm[row['subject_id']].append(dict_readm['readm_0'])
        else :
            result_train_readm[row['subject_id']].append(dict_readm['readm_1'])

    result_train_end=defaultdict(list)
    temp=pd.DataFrame(adm_train[(adm_train['is_lasted'] == 0)&(adm_train['hospital_expire_flag']=='end_1')]['subject_id'].drop_duplicates())
    temp2=pd.DataFrame(adm_train[~adm_train['subject_id'].isin(temp['subject_id'])]['subject_id'].drop_duplicates())
    for index,row in temp.iterrows():
        result_train_end[row['subject_id']].append(dict_end['end_1'])
    for index,row in temp2.iterrows():
        result_train_end[row['subject_id']].append(dict_end['end_0'])

    result_train_diag = defaultdict(list)
    temp = adm_train[adm_train['is_lasted'] == 1]['hadm_id'].drop_duplicates()
    diag_temp = diag_delete[diag_delete['hadm_id'].isin(temp)].groupby(['subject_id'])
    for key,value in diag_temp:
        result_train_diag[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))

    # validation set####################################################################################################
    ####################################################################################################################
    adm_valid=adm_pd[adm_pd['division']==1]
    subject_valid=defaultdict(list)
    diag_valid=defaultdict(list)
    pres_valid = defaultdict(list)
    pro_valid = defaultdict(list)
    age_valid=defaultdict(list)
    gender_valid = defaultdict(list)
    is_edreg_valid = defaultdict(list)

    for key,value in adm_valid[adm_valid['is_lasted'] == 0].groupby(['subject_id']):
        subject_valid[key].append(value['hadm_id'].to_list())

    temp=adm_valid[adm_valid['is_lasted'] == 0]['hadm_id'].drop_duplicates()
    diag_temp=diag_delete[diag_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pres_temp = pres_delete[pres_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pro_temp = proced_delete[proced_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    for key,value in diag_temp:
        diag_valid[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))
    for key,value in pres_temp:
        pres_valid[key].append(itemgetter(*value['ATC4'].to_list())(dict_med))
    for key,value in pro_temp:
        pro_valid[key].append(itemgetter(*value['icd9_code'].to_list())(dict_pro))

    temp=adm_valid[adm_valid['is_lasted'] == 0].groupby(['hadm_id'])
    for key,value in temp:
        age_valid[key].append(itemgetter(*value['age'].to_list())(dict_age))
    for key,value in temp:
        gender_valid[key].append(itemgetter(*value['gender'].to_list())(dict_gender))
    for key,value in temp:
        is_edreg_valid[key].append(itemgetter(*value['is_edreg'].to_list())(dict_edreg))

    result_valid_readm=defaultdict(list)
    temp=adm_valid.groupby(['subject_id'])['hadm_id'].count().reset_index(name='count')
    for index,row in temp.iterrows():
        if row['count']==1:
            result_valid_readm[row['subject_id']].append(dict_readm['readm_0'])
        else :
            result_valid_readm[row['subject_id']].append(dict_readm['readm_1'])

    result_valid_end=defaultdict(list)
    temp=pd.DataFrame(adm_valid[(adm_valid['is_lasted'] == 0)&(adm_valid['hospital_expire_flag']=='end_1')]['subject_id'].drop_duplicates())
    temp2=pd.DataFrame(adm_valid[~adm_valid['subject_id'].isin(temp['subject_id'])]['subject_id'].drop_duplicates())
    for index,row in temp.iterrows():
        result_valid_end[row['subject_id']].append(dict_end['end_1'])
    for index,row in temp2.iterrows():
        result_valid_end[row['subject_id']].append(dict_end['end_0'])

    result_valid_diag = defaultdict(list)
    temp = adm_valid[adm_valid['is_lasted'] == 1]['hadm_id'].drop_duplicates()
    diag_temp = diag_delete[diag_delete['hadm_id'].isin(temp)].groupby(['subject_id'])
    for key,value in diag_temp:
        result_valid_diag[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))

    # test set####################################################################################################
    ####################################################################################################################
    adm_test=adm_pd[adm_pd['division']==2]
    subject_test=defaultdict(list)
    diag_test=defaultdict(list)
    pres_test = defaultdict(list)
    pro_test= defaultdict(list)
    age_test=defaultdict(list)
    gender_test = defaultdict(list)
    is_edreg_test = defaultdict(list)

    for key,value in adm_test[adm_test['is_lasted'] == 0].groupby(['subject_id']):
        subject_test[key].append(value['hadm_id'].to_list())

    temp=adm_test[adm_test['is_lasted'] == 0]['hadm_id'].drop_duplicates()
    diag_temp=diag_delete[diag_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pres_temp = pres_delete[pres_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    pro_temp = proced_delete[proced_delete['hadm_id'].isin(temp)].groupby(['hadm_id'])
    for key,value in diag_temp:
        diag_test[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))
    for key,value in pres_temp:
        pres_test[key].append(itemgetter(*value['ATC4'].to_list())(dict_med))
    for key,value in pro_temp:
        pro_test[key].append(itemgetter(*value['icd9_code'].to_list())(dict_pro))

    temp=adm_test[adm_test['is_lasted'] == 0].groupby(['hadm_id'])
    for key,value in temp:
        age_test[key].append(itemgetter(*value['age'].to_list())(dict_age))
    for key,value in temp:
        gender_test[key].append(itemgetter(*value['gender'].to_list())(dict_gender))
    for key,value in temp:
        is_edreg_test[key].append(itemgetter(*value['is_edreg'].to_list())(dict_edreg))

    result_test_readm=defaultdict(list)
    temp=adm_test.groupby(['subject_id'])['hadm_id'].count().reset_index(name='count')
    for index,row in temp.iterrows():
        if row['count']==1:
            result_test_readm[row['subject_id']].append(dict_readm['readm_0'])
        else :
            result_test_readm[row['subject_id']].append(dict_readm['readm_1'])

    result_test_end=defaultdict(list)
    temp=pd.DataFrame(adm_test[(adm_test['is_lasted'] == 0)&(adm_test['hospital_expire_flag']=='end_1')]['subject_id'].drop_duplicates())
    temp2=pd.DataFrame(adm_test[~adm_test['subject_id'].isin(temp['subject_id'])]['subject_id'].drop_duplicates())
    for index,row in temp.iterrows():
        result_test_end[row['subject_id']].append(dict_end['end_1'])
    for index,row in temp2.iterrows():
        result_test_end[row['subject_id']].append(dict_end['end_0'])

    result_test_diag = defaultdict(list)
    temp = adm_test[adm_test['is_lasted'] == 1]['hadm_id'].drop_duplicates()
    diag_temp = diag_pd[diag_pd['hadm_id'].isin(temp)]
    diag_temp=diag_temp.drop(index=diag_temp[~diag_temp['icd9_code'].isin(diag_delete['icd9_code'].drop_duplicates())].index).groupby(['subject_id'])
    for key,value in diag_temp:
        result_test_diag[key].append(itemgetter(*value['icd9_code'].to_list())(dict_diag))

    #######################################create dataset###############################################################
    dict_list=[dict_diag,dict_med,dict_pro,dict_age,dict_gender,dict_edreg,dict_readm,dict_end]
    f = open(input_save_path + 'dict_list.pkl', 'wb')
    pickle.dump(dict_list,f,0)
    f.close()

    train = [subject_train, diag_train, pres_train, pro_train, age_train, gender_train, is_edreg_train,result_train_readm, result_train_end, result_train_diag]
    f = open(input_save_path + 'train.pkl', 'wb')
    pickle.dump(train, f, 0)
    f.close()

    valid=[subject_valid,diag_valid,pres_valid,pro_valid,age_valid,gender_valid,is_edreg_valid,result_valid_readm,result_valid_end,result_valid_diag]
    f = open(input_save_path + 'valid.pkl', 'wb')
    pickle.dump(valid,f,0)
    f.close()

    test=[subject_test,diag_test,pres_test,pro_test,age_test,gender_test,is_edreg_test,result_test_readm,result_test_end,result_test_diag]
    f = open(input_save_path + 'test.pkl', 'wb')
    pickle.dump(test,f,0)
    f.close()

if __name__ == '__main__':

    patients_path=dataset_path+'{}/processed/patients.csv'.format(dataset_name)
    adm_path = dataset_path+'{}/processed/admissions.csv'.format(dataset_name)
    diag_path = dataset_path+'{}/processed/diagnoses_icd.csv'.format(dataset_name)
    pres_path = dataset_path+'{}/processed/prescriptions.csv'.format(dataset_name)
    med2diag_path=dataset_path+'{}/processed/med2diag.csv'.format(dataset_name)
    proced_path = dataset_path+'{}/processed/procedures_icd.csv'.format(dataset_name)
    graph_save_path=dataset_path+'{}/graph/'.format(dataset_name)
    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)
    save_path_input = dataset_path+'{}/input/'.format(dataset_name)
    if not os.path.exists(save_path_input):
        os.makedirs(save_path_input)

    print('build matrix')
    patients_pd=divide_patients(patients_path)
    adm_pd, diag_pd, pres_pd, proced_pd,match_med2diag=read_data(patients_pd,adm_path,diag_path,pres_path,proced_path,med2diag_path)
    graph_file_name='graph.csv'
    adm_pd, diag_pd, diag_delete, pres_delete, proced_delete, graph_edge_all=matrix2graph(adm_pd, diag_pd, pres_pd, proced_pd,match_med2diag,graph_save_path+graph_file_name)
    print("node2vec")
    file_name='node_embedding_'+str(embedding_size)
    node2emb(graph_edge_all,embedding_size,save_path_input+file_name)
    print("create dataset")
    # create_input(adm_pd,diag_pd,diag_delete,pres_delete,proced_delete,graph_edge_all,save_path_input)
    print(len(diag_pd['icd9_code'].unique()),len(pres_pd['ATC4'].unique()),len(proced_pd['icd9_code'].unique()))