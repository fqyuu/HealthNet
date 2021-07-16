import numpy as np
import pickle
from data_process.util import dataset_name,batch_size,task,dataset_path
from collections import namedtuple
from operator import itemgetter

HParams=namedtuple("HParams","mode,task,batch_size,num_diag,num_encounter,embedding_size,diag_step,med_step,pro_step,learning_rate")

def get_input_batch(mode,embedding_path,dict_list_path,input_path):
    hps=[]
    embedding_list=[]
    input_list=[]
    if task == 'readm':
        hps, embedding_list, input_list=get_input_batch_readm(mode,embedding_path,dict_list_path,input_path)
    if task == 'end':
        hps, embedding_list, input_list=get_input_batch_end(mode,embedding_path,dict_list_path,input_path)
    if task == 'diag':
        hps, embedding_list, input_list=get_input_batch_diag(mode,embedding_path,dict_list_path,input_path)
    return hps,embedding_list,input_list

def get_input_batch_readm(mode,embedding_path,dict_list_path,input_path):
    f= open(dict_list_path,'rb')
    dict_list=pickle.load(f)
    f.close()
    dict_diag,dict_med,dict_pro,dict_age, dict_gender, dict_edreg, dict_readm, _=dict_list

    f=open(embedding_path,'r')
    node_num,embedding_size=str.split(f.readline())
    embedding_size=int(embedding_size)
    embedding_diag = np.zeros(shape=[len(dict_diag),embedding_size],dtype=np.float32)
    embedding_med = np.zeros(shape=[len(dict_med),embedding_size],dtype=np.float32)
    embedding_pro = np.zeros(shape=[len(dict_pro),embedding_size],dtype=np.float32)
    embedding_age = np.zeros(shape=[len(dict_age), embedding_size], dtype=np.float32)
    embedding_gender = np.zeros(shape=[len(dict_gender), embedding_size], dtype=np.float32)
    embedding_edreg = np.zeros(shape=[len(dict_edreg), embedding_size], dtype=np.float32)
    embedding_readm = np.zeros(shape=[len(dict_readm), embedding_size], dtype=np.float32)
    for row in f:
        row_sl=str.split(row)
        if row_sl[0] in dict_diag:
            embedding_diag[dict_diag[row_sl[0]]]=row_sl[1:]
        if row_sl[0] in dict_med:
            embedding_med[dict_med[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_pro:
            embedding_pro[dict_pro[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_age:
            embedding_age[dict_age[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_gender:
            embedding_gender[dict_gender[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_edreg:
            embedding_edreg[dict_edreg[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_readm:
            embedding_readm[dict_readm[row_sl[0]]] = row_sl[1:]
    embedding_list = [embedding_diag, embedding_med, embedding_pro, embedding_age, embedding_gender,
                        embedding_edreg, embedding_readm]
    f.close()

    f = open(input_path, 'rb')
    subject, diag, med, pro, age, gender, is_edreg, result_readm, _, _= pickle.load(f)
    f.close()
    num_diag = len(dict_diag)
    num_encounter=1
    diag_step=1
    med_step=1
    pro_step=1
    # print(len(subject))
    for i in range (int(len(subject)/batch_size)):
        temp=list(subject.values())[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            if len(temp[j][0]) > num_encounter:
                num_encounter = len(temp[j][0])
            for k in range(len(hadm_list)):
                # 说明有值
                if len(diag[hadm_list[k]]) > 0:
                    # 值不是int类型，说明是个长度大于一的列表
                    if not isinstance(diag[hadm_list[k]][0], int):
                        if len(diag[hadm_list[k]][0]) > diag_step:
                            diag_step = len(diag[hadm_list[k]][0])
                if len(med[hadm_list[k]])>0:
                    if not isinstance(med[hadm_list[k]][0], int):
                        if len(med[hadm_list[k]][0])>med_step:
                            med_step=len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if not isinstance(pro[hadm_list[k]][0], int):
                        if len(pro[hadm_list[k]][0]) > pro_step:
                            pro_step = len(pro[hadm_list[k]][0])

    real_lenth_encounter_list=[]
    diag_input_list=[]
    med_input_list = []
    pro_input_list = []
    age_input_list=[]
    gender_input_list=[]
    is_edreg_input_list=[]
    real_lenth_med_list = []
    real_lenth_pro_list = []
    result_input_list = []
    for i in range (int(len(subject)/batch_size)):
        real_lenth_encounter = np.zeros(shape=[batch_size], dtype=np.int32)
        diag_input = np.zeros(shape=[batch_size, num_encounter, num_diag], dtype=np.int32)
        med_input = np.zeros(shape=[batch_size, num_encounter, med_step], dtype=np.int32)
        pro_input = np.zeros(shape=[batch_size, num_encounter, pro_step], dtype=np.int32)
        age_input=np.zeros(shape=[batch_size,num_encounter], dtype=np.int32)
        gender_input = np.zeros(shape=[batch_size], dtype=np.int32)
        is_edreg_input = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_med = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_pro = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        result_input = np.zeros(shape=[batch_size], dtype=np.int32)
        temp=list(subject.values())[i*batch_size:batch_size*(i+1)]
        subject_batch=list(subject.keys())[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            real_lenth_encounter[j]=len(hadm_list)
            if len(hadm_list) > num_encounter:
                num_encounter = len(hadm_list)
            for k in range(len(hadm_list)):
                if len(diag[hadm_list[k]]) > 0:
                    if isinstance(diag[hadm_list[k]][0], int):
                        diag_input[j,k,diag[hadm_list[k]][0]]=1
                    else:
                        diag_input[j,k,list(diag[hadm_list[k]][0])]=1
                if len(med[hadm_list[k]])>0:
                    if isinstance(med[hadm_list[k]][0], int):
                        med_input[j,k,0]=med[hadm_list[k]][0]
                        real_lenth_med[j, k] = 1
                    else :
                        med_input[j,k]=np.array(list(med[hadm_list[k]][0])+[0]*(med_step-len(med[hadm_list[k]][0])))
                        real_lenth_med[j, k] = len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if isinstance(pro[hadm_list[k]][0], int):
                        pro_input[j,k,0]=pro[hadm_list[k]][0]
                        real_lenth_pro[j, k] = 1
                    else :
                        pro_input[j,k]=np.array(list(pro[hadm_list[k]][0])+[0]*(pro_step-len(pro[hadm_list[k]][0])))
                        real_lenth_pro[j, k] = len(pro[hadm_list[k]][0])
                if len(age[hadm_list[k]])==1:
                    age_input[j,k]=age[hadm_list[k]][0]
                if len(is_edreg[hadm_list[k]])==1:
                    is_edreg_input[j,k]=is_edreg[hadm_list[k]][0]
            gender_input[j]=gender[hadm_list[0]][0]
            # result
            result_input[j]=result_readm[subject_batch[j]][0]

        real_lenth_encounter_list.append(real_lenth_encounter)
        diag_input_list.append(diag_input)
        med_input_list.append(med_input)
        pro_input_list.append(pro_input)
        age_input_list.append(age_input)
        gender_input_list.append(gender_input)
        is_edreg_input_list.append(is_edreg_input)
        real_lenth_med_list.append(real_lenth_med)
        real_lenth_pro_list.append(real_lenth_pro)
        result_input_list.append(result_input)

    hps = HParams(
        mode=mode,
        task=task,
        batch_size=batch_size,
        num_diag=num_diag,
        embedding_size=embedding_size,
        num_encounter=num_encounter,
        diag_step=diag_step,
        med_step=med_step,
        pro_step=pro_step,
        learning_rate=0.0001,
    )
    input_list=[real_lenth_encounter_list,diag_input_list,med_input_list,real_lenth_med_list,pro_input_list,real_lenth_pro_list,age_input_list,gender_input_list,is_edreg_input_list,result_input_list]
    return hps,embedding_list,input_list

def get_input_batch_end(mode,embedding_path,dict_list_path,input_path):
    f= open(dict_list_path,'rb')
    dict_list=pickle.load(f)
    f.close()
    dict_diag,dict_med,dict_pro,dict_age, dict_gender, dict_edreg, _, dict_end=dict_list

    f=open(embedding_path,'r')
    node_num,embedding_size=str.split(f.readline())
    embedding_size=int(embedding_size)
    embedding_diag = np.zeros(shape=[len(dict_diag),embedding_size],dtype=np.float32)
    embedding_med = np.zeros(shape=[len(dict_med),embedding_size],dtype=np.float32)
    embedding_pro = np.zeros(shape=[len(dict_pro),embedding_size],dtype=np.float32)
    embedding_age = np.zeros(shape=[len(dict_age), embedding_size], dtype=np.float32)
    embedding_gender = np.zeros(shape=[len(dict_gender), embedding_size], dtype=np.float32)
    embedding_edreg = np.zeros(shape=[len(dict_edreg), embedding_size], dtype=np.float32)
    embedding_end = np.zeros(shape=[len(dict_end), embedding_size], dtype=np.float32)
    for row in f:
        row_sl=str.split(row)
        if row_sl[0] in dict_diag:
            embedding_diag[dict_diag[row_sl[0]]]=row_sl[1:]
        if row_sl[0] in dict_med:
            embedding_med[dict_med[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_pro:
            embedding_pro[dict_pro[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_age:
            embedding_age[dict_age[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_gender:
            embedding_gender[dict_gender[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_edreg:
            embedding_edreg[dict_edreg[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_end:
            embedding_end[dict_end[row_sl[0]]] = row_sl[1:]

    embedding_list = [embedding_diag, embedding_med, embedding_pro, embedding_age, embedding_gender,
                    embedding_edreg, embedding_end]
    f.close()

    f = open(input_path, 'rb')
    subject, diag, med, pro, age, gender, is_edreg, _, result_end, _= pickle.load(f)
    f.close()
    num_diag = len(dict_diag)
    num_encounter=1
    diag_step=1
    med_step=1
    pro_step=1
    for i in range (int(len(subject)/batch_size)):
        temp=list(subject.values())[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            if len(temp[j][0]) > num_encounter:
                num_encounter = len(temp[j][0])
            for k in range(len(hadm_list)):
                # 说明有值
                if len(diag[hadm_list[k]]) > 0:
                    # 值不是int类型，说明是个长度大于一的列表
                    if not isinstance(diag[hadm_list[k]][0], int):
                        if len(diag[hadm_list[k]][0]) > diag_step:
                            diag_step = len(diag[hadm_list[k]][0])
                if len(med[hadm_list[k]])>0:
                    if not isinstance(med[hadm_list[k]][0], int):
                        if len(med[hadm_list[k]][0])>med_step:
                            med_step=len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if not isinstance(pro[hadm_list[k]][0], int):
                        if len(pro[hadm_list[k]][0]) > pro_step:
                            pro_step = len(pro[hadm_list[k]][0])

    real_lenth_encounter_list=[]
    diag_input_list=[]
    med_input_list = []
    pro_input_list = []
    age_input_list=[]
    gender_input_list=[]
    is_edreg_input_list=[]
    real_lenth_med_list = []
    real_lenth_pro_list = []
    result_input_list = []
    for i in range (int(len(subject)/batch_size)):
        real_lenth_encounter = np.zeros(shape=[batch_size], dtype=np.int32)
        diag_input = np.zeros(shape=[batch_size, num_encounter, num_diag], dtype=np.int32)
        med_input = np.zeros(shape=[batch_size, num_encounter, med_step], dtype=np.int32)
        pro_input = np.zeros(shape=[batch_size, num_encounter, pro_step], dtype=np.int32)
        age_input=np.zeros(shape=[batch_size,num_encounter], dtype=np.int32)
        gender_input = np.zeros(shape=[batch_size], dtype=np.int32)
        is_edreg_input = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_med = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_pro = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        result_input = np.zeros(shape=[batch_size], dtype=np.int32)
        temp=list(subject.values())[i*batch_size:batch_size*(i+1)]
        subject_batch=list(subject.keys())[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            real_lenth_encounter[j]=len(hadm_list)
            if len(hadm_list) > num_encounter:
                num_encounter = len(hadm_list)
            for k in range(len(hadm_list)):
                if len(diag[hadm_list[k]]) > 0:
                    if isinstance(diag[hadm_list[k]][0], int):
                        diag_input[j,k,diag[hadm_list[k]][0]]=1
                    else:
                        diag_input[j,k,list(diag[hadm_list[k]][0])]=1
                if len(med[hadm_list[k]])>0:
                    if isinstance(med[hadm_list[k]][0], int):
                        med_input[j,k,0]=med[hadm_list[k]][0]
                        real_lenth_med[j, k] = 1
                    else :
                        med_input[j,k]=np.array(list(med[hadm_list[k]][0])+[0]*(med_step-len(med[hadm_list[k]][0])))
                        real_lenth_med[j, k] = len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if isinstance(pro[hadm_list[k]][0], int):
                        pro_input[j,k,0]=pro[hadm_list[k]][0]
                        real_lenth_pro[j, k] = 1
                    else :
                        pro_input[j,k]=np.array(list(pro[hadm_list[k]][0])+[0]*(pro_step-len(pro[hadm_list[k]][0])))
                        real_lenth_pro[j, k] = len(pro[hadm_list[k]][0])
                if len(age[hadm_list[k]])==1:
                    age_input[j,k]=age[hadm_list[k]][0]
                if len(is_edreg[hadm_list[k]])==1:
                    is_edreg_input[j,k]=is_edreg[hadm_list[k]][0]
            gender_input[j]=gender[hadm_list[0]][0]
            # result
            result_input[j]=result_end[subject_batch[j]][0]

        real_lenth_encounter_list.append(real_lenth_encounter)
        diag_input_list.append(diag_input)
        med_input_list.append(med_input)
        pro_input_list.append(pro_input)
        age_input_list.append(age_input)
        gender_input_list.append(gender_input)
        is_edreg_input_list.append(is_edreg_input)
        real_lenth_med_list.append(real_lenth_med)
        real_lenth_pro_list.append(real_lenth_pro)
        result_input_list.append(result_input)
    hps = HParams(
        mode=mode,
        task=task,
        batch_size=batch_size,
        num_diag=num_diag,
        embedding_size=embedding_size,
        num_encounter=num_encounter,
        diag_step=diag_step,
        med_step=med_step,
        pro_step=pro_step,
        learning_rate=0.0001,
    )
    input_list=[real_lenth_encounter_list,diag_input_list,med_input_list,real_lenth_med_list,pro_input_list,real_lenth_pro_list,age_input_list,gender_input_list,is_edreg_input_list,result_input_list]
    return hps,embedding_list,input_list

def get_input_batch_diag(mode,embedding_path,dict_list_path,input_path):
    f= open(dict_list_path,'rb')
    dict_list=pickle.load(f)
    f.close()
    dict_diag,dict_med,dict_pro,dict_age, dict_gender, dict_edreg, _, _=dict_list

    f=open(embedding_path,'r')
    node_num,embedding_size=str.split(f.readline())
    embedding_size=int(embedding_size)
    embedding_diag = np.zeros(shape=[len(dict_diag),embedding_size],dtype=np.float32)
    embedding_med = np.zeros(shape=[len(dict_med),embedding_size],dtype=np.float32)
    embedding_pro = np.zeros(shape=[len(dict_pro),embedding_size],dtype=np.float32)
    embedding_age = np.zeros(shape=[len(dict_age), embedding_size], dtype=np.float32)
    embedding_gender = np.zeros(shape=[len(dict_gender), embedding_size], dtype=np.float32)
    embedding_edreg = np.zeros(shape=[len(dict_edreg), embedding_size], dtype=np.float32)
    for row in f:
        row_sl=str.split(row)
        if row_sl[0] in dict_diag:
            embedding_diag[dict_diag[row_sl[0]]]=row_sl[1:]
        if row_sl[0] in dict_med:
            embedding_med[dict_med[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_pro:
            embedding_pro[dict_pro[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_age:
            embedding_age[dict_age[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_gender:
            embedding_gender[dict_gender[row_sl[0]]] = row_sl[1:]
        if row_sl[0] in dict_edreg:
            embedding_edreg[dict_edreg[row_sl[0]]] = row_sl[1:]

    embedding_list = [embedding_diag, embedding_med, embedding_pro, embedding_age, embedding_gender,
                      embedding_edreg]
    f.close()

    f = open(input_path, 'rb')
    subject, diag, med, pro, age, gender, is_edreg, _, _, result_diag= pickle.load(f)
    f.close()
    subject_list=list(result_diag.keys())
    subject=itemgetter(*subject_list)(subject)
    num_diag=len(dict_diag)
    num_encounter=1
    diag_step=1
    med_step=1
    pro_step=1
    # print(len(subject_list))
    for i in range (int(len(subject_list)/batch_size)):
        temp=subject[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            if len(temp[j][0]) > num_encounter:
                num_encounter = len(temp[j][0])
            for k in range(len(hadm_list)):
                # 说明有值
                if len(diag[hadm_list[k]]) > 0:
                    # 值不是int类型，说明是个长度大于一的列表
                    if not isinstance(diag[hadm_list[k]][0], int):
                        if len(diag[hadm_list[k]][0]) > diag_step:
                            diag_step = len(diag[hadm_list[k]][0])
                if len(med[hadm_list[k]])>0:
                    if not isinstance(med[hadm_list[k]][0], int):
                        if len(med[hadm_list[k]][0])>med_step:
                            med_step=len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if not isinstance(pro[hadm_list[k]][0], int):
                        if len(pro[hadm_list[k]][0]) > pro_step:
                            pro_step = len(pro[hadm_list[k]][0])

    real_lenth_encounter_list=[]
    diag_input_list=[]
    med_input_list = []
    pro_input_list = []
    age_input_list=[]
    gender_input_list=[]
    is_edreg_input_list=[]
    real_lenth_med_list = []
    real_lenth_pro_list = []
    result_input_list = []
    for i in range (int(len(subject_list)/batch_size)):
        real_lenth_encounter = np.zeros(shape=[batch_size], dtype=np.int32)
        diag_input = np.zeros(shape=[batch_size, num_encounter, num_diag], dtype=np.int32)
        med_input = np.zeros(shape=[batch_size, num_encounter, med_step], dtype=np.int32)
        pro_input = np.zeros(shape=[batch_size, num_encounter, pro_step], dtype=np.int32)
        age_input=np.zeros(shape=[batch_size,num_encounter], dtype=np.int32)
        gender_input = np.zeros(shape=[batch_size], dtype=np.int32)
        is_edreg_input = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_med = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        real_lenth_pro = np.zeros(shape=[batch_size, num_encounter], dtype=np.int32)
        result_input = np.zeros(shape=[batch_size,len(dict_diag)], dtype=np.int32)
        temp=subject[i*batch_size:batch_size*(i+1)]
        subject_batch=subject_list[i*batch_size:batch_size*(i+1)]
        for j in range(batch_size):
            hadm_list = temp[j][0]
            real_lenth_encounter[j]=len(hadm_list)
            if len(hadm_list) > num_encounter:
                num_encounter = len(hadm_list)
            for k in range(len(hadm_list)):
                if len(diag[hadm_list[k]]) > 0:
                    if isinstance(diag[hadm_list[k]][0], int):
                        diag_input[j,k,diag[hadm_list[k]][0]]=1
                    else:
                        diag_input[j,k,list(diag[hadm_list[k]][0])]=1
                if len(med[hadm_list[k]])>0:
                    if isinstance(med[hadm_list[k]][0], int):
                        med_input[j,k,0]=med[hadm_list[k]][0]
                        real_lenth_med[j, k] = 1
                    else :
                        med_input[j,k]=np.array(list(med[hadm_list[k]][0])+[0]*(med_step-len(med[hadm_list[k]][0])))
                        real_lenth_med[j, k] = len(med[hadm_list[k]][0])
                if len(pro[hadm_list[k]]) > 0:
                    if isinstance(pro[hadm_list[k]][0], int):
                        pro_input[j,k,0]=pro[hadm_list[k]][0]
                        real_lenth_pro[j, k] = 1
                    else :
                        pro_input[j,k]=np.array(list(pro[hadm_list[k]][0])+[0]*(pro_step-len(pro[hadm_list[k]][0])))
                        real_lenth_pro[j, k] = len(pro[hadm_list[k]][0])
                if len(age[hadm_list[k]])==1:
                    age_input[j,k]=age[hadm_list[k]][0]
                if len(is_edreg[hadm_list[k]])==1:
                    is_edreg_input[j,k]=is_edreg[hadm_list[k]][0]
            gender_input[j]=gender[hadm_list[0]][0]
            # result

            if len(result_diag[subject_batch[j]]) > 0:
                # real_result_diagnosis[j]=1
                if isinstance(result_diag[subject_batch[j]][0], int):
                    result_input[j][result_diag[subject_batch[j]][0]] = 1
                else:
                    for n in range(len(result_diag[subject_batch[j]][0])):
                        result_input[j][result_diag[subject_batch[j]][0][n]]=1
        real_lenth_encounter_list.append(real_lenth_encounter)
        diag_input_list.append(diag_input)
        med_input_list.append(med_input)
        pro_input_list.append(pro_input)
        age_input_list.append(age_input)
        gender_input_list.append(gender_input)
        is_edreg_input_list.append(is_edreg_input)
        real_lenth_med_list.append(real_lenth_med)
        real_lenth_pro_list.append(real_lenth_pro)
        result_input_list.append(result_input)

    hps = HParams(
        mode=mode,
        task=task,
        batch_size=batch_size,
        num_diag=num_diag,
        embedding_size=embedding_size,
        num_encounter=num_encounter,
        diag_step=diag_step,
        med_step=med_step,
        pro_step=pro_step,
        learning_rate=0.0001,
    )
    input_list=[real_lenth_encounter_list,diag_input_list,med_input_list,real_lenth_med_list,pro_input_list,real_lenth_pro_list,age_input_list,gender_input_list,is_edreg_input_list,result_input_list]
    return hps,embedding_list,input_list

if __name__ == '__main__':
    embedding_path=dataset_path+'{}/input/node_embedding_160'.format(dataset_name)
    dict_list_path=dataset_path+'{}/input/dict_list.pkl'.format(dataset_name)
    input_path=dataset_path+'{}/input/train.pkl'.format(dataset_name)
    # batch_size=10
    hps,embedding_list,_=get_input_batch('train',embedding_path,dict_list_path,input_path)
    # print(hps.num_encounter)
    # print(hps.diag_step)
    # print(hps.med_step)
    # print(hps.pro_step)
