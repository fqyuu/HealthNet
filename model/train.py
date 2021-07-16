import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model.DepHIM import CausalModel
# from model.CausalModel_NC import CausalModel
# from model.CausalModel_NT import CausalModel
import numpy as np
from data_process.batch_reader import get_input_batch
from sklearn.metrics import roc_auc_score
from data_process.util import dataset_name,embedding_size,task,dataset_path

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.flags.DEFINE_string("mode", "train", "set train for training and validation set, set test for testiing set")
FLAGS = tf.compat.v1.flags.FLAGS

embedding_path = dataset_path+'{}/input/node_embedding_'.format(dataset_name)+str(embedding_size)
dict_list_path = dataset_path+'{}/input/dict_list.pkl'.format(dataset_name)
input_path_train = dataset_path+'{}/input/train.pkl'.format(dataset_name)
input_path_valid = dataset_path+'{}/input/valid.pkl'.format(dataset_name)
input_path_test = dataset_path+'{}/input/test.pkl'.format(dataset_name)

save_model_dir='../../save/Causalmodel/{}/'.format(dataset_name)+task+'/'+str(embedding_size)
# save_model_dir=save_model_dir+'/NT/'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

def _train_b():
    start_global_step = 1
    hps,embedding_list,input_list=get_input_batch('train',embedding_path,dict_list_path,input_path_train)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            model = CausalModel(hps,embedding_list)
            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),max_to_keep=0)
            if os.path.exists(os.path.join(save_model_dir, "checkpoint")):
                print("continue to train")
                ckpt = tf.train.get_checkpoint_state(save_model_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                start_global_step = int(ckpt.model_checkpoint_path.split("-")[-1]) + 1

            print("start to train")
            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            for step in range(start_global_step,15):
                model.assign_global_step(sess, step)
                count=0
                loss_all=0
                prediction_list=[]
                for i in range(len(real_lenth_encounter_list)):
                    loss,prediction, _ = model.run_train_step(sess=sess,
                                                         real_lenth_encounter=real_lenth_encounter_list[i],
                                                         diagnoses=diag_input_list[i],
                                                         medication=med_input_list[i],
                                                         real_lenth_med=real_lenth_med_list[i],
                                                         procedure=pro_input_list[i],
                                                         real_lenth_pro=real_lenth_pro_list[i],
                                                         age=age_input_list[i],
                                                         gender=gender_input_list[i],
                                                         is_edreg=is_edreg_input_list[i],
                                                         lab_result=result_input_list[i])

                    count+=1
                    loss_all += loss
                    prediction_list.append(prediction)
                pre,rec,f1,acc,auc=pre_rec_f1(result_input_list,prediction_list)
                # print("train auc: {},acc: {},pre: {},rec: {},f1: {}".format(auc,acc,pre,rec,f1))

                checkpoint_path = os.path.join(save_model_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                # _test_b()
                _valid_b()

def _valid_b():
    hps, embedding_list, input_list = get_input_batch('train', embedding_path,
                                                      dict_list_path, input_path_valid)
    ckpt = tf.train.get_checkpoint_state(save_model_dir)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            model = CausalModel(hps,embedding_list)
            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            saver.restore(sess, ckpt.model_checkpoint_path) #the newest
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-{}".format(1)))

            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            count = 0
            loss_all = 0
            prediction_list=[]
            for i in range(len(real_lenth_encounter_list)):
                loss, prediction,_ = model.run_train_step(sess=sess,
                                                            real_lenth_encounter=real_lenth_encounter_list[i],
                                                            diagnoses=diag_input_list[i],
                                                            medication=med_input_list[i],
                                                            real_lenth_med=real_lenth_med_list[i],
                                                            procedure=pro_input_list[i],
                                                            real_lenth_pro=real_lenth_pro_list[i],
                                                            age=age_input_list[i],
                                                            gender=gender_input_list[i],
                                                            is_edreg=is_edreg_input_list[i],
                                                            lab_result=result_input_list[i])
                count += 1
                loss_all += loss
                prediction_list.append(prediction)
            pre, rec, f1, acc, auc = pre_rec_f1(result_input_list, prediction_list)
            print("valid auc: {},acc: {},pre: {},rec: {},f1: {}".format(auc, acc, pre, rec, f1))
            _test_b()

def _test_b():
    hps, embedding_list, input_list = get_input_batch('test', embedding_path,
                                                      dict_list_path, input_path_test)
    ckpt = tf.train.get_checkpoint_state(save_model_dir)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            model = CausalModel(hps,embedding_list)
            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            for j in range(1):
                count = 0
                loss_all = 0
                prediction_list = []
                if FLAGS.mode == "train":
                    saver.restore(sess, ckpt.model_checkpoint_path)  # the newest
                if FLAGS.mode == "test":
                    saver.restore(sess, os.path.join(save_model_dir, "model.ckpt-{}".format(j + 1)))
                for i in range(len(real_lenth_encounter_list)):
                    loss, prediction = model.run_test_step(sess=sess,
                                                           real_lenth_encounter=real_lenth_encounter_list[i],
                                                           diagnoses=diag_input_list[i],
                                                           medication=med_input_list[i],
                                                           real_lenth_med=real_lenth_med_list[i],
                                                           procedure=pro_input_list[i],
                                                           real_lenth_pro=real_lenth_pro_list[i],
                                                           age=age_input_list[i],
                                                           gender=gender_input_list[i],
                                                           is_edreg=is_edreg_input_list[i],
                                                           lab_result=result_input_list[i])
                    count += 1
                    loss_all += loss
                    prediction_list.append(prediction)
                pre, rec, f1, acc ,auc= pre_rec_f1(result_input_list, prediction_list)
                print("test auc: {},acc: {},pre: {},rec: {},f1: {}".format(auc, acc, pre,rec, f1))
                f = dataset_name+"_end"+str(embedding_size)+".txt"
                with open(f, "a") as file:
                    file.write("auc: {}, acc: {},pre: {},rec: {},f1: {}".format(auc,acc, pre, rec,f1) + '\n')

def _train_m():
    start_global_step = 1
    hps,embedding_list,input_list=get_input_batch(FLAGS.mode,embedding_path,dict_list_path,input_path_train)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            train_model = CausalModel(hps,embedding_list)
            train_model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),max_to_keep=0)
            if os.path.exists(os.path.join(save_model_dir, "checkpoint")):
                print("continue to train")
                ckpt = tf.train.get_checkpoint_state(save_model_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                start_global_step = int(ckpt.model_checkpoint_path.split("-")[-1]) + 1

            print("start to train")
            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            for step in range(start_global_step,21):
                count=0
                loss_all=0
                pre5_all=0
                rec5_all=0
                pre10_all=0
                rec10_all=0
                pre15_all=0
                rec15_all=0
                train_model.assign_global_step(sess, step)
                for i in range(len(real_lenth_encounter_list)):
                    loss,prediction,_ = train_model.run_train_step(sess=sess,
                                                         real_lenth_encounter=real_lenth_encounter_list[i],
                                                         diagnoses=diag_input_list[i],
                                                         medication=med_input_list[i],
                                                         real_lenth_med=real_lenth_med_list[i],
                                                         procedure=pro_input_list[i],
                                                         real_lenth_pro=real_lenth_pro_list[i],
                                                         age=age_input_list[i],
                                                         gender=gender_input_list[i],
                                                         is_edreg=is_edreg_input_list[i],
                                                         lab_result=result_input_list[i])
                    count+=1
                    loss_all += loss
                    pre_5,rec_5=pre_rec_m(result_input_list[i],prediction[:,0:5])
                    pre_10, rec_10 = pre_rec_m(result_input_list[i], prediction[:, 0:10])
                    pre_15, rec_15 = pre_rec_m(result_input_list[i], prediction[:, 0:15])
                    pre5_all+=pre_5
                    rec5_all+=rec_5
                    pre10_all+=pre_10
                    rec10_all+=rec_10
                    pre15_all+=pre_15
                    rec15_all+=rec_15
                # print("train loss: {},pre_5: {},rec_5: {},pre_10: {},rec_10: {},pre_15: {},rec_15: {}"
                #       .format(loss_all / count, pre5_all / count, rec5_all / count, pre10_all / count, rec10_all / count, pre15_all / count, rec15_all / count))

                checkpoint_path = os.path.join(save_model_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                # _valid_m(step)
                _test_m()

def _valid_m():
    hps, embedding_list, input_list = get_input_batch('train', embedding_path,dict_list_path, input_path_valid)
    ckpt = tf.train.get_checkpoint_state(save_model_dir)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            model = CausalModel(hps,embedding_list)
            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            saver.restore(sess, ckpt.model_checkpoint_path) #the newest
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-{}".format(1)))

            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            count = 0
            loss_all = 0
            pre5_all = 0
            rec5_all = 0
            pre10_all = 0
            rec10_all = 0
            pre15_all = 0
            rec15_all = 0
            for i in range(len(real_lenth_encounter_list)):
                loss, prediction, _ = model.run_train_step(sess=sess,
                                                                 real_lenth_encounter=real_lenth_encounter_list[i],
                                                                 diagnoses=diag_input_list[i],
                                                                 medication=med_input_list[i],
                                                                 real_lenth_med=real_lenth_med_list[i],
                                                                 procedure=pro_input_list[i],
                                                                 real_lenth_pro=real_lenth_pro_list[i],
                                                                 age=age_input_list[i],
                                                                 gender=gender_input_list[i],
                                                                 is_edreg=is_edreg_input_list[i],
                                                                 lab_result=result_input_list[i])
                count += 1
                loss_all += loss
                pre_5, rec_5 = pre_rec_m(result_input_list[i], prediction[:, 0:5])
                pre_10, rec_10 = pre_rec_m(result_input_list[i], prediction[:, 0:10])
                pre_15, rec_15 = pre_rec_m(result_input_list[i], prediction[:, 0:15])
                pre5_all += pre_5
                rec5_all += rec_5
                pre10_all += pre_10
                rec10_all += rec_10
                pre15_all += pre_15
                rec15_all += rec_15
            print("valid loss: {},pre_5: {},rec_5: {},pre_10: {},rec_10: {},pre_15: {},rec_15: {}"
                  .format(loss_all / count, pre5_all / count, rec5_all / count, pre10_all / count, rec10_all / count,
                          pre15_all / count, rec15_all / count))
            _test_m()

def _test_m():
    hps, embedding_list, input_list = get_input_batch('test', embedding_path,dict_list_path, input_path_test)
    ckpt = tf.train.get_checkpoint_state(save_model_dir)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope("Model"):
            model = CausalModel(hps,embedding_list)
            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            saver.restore(sess, ckpt.model_checkpoint_path) #the newest
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-{}".format(1)))

            real_lenth_encounter_list=input_list[0]
            diag_input_list=input_list[1]
            med_input_list = input_list[2]
            real_lenth_med_list = input_list[3]
            pro_input_list= input_list[4]
            real_lenth_pro_list = input_list[5]
            age_input_list = input_list[6]
            gender_input_list = input_list[7]
            is_edreg_input_list = input_list[8]
            result_input_list=input_list[9]
            count = 0
            loss_all = 0
            pre5_all = 0
            rec5_all = 0
            pre10_all = 0
            rec10_all = 0
            pre15_all = 0
            rec15_all = 0
            for i in range(len(real_lenth_encounter_list)):
                loss, prediction = model.run_test_step(sess=sess,
                                                                 real_lenth_encounter=real_lenth_encounter_list[i],
                                                                 diagnoses=diag_input_list[i],
                                                                 medication=med_input_list[i],
                                                                 real_lenth_med=real_lenth_med_list[i],
                                                                 procedure=pro_input_list[i],
                                                                 real_lenth_pro=real_lenth_pro_list[i],
                                                                 age=age_input_list[i],
                                                                 gender=gender_input_list[i],
                                                                 is_edreg=is_edreg_input_list[i],
                                                                 lab_result=result_input_list[i])
                count += 1
                loss_all += loss
                pre_5, rec_5 = pre_rec_m(result_input_list[i], prediction[:, 0:5])
                pre_10, rec_10 = pre_rec_m(result_input_list[i], prediction[:, 0:10])
                pre_15, rec_15 = pre_rec_m(result_input_list[i], prediction[:, 0:15])
                pre5_all += pre_5
                rec5_all += rec_5
                pre10_all += pre_10
                rec10_all += rec_10
                pre15_all += pre_15
                rec15_all += rec_15
            print("test,pre_5: {},rec_5: {},pre_10: {},rec_10: {},pre_15: {},rec_15: {}"
                  .format(pre5_all / count, rec5_all / count, pre10_all / count, rec10_all / count,
                          pre15_all / count, rec15_all / count))
            f = dataset_name+"_diag" + str(embedding_size) + ".txt"
            with open(f, "a") as file:
                file.write("test,pre_5: {},rec_5: {},pre_10: {},rec_10: {},pre_15: {},rec_15: {}"
                  .format(pre5_all / count, rec5_all / count, pre10_all / count, rec10_all / count,
                          pre15_all / count, rec15_all / count) + '\n')

def pre_rec_f1(result_label,prediction):
    temp_label=np.hstack(result_label)
    temp_prediction=np.hstack(prediction)
    auc=roc_auc_score(temp_label, temp_prediction)
    temp_prediction = np.int64(temp_prediction > 0.55)
    TP=np.sum(temp_label*temp_prediction)
    FN=np.sum(temp_label*np.array([1-x for x in temp_prediction]))
    FP=np.sum(np.array([1-x for x in temp_label])*temp_prediction)
    TN=np.sum(np.array([1-x for x in temp_label])*np.array([1-x for x in temp_prediction]))
    acc=(TP+TN)/(TP+FN+FP+TN)
    rec = TP / (TP + FN)
    if TP+FP==0:
        pre=0
        f1=0
    else:
        pre = TP / (TP + FP)
        f1 = (2 * pre * rec) / (pre + rec)
    return pre,rec,f1,acc,auc

def pre_rec_m(result_label,prediction):
    bs=result_label.shape[0]
    pre=0
    rec=0
    for i in range(bs):
        label=np.where(result_label[i]==1)[0]
        predict=prediction[i]
        pre+=len(set(label).intersection(set(predict)))/prediction.shape[1]
        rec+=len(set(label).intersection(set(predict)))/len(label)

    return pre/bs,rec/bs

def main(_):
    if task == 'readm' or task == 'end':
        if FLAGS.mode == "train":
            _train_b()
        if FLAGS.mode == "test":
            _test_b()
    if task == 'diag':
        if FLAGS.mode == "train":
            _train_m()
        if FLAGS.mode == "test":
            _test_m()
if __name__ == '__main__':
    tf.compat.v1.app.run()