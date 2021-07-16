import tensorflow as tf
from data_process.batch_reader import HParams
from tensorflow.python.ops.rnn import dynamic_rnn

class CausalModel(object):
    def __init__(self,hps:HParams,embedding_list):
        self._hps=hps
        self._embedidng_list=embedding_list

    def run_train_step(self,sess:tf.compat.v1.Session(),real_lenth_encounter,diagnoses,medication,real_lenth_med,procedure,real_lenth_pro,age,gender,is_edreg,lab_result):
        to_return = [self._loss,self.prediction,self.train_op]
        return sess.run(to_return,
                        feed_dict={
                            self._real_lenth_encounter: real_lenth_encounter,
                            self._diagnoses:diagnoses,
                            self._medication: medication,
                            self._real_lenth_med:real_lenth_med,
                            self._procedure: procedure,
                            self._real_lenth_pro:real_lenth_pro,
                            self._age: age,
                            self._gender: gender,
                            self._is_edreg: is_edreg,
                            self._lab_result: lab_result,
                        })
    def run_test_step(self,sess:tf.compat.v1.Session(),real_lenth_encounter,diagnoses,medication,real_lenth_med,procedure,real_lenth_pro,age,gender,is_edreg,lab_result):
        to_return = [self._loss,self.prediction]
        return sess.run(to_return,
                        feed_dict={
                            self._real_lenth_encounter: real_lenth_encounter,
                            self._diagnoses:diagnoses,
                            self._medication: medication,
                            self._real_lenth_med:real_lenth_med,
                            self._procedure: procedure,
                            self._real_lenth_pro:real_lenth_pro,
                            self._age: age,
                            self._gender: gender,
                            self._is_edreg: is_edreg,
                            self._lab_result: lab_result,
                        })

    def add_placeholders(self):
        hps=self._hps
        self._real_lenth_encounter = tf.compat.v1.placeholder(tf.int32, name="_real_lenth_encounter",shape=[hps.batch_size])

        self._diagnoses=tf.compat.v1.placeholder(tf.float32,name="_diagnoses",shape=[hps.batch_size,hps.num_encounter,hps.num_diag])

        self._medication=tf.compat.v1.placeholder(tf.int32,name="_medication",shape=[hps.batch_size,hps.num_encounter,hps.med_step])
        self._real_lenth_med=tf.compat.v1.placeholder(tf.int32,name="_real_lenth_med",shape=[hps.batch_size,hps.num_encounter])

        self._procedure=tf.compat.v1.placeholder(tf.int32,name="_procedure",shape=[hps.batch_size,hps.num_encounter,hps.pro_step])
        self._real_lenth_pro = tf.compat.v1.placeholder(tf.int32, name="_real_lenth_pro",shape=[hps.batch_size, hps.num_encounter])

        self._age=tf.compat.v1.placeholder(tf.int32,name="_age",shape=[hps.batch_size,hps.num_encounter])
        self._gender=tf.compat.v1.placeholder(tf.int32,name="_gender",shape=[hps.batch_size])
        self._is_edreg = tf.compat.v1.placeholder(tf.int32, name="_is_edreg", shape=[hps.batch_size,hps.num_encounter])

        if hps.task=='diag':
            self._lab_result = tf.compat.v1.placeholder(tf.int32, name="_lab_result", shape=[hps.batch_size,hps.num_diag])
        else:#readm or end
            self._lab_result = tf.compat.v1.placeholder(tf.int32, name="_lab_result", shape=[hps.batch_size])

    def create_model(self):
        hps=self._hps
        with tf.compat.v1.variable_scope("embedding"):
            embedding_diag_c = tf.compat.v1.get_variable("embedding_diag_c",dtype=tf.float32,initializer=self._embedidng_list[0],trainable=False)
            embedding_med_c = tf.compat.v1.get_variable("embedding_med_c",dtype=tf.float32,initializer=self._embedidng_list[1],trainable=False)
            embedding_pro_c = tf.compat.v1.get_variable("embedding_pro_c",dtype=tf.float32,initializer=self._embedidng_list[2],trainable=False)
            embedding_age_c = tf.compat.v1.get_variable("embedding_age_c",dtype=tf.float32,initializer=self._embedidng_list[3],trainable=False)
            embedding_gender_c = tf.compat.v1.get_variable("embedding_gender_c",dtype=tf.float32,initializer=self._embedidng_list[4],trainable=False)
            embedding_edreg_c = tf.compat.v1.get_variable("embedding_edreg_c",dtype=tf.float32,initializer=self._embedidng_list[5],trainable=False)
            # embedding_result is embedding_readm or embedding_end
            if hps.task=='diag':
                embedding_result=[]
            else:
                embedding_result = tf.compat.v1.get_variable("embedding_result",dtype=tf.float32,initializer=self._embedidng_list[6],trainable=False)

            embedding_diag_t = tf.compat.v1.get_variable("embedding_diag_t",self._embedidng_list[0].shape,dtype=tf.float32)
            embedding_med_t = tf.compat.v1.get_variable("embedding_med_t",self._embedidng_list[1].shape,dtype=tf.float32)
            embedding_pro_t = tf.compat.v1.get_variable("embedding_pro_t",self._embedidng_list[2].shape,dtype=tf.float32)

        with tf.compat.v1.variable_scope("within_encounter"):
            w_d=tf.compat.v1.get_variable("w_d",[2*hps.embedding_size,hps.embedding_size],dtype=tf.float32)
            b_d = tf.compat.v1.get_variable("b_d", [hps.embedding_size], dtype=tf.float32)
            w_m=tf.compat.v1.get_variable("w_m",[2*hps.embedding_size,hps.embedding_size],dtype=tf.float32)
            b_m = tf.compat.v1.get_variable("b_m", [hps.embedding_size], dtype=tf.float32)
            w_p=tf.compat.v1.get_variable("w_p",[2*hps.embedding_size,hps.embedding_size],dtype=tf.float32)
            b_p = tf.compat.v1.get_variable("b_p", [hps.embedding_size], dtype=tf.float32)

            cell_med=tf.compat.v1.nn.rnn_cell.LSTMCell(hps.embedding_size)
            inistate_med = cell_med.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
            cell_pro = tf.compat.v1.nn.rnn_cell.LSTMCell(hps.embedding_size)
            inistate_pro = cell_pro.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
            matrix_fusion=tf.compat.v1.get_variable("matrix_fusion",[hps.embedding_size*5,hps.embedding_size])
            encounter_input=[]
            for i in range(hps.num_encounter):
                medication_c=tf.nn.embedding_lookup(embedding_med_c,self._medication[:,i,:])
                medication_t=tf.nn.embedding_lookup(embedding_med_t,self._medication[:,i,:])
                medication = tf.nn.relu(tf.matmul(tf.concat([medication_c, medication_t], axis=2), w_m) + b_m)
                _,outputs_med = dynamic_rnn(
                    cell=cell_med,
                    inputs=medication,
                    sequence_length=self._real_lenth_med[:,i], initial_state=inistate_med,scope="rnn_med")

                procedure_c=tf.nn.embedding_lookup(embedding_pro_c,self._procedure[:,i,:])
                procedure_t=tf.nn.embedding_lookup(embedding_pro_t,self._procedure[:,i,:])
                procedure = tf.nn.relu(tf.matmul(tf.concat([procedure_c, procedure_t], axis=2), w_p) + b_p)
                _,outputs_pro = dynamic_rnn(
                    cell=cell_pro,
                    inputs=procedure,
                    sequence_length=self._real_lenth_pro[:,i], initial_state=inistate_pro,scope="rnn_pro")

                age=tf.nn.embedding_lookup(embedding_age_c,self._age[:,i])
                gender=tf.nn.embedding_lookup(embedding_gender_c,self._gender)
                is_edreg=tf.nn.embedding_lookup(embedding_edreg_c,self._is_edreg[:,i])
                output_concat=tf.concat([outputs_med[1],outputs_pro[1],age,gender,is_edreg],axis=1)
                encounter_input.append(tf.matmul(output_concat,matrix_fusion))
            encounter_input=tf.stack(encounter_input,axis=1)

        with tf.compat.v1.variable_scope("global"):
            w_g=tf.compat.v1.get_variable("w_g",[2*hps.embedding_size,hps.embedding_size],dtype=tf.float32)
            b_g = tf.compat.v1.get_variable("b_g", [hps.embedding_size], dtype=tf.float32)
            diagnosis_c = tf.matmul(self._diagnoses,embedding_diag_c)
            diagnosis_t = tf.matmul(self._diagnoses,embedding_diag_t)
            diagnosis = tf.nn.relu(tf.matmul(tf.concat([diagnosis_c, diagnosis_t], axis=2), w_d) + b_d)
            input_enc = tf.nn.relu(tf.matmul(tf.concat([diagnosis, encounter_input], axis=2), w_g) + b_g)

            cell_global = tf.compat.v1.nn.rnn_cell.LSTMCell(hps.embedding_size)
            inistate_global=cell_global.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
            outputs,output_final= dynamic_rnn(
                cell=cell_global,
                inputs=input_enc,
                sequence_length=self._real_lenth_encounter, initial_state=inistate_global, scope="rnn_global")

            w_hs=tf.compat.v1.get_variable("w_hs",[1,hps.embedding_size],dtype=tf.float32)
            b_hs = tf.compat.v1.get_variable("b_hs", 1, dtype=tf.float32)
            if hps.task=='diag':
                loss = 0
                w_1=tf.compat.v1.get_variable("w_1",[hps.embedding_size,hps.num_diag],dtype=tf.float32)
                b_1 = tf.compat.v1.get_variable("b_1", [hps.num_diag], dtype=tf.float32)
                for i in range(hps.num_encounter):
                    if i < hps.num_encounter - 1:
                        alpha = tf.nn.softmax(tf.reshape(tf.matmul(outputs[:, :i + 1, :], tf.transpose(w_hs)) + b_hs,
                                                         [hps.batch_size, i + 1]))
                        c_i = tf.zeros(shape=[hps.batch_size, hps.embedding_size])
                        for j in range(i + 1):
                            c_i_j = tf.transpose(tf.transpose(diagnosis_t[:, j, :]) * tf.transpose(alpha)[j])
                            c_i = c_i + c_i_j
                        y = tf.nn.softmax(tf.matmul(c_i, w_1) + b_1)
                        loss += -tf.reduce_sum(self._diagnoses[:, i + 1, :] * tf.compat.v1.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                            (1 - self._diagnoses[:, i + 1, :]) * tf.compat.v1.log(tf.clip_by_value(1 - y, 1e-10, 1.0)),axis=1)
                    else:
                        alpha = tf.nn.softmax(tf.reshape(tf.matmul(outputs[:, :i + 1, :], tf.transpose(w_hs)) + b_hs,
                                                         [hps.batch_size, i + 1]))
                        c_i = tf.zeros(shape=[hps.batch_size, hps.embedding_size])
                        for j in range(i + 1):
                            c_i_j = tf.transpose(tf.transpose(diagnosis_t[:, j, :]) * tf.transpose(alpha)[j])
                            c_i = c_i + c_i_j
                        y = tf.nn.softmax(tf.matmul(c_i, w_1) + b_1)
                        self.prediction = tf.nn.top_k(y, 15).indices
                        loss += -tf.reduce_sum(tf.cast(self._lab_result, dtype=tf.float32) * tf.compat.v1.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                               (1 - tf.cast(self._lab_result, dtype=tf.float32)) * tf.compat.v1.log(tf.clip_by_value(1 - y, 1e-10, 1.0)), axis=1)
                self._loss = tf.reduce_sum(loss * (1 / tf.cast(self._real_lenth_encounter, dtype=tf.float32))) / hps.batch_size
            else:#readm or end
                w_1 = tf.compat.v1.get_variable("w_1", [ hps.embedding_size, 1], dtype=tf.float32)
                b_1 = tf.compat.v1.get_variable("b_1", 1, dtype=tf.float32)

                # alpha = tf.nn.softmax(tf.reshape(tf.matmul(outputs, tf.transpose(w_hs)) + b_hs,
                #                                  [hps.batch_size, hps.num_encounter]))
                # c_i = tf.zeros(shape=[hps.batch_size, hps.embedding_size])
                # for j in range(hps.num_encounter):
                #     c_i_j = tf.transpose(tf.transpose(diagnosis_t[:, j, :]) * tf.transpose(alpha)[j])
                #     c_i = c_i + c_i_j
                # y = tf.reshape(tf.nn.sigmoid(tf.matmul(c_i, w_1) + b_1),[hps.batch_size])
                # norm = tf.reduce_sum(tf.norm(c_i - tf.nn.embedding_lookup(embedding_result, self._lab_result), axis=1))

                y = tf.reshape(tf.nn.sigmoid(tf.matmul(output_final[1], w_1) + b_1),[hps.batch_size])
                norm = tf.reduce_sum(tf.norm(output_final[1] - tf.nn.embedding_lookup(embedding_result, self._lab_result), axis=1))

                self.prediction = y
                self._loss = -tf.reduce_sum(tf.cast(self._lab_result, dtype=tf.float32) * tf.compat.v1.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                       (1 - tf.cast(self._lab_result, dtype=tf.float32)) * tf.compat.v1.log(tf.clip_by_value(1-y, 1e-10, 1.0)))/hps.batch_size+0.3*norm/hps.batch_size

    def _add_train_op(self):
        # tvars = tf.compat.v1.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(
        #     tf.gradients(self._loss, tvars), self.hps.max_grad_norm
        # )
        # optimizer = tf.compat.v1.train.AdamOptimizer()
        #
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars),
        #                                           global_step=self.global_step)

        self.train_op = tf.compat.v1.train.AdamOptimizer().minimize(self._loss)
        # self.train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self._hps.learning_rate).minimize(self._loss)

    def assign_global_step(self, sess: tf.compat.v1.Session(), new_value):
        sess.run(tf.compat.v1.assign(self.global_step, new_value))

    def build_graph(self):
        self.add_placeholders()
        self.create_model()
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        if self._hps.mode == "train":
            self._add_train_op()
