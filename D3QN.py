#
# 文件名: D3QN.py (已修复)
# 修复：
# 1. 在 choose_action_with_coordination 的 "利用" (Exploitation) 阶段
#    移除了手动修改 Q 值的逻辑。
# 2. 保留了 "探索" (Explore) 阶段的智能（负载感知）探索。
#
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
from Config import Config

tf.disable_v2_behavior()


class DuelingDoubleDeepQNetwork:

    def __init__(self,
                 n_actions,
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=Config.LEARNING_RATE,
                 reward_decay=Config.REWARD_DECAY,
                 replace_target_iter=Config.N_NETWORK_UPDATE,
                 memory_size=Config.MEMORY_SIZE,
                 batch_size=32,
                 dueling=True,
                 double_q=True,
                 N_L1=20,
                 N_lstm=20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.N_lstm = N_lstm
        self.n_lstm_step = 10
        self.n_lstm_state = n_lstm_features

        # --- epsilon schedule (robust to missing Config fields) ---
        # If your runtime Config.py doesn't define these fields, we fall back to safe defaults.
        self.epsilon = float(getattr(Config, "EPSILON_START", 1.0))
        self.epsilon_min = float(getattr(Config, "EPSILON_END", 0.01))
        self.epsilon_decay = float(getattr(Config, "EPSILON_DECAY", 0.9995))

        # Optional compatibility: some older code uses EPSILON_INCREMENT (increase) style
        # If user mistakenly set DECAY > 1, clamp it.
        if self.epsilon_decay <= 0:
            self.epsilon_decay = 0.9995
        if self.epsilon_decay >= 1.0:
            # keep it < 1 so epsilon actually decays
            self.epsilon_decay = 0.9995

        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))
        self.store_q_value = list()
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def _build_net(self):
        tf.reset_default_graph()

        def build_layers(s, lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer):
            with tf.variable_scope('l0'):
                lstm_dnn = tf.nn.rnn_cell.BasicLSTMCell(n_lstm)
                lstm_dnn.zero_state(self.batch_size, tf.float32)
                lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [n_lstm + self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s], 1), w1) + b1)
            with tf.variable_scope('l12'):
                w12 = tf.get_variable('w12', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b12 = tf.get_variable('b12', [1, n_l1], initializer=b_initializer, collections=c_names)
                l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)
            if self.dueling:
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l12, w2) + b2
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l12, w2) + b2
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.lstm_s = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name='lstm1_s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.lstm_s_ = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name='lstm1_s_')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_lstm, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1, self.N_lstm, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.s, self.lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer)
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.lstm_s_, c_names, n_l1, n_lstm, w_initializer, b_initializer)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            lstm_observation = np.array(self.lstm_history)
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation,
                                                     self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step,
                                                                                           self.n_lstm_state)})
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action_with_coordination(self, observation, edge_loads=None):
        """
        🔧 修复版：
        1. "利用" (Exploit) 阶段: 完全信任Q网络 (移除手动惩罚)。
        2. "探索" (Explore) 阶段: 保留智能探索 (避开高负载节点)。
        """
        observation_input = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # --- 利用 (Exploit) ---
            # ✅ 修复: Agent应该完全信任自己的Q网络
            # 移除所有手动修改Q值的代码
            # Agent会通过QoE奖励(包含负载惩罚)来 *学习* 避免高负载节点
            lstm_observation = np.array(self.lstm_history)
            q_values = self.sess.run(self.q_eval,
                                     feed_dict={self.s: observation_input,
                                                self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step,
                                                                                      self.n_lstm_state)})

            # (原有的手动惩罚逻辑已被删除)

            self.store_q_value.append({'observation': observation_input, 'q_value': q_values})
            action = np.argmax(q_values)

        else:
            # --- 探索 (Explore) ---
            # ✅ 保留: "智能探索" 是一个好策略
            # 优先探索低负载节点
            if edge_loads is not None and np.random.uniform() < 0.9:  # 90% 概率进行智能探索
                action_probs = np.ones(self.n_actions)

                # 本地处理的基础概率
                action_probs[0] = 0.15  # 降低本地处理概率，鼓励探索卸载

                # 边缘服务器的概率基于负载（负载越低，概率越高）
                for i in range(1, self.n_actions):
                    edge_idx = i - 1
                    # 负载越低概率越高
                    action_probs[i] = 1.0 / (1.0 + edge_loads[edge_idx] * 10)  # 放大差异

                # 归一化概率
                action_probs = action_probs / np.sum(action_probs)
                action = np.random.choice(self.n_actions, p=action_probs)
            else:
                # 10% 概率进行完全随机探索
                action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii, jj, :] = self.memory[sample_index[ii] + jj,
                                               self.n_features + 1 + 1 + self.n_features:]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:], self.lstm_s_: lstm_batch_memory[:, :, self.n_lstm_state:],
                self.s: batch_memory[:, -self.n_features:], self.lstm_s: lstm_batch_memory[:, :, self.n_lstm_state:],
            })
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features],
                                             self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                                                self.q_target: q_target})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        return self.cost

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]
        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    def Initialize(self, sess, iot):
        self.sess = sess
        self.load_model(iot)

    def load_model(self, iot):
        latest_ckpt = tf.train.latest_checkpoint("./TrainedModel_20UE_2EN_PerformanceMode/800/" + str(iot) + "_X_model")
        print(latest_ckpt, "_____+______________________________________________")
        if latest_ckpt is not None:
            self.saver.restore(self.sess, latest_ckpt)