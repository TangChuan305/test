 # MEC_Env_FIFO.py (QECO Baseline - 完整修复版)
# 🔧 关键特性：
# 1. 使用FIFO队列（不是优先级队列）
# 2. 不考虑安全性（允许任何动作）
# 3. 不使用负载均衡
# 4. 纯粹追求性能（延迟和能耗）

from Config import Config
import numpy as np
import random
import math


class MEC:
    def __init__(self, num_ue, num_edge, num_time, num_component, max_delay):
        self.n_ue = num_ue
        self.n_edge = num_edge
        self.n_time = num_time
        self.n_component = num_component
        self.max_delay = max_delay
        self.duration = Config.DURATION
        self.ue_p_comp = Config.UE_COMP_ENERGY
        self.ue_p_tran = Config.UE_TRAN_ENERGY
        self.ue_p_idle = Config.UE_IDLE_ENERGY
        self.edge_p_comp = Config.EDGE_COMP_ENERGY

        self.en_trust_levels = Config.EN_TRUST_LEVELS
        self.task_sensitivity_levels = Config.TASK_SENSITIVITY_LEVELS

        self.time_count = 0
        self.task_count_ue = 0
        self.task_count_edge = 0
        self.n_actions = 1 + self.n_edge

        self.previous_actions = np.zeros(self.n_ue)
        self.n_features = 5 + self.n_edge + 2 + self.n_edge

        self.n_lstm_state = self.n_edge

        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0
        self.security_violation_count = 0
        self.security_blocked_count = 0


        self.comp_cap_ue = Config.UE_COMP_CAP * np.ones(self.n_ue) * self.duration
        self.comp_cap_edge = Config.EDGE_COMP_CAP * np.ones([self.n_edge]) * self.duration
        self.tran_cap_ue = Config.UE_TRAN_CAP * np.ones([self.n_ue, self.n_edge]) * self.duration
        self.n_cycle = 1
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB
        self.max_arrive_size = Config.TASK_MAX_SIZE
        self.min_arrive_size = Config.TASK_MIN_SIZE
        self.arrive_task_size_set = np.arange(self.min_arrive_size, self.max_arrive_size, 0.1)
        self.ue_energy_state = [Config.UE_ENERGY_STATE[np.random.randint(0, len(Config.UE_ENERGY_STATE))] for ue in
                                range(self.n_ue)]

        self.arrive_task_size = np.zeros([self.n_time, self.n_ue])
        self.arrive_task_dens = np.zeros([self.n_time, self.n_ue])
        self.arrive_task_sens = np.zeros([self.n_time, self.n_ue])

        self.n_task = int(self.n_time * self.task_arrive_prob)

        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        # 🔧 FIFO队列 - 使用简单的列表
        self.ue_computation_queue = [[] for _ in range(self.n_ue)]
        self.ue_transmission_queue = [[] for _ in range(self.n_ue)]
        self.edge_computation_queue = [[[] for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        self.edge_ue_m = np.zeros(self.n_edge)
        self.edge_ue_m_observe = np.zeros(self.n_edge)

        self.edge_load_factors = np.zeros(self.n_edge)

        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'SENS': np.nan, 'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in
                                   range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'SENS': np.nan, 'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in
                                    range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'SENS': np.nan, 'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _
                                  in range(self.n_ue)]

        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue


    def check_security_constraint(self, edge_idx, task_sensitivity):
        """检查安全约束。True=安全，False=违规。"""
        if edge_idx < 0 or edge_idx >= len(self.en_trust_levels):
            return True
        edge_trust = self.en_trust_levels[edge_idx]
        if task_sensitivity >= 2 and edge_trust < 0.9:
            return False
        elif task_sensitivity >= 1 and edge_trust < 0.6:
            return False
        return True

    def enforce_security_constraint(self, action, task_sensitivity):
        """若违反安全约束则强制改为本地执行。"""
        if action == 0:
            return action, False
        edge_idx = int(action - 1)
        is_safe = self.check_security_constraint(edge_idx, task_sensitivity)
        if not is_safe:
            self.security_blocked_count += 1
            return 0, True
        return action, False

    def get_edge_load_factor(self):
        """简单的负载因子计算（用于观测，但不用于决策）"""
        load_factors = np.zeros(self.n_edge)
        for edge_idx in range(self.n_edge):
            total_queue_length = 0
            for ue in range(self.n_ue):
                total_queue_length += len(self.edge_computation_queue[ue][edge_idx])
            load_factors[edge_idx] = total_queue_length / max(self.n_ue, 1)
        return load_factors

    def reset(self, arrive_task_size, arrive_task_dens, arrive_task_sens):
        print("RESET FROM:", __file__)
        print("RESET security_blocked_count -> 0")
        self.security_blocked_count = 0
        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0
        self.security_violation_count = 0

        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue
        self.previous_actions = np.zeros(self.n_ue)

        self.arrive_task_size = arrive_task_size
        self.arrive_task_dens = arrive_task_dens
        self.arrive_task_sens = arrive_task_sens

        self.time_count = 0

        self.ue_computation_queue = [[] for _ in range(self.n_ue)]
        self.ue_transmission_queue = [[] for _ in range(self.n_ue)]
        self.edge_computation_queue = [[[] for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        self.edge_load_factors = np.zeros(self.n_edge)

        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'SENS': np.nan, 'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in
                                   range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'SENS': np.nan, 'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in
                                    range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'SENS': np.nan, 'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _
                                  in range(self.n_ue)]

        UEs_OBS = np.zeros([self.n_ue, self.n_features])
        for ue_index in range(self.n_ue):
            if self.arrive_task_size[self.time_count, ue_index] != 0:
                scalar_part = np.array([
                    self.arrive_task_size[self.time_count, ue_index],
                    self.t_ue_comp[ue_index],
                    self.t_ue_tran[ue_index],
                    self.ue_energy_state[ue_index],
                    self.arrive_task_sens[self.time_count, ue_index]
                ])
                array_part = np.squeeze(self.b_edge_comp[ue_index, :])
                local_processing_ratio = np.sum(self.previous_actions == 0) / self.n_ue
                total_offload_count = np.sum(self.previous_actions > 0)
                coordination_features = np.array([local_processing_ratio, total_offload_count])
                edge_loads = self.get_edge_load_factor()
                UEs_OBS[ue_index, :] = np.concatenate(
                    (scalar_part[:3], array_part, scalar_part[3:], coordination_features, edge_loads))

        UEs_lstm_state = np.zeros([self.n_ue, self.n_lstm_state])
        return UEs_OBS, UEs_lstm_state

    def step(self, action):



        action = np.squeeze(action)

        ue_action_local = np.zeros(self.n_ue)
        ue_action_offload = np.zeros(self.n_ue)
        ue_arrive_task_size = np.zeros(self.n_ue)
        ue_arrive_task_dens = np.zeros(self.n_ue)
        ue_arrive_task_sens = np.zeros(self.n_ue)

        for ue_index in range(self.n_ue):
            if self.time_count < len(self.arrive_task_size):
                ue_arrive_task_size[ue_index] = self.arrive_task_size[self.time_count, ue_index]
                ue_arrive_task_dens[ue_index] = self.arrive_task_dens[self.time_count, ue_index]
                ue_arrive_task_sens[ue_index] = self.arrive_task_sens[self.time_count, ue_index]

            ue_action_index = int(action[ue_index])
            task_sens = ue_arrive_task_sens[ue_index]
            ue_action_index, _ = self.enforce_security_constraint(ue_action_index, task_sens)
            if ue_action_index == 0:
                ue_action_local[ue_index] = 1
                ue_action_offload[ue_index] = -1
            else:
                ue_action_offload[ue_index] = ue_action_index - 1

        # COMPUTE TASK (UE本地计算) - FIFO顺序
        for ue_index in range(self.n_ue):
            ue_comp_cap = self.comp_cap_ue[ue_index]

            # 🔧 FIFO：直接append到队列末尾
            if ue_action_local[ue_index] == 1 and ue_arrive_task_size[ue_index] > 0:
                tmp_dict = {
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'DIV': ue_action_local[ue_index],
                    'SIZE': ue_arrive_task_size[ue_index],
                    'DENS': ue_arrive_task_dens[ue_index],
                    'SENS': ue_arrive_task_sens[ue_index],
                    'TIME': self.time_count,
                    'EDGE': ue_action_offload[ue_index],
                }
                self.ue_computation_queue[ue_index].append(tmp_dict)

            # 🔧 FIFO：从队列头部取任务（先进先出）
            if math.isnan(self.local_process_task[ue_index]['REMAIN']) and len(self.ue_computation_queue[ue_index]) > 0:
                while len(self.ue_computation_queue[ue_index]) > 0:
                    get_task = self.ue_computation_queue[ue_index].pop(0)  # FIFO: pop(0)
                    if get_task['SIZE'] != 0:
                        if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                            self.local_process_task[ue_index] = get_task.copy()
                            self.local_process_task[ue_index]['REMAIN'] = get_task['SIZE'] * get_task['DENS']
                            break
                        else:
                            self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                            self.unfinish_task[get_task['TIME'], ue_index] = 1

            if self.local_process_task[ue_index]['REMAIN'] > 0:
                processed = min(self.local_process_task[ue_index]['REMAIN'], ue_comp_cap)
                self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += processed * self.ue_p_comp
                self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += processed
                self.local_process_task[ue_index]['REMAIN'] -= ue_comp_cap

                if self.local_process_task[ue_index]['REMAIN'] <= 0:
                    self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = \
                        self.time_count - self.local_process_task[ue_index]['TIME'] + 1
                    self.local_process_task[ue_index]['REMAIN'] = np.nan
                    self.task_count_ue += 1
                elif self.time_count - self.local_process_task[ue_index]['TIME'] + 1 == self.max_delay:
                    self.local_process_task[ue_index]['REMAIN'] = np.nan
                    self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.max_delay
                    self.unfinish_task[self.local_process_task[ue_index]['TIME'], ue_index] = 1
                    self.drop_ue_count += 1

        # EDGE COMPUTE - FIFO顺序
        for edge_index in range(self.n_edge):
            edge_comp_cap = self.comp_cap_edge[edge_index]
            for ue_index in range(self.n_ue):
                # 🔧 FIFO：从队列头部取任务
                if math.isnan(self.edge_process_task[ue_index][edge_index]['REMAIN']) and \
                        len(self.edge_computation_queue[ue_index][edge_index]) > 0:
                    while len(self.edge_computation_queue[ue_index][edge_index]) > 0:
                        get_task = self.edge_computation_queue[ue_index][edge_index].pop(0)  # FIFO: pop(0)
                        if get_task['SIZE'] != 0:
                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.edge_process_task[ue_index][edge_index] = get_task.copy()
                                self.edge_process_task[ue_index][edge_index]['REMAIN'] = get_task['SIZE'] * get_task[
                                    'DENS']
                                break
                            else:
                                self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], ue_index] = 1

                if self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0:
                    processed = min(self.edge_process_task[ue_index][edge_index]['REMAIN'], edge_comp_cap)
                    self.edge_comp_energy[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += \
                        processed * self.edge_p_comp
                    self.edge_bit_processed[
                        self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += processed
                    self.edge_process_task[ue_index][edge_index]['REMAIN'] -= edge_comp_cap

                    if self.edge_process_task[ue_index][edge_index]['REMAIN'] <= 0:
                        self.process_delay[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = \
                            self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1
                        self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                    elif self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1 == self.max_delay:
                        self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                        self.process_delay[
                            self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = 1
                        self.drop_edge_count += 1

        # TRANSMISSION - FIFO顺序
        for ue_index in range(self.n_ue):
            ue_tran_cap = self.tran_cap_ue[ue_index, :]

            tmp_dict = {
                'UE_ID': ue_index,
                'TASK_ID': self.UE_TASK[ue_index],
                'DIV': ue_action_local[ue_index],
                'SIZE': ue_arrive_task_size[ue_index],
                'DENS': ue_arrive_task_dens[ue_index],
                'SENS': ue_arrive_task_sens[ue_index],
                'TIME': self.time_count,
                'EDGE': ue_action_offload[ue_index],
            }

            # 🔧 FIFO：直接append
            if ue_action_local[ue_index] == 0 and ue_arrive_task_size[ue_index] > 0:
                self.ue_transmission_queue[ue_index].append(tmp_dict)

            # 🔧 FIFO：pop(0)
            if math.isnan(self.local_transmit_task[ue_index]['REMAIN']) and len(
                    self.ue_transmission_queue[ue_index]) > 0:
                while len(self.ue_transmission_queue[ue_index]) > 0:
                    get_task = self.ue_transmission_queue[ue_index].pop(0)  # FIFO: pop(0)
                    if get_task['SIZE'] != 0:
                        if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                            self.local_transmit_task[ue_index] = get_task.copy()
                            self.local_transmit_task[ue_index]['REMAIN'] = get_task['SIZE']
                            break
                        else:
                            self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                            self.unfinish_task[get_task['TIME'], ue_index] = 1

            if self.local_transmit_task[ue_index]['REMAIN'] > 0:
                chosen_edge_tran_cap = ue_tran_cap[int(self.local_transmit_task[ue_index]['EDGE'])]
                transmitted = min(self.local_transmit_task[ue_index]['REMAIN'], chosen_edge_tran_cap)

                self.ue_tran_energy[
                    self.local_transmit_task[ue_index]['TIME'], ue_index] += transmitted * self.ue_p_tran
                self.ue_bit_transmitted[self.local_transmit_task[ue_index]['TIME'], ue_index] += transmitted
                self.local_transmit_task[ue_index]['REMAIN'] -= chosen_edge_tran_cap

                if self.local_transmit_task[ue_index]['REMAIN'] <= 0:
                    tmp_dict = {
                        'UE_ID': self.local_transmit_task[ue_index]['UE_ID'],
                        'TASK_ID': self.local_transmit_task[ue_index]['TASK_ID'],
                        'SIZE': self.local_transmit_task[ue_index]['SIZE'],
                        'DENS': self.local_transmit_task[ue_index]['DENS'],
                        'SENS': self.local_transmit_task[ue_index]['SENS'],
                        'TIME': self.local_transmit_task[ue_index]['TIME'],
                        'EDGE': self.local_transmit_task[ue_index]['EDGE'],
                        'DIV': self.local_transmit_task[ue_index]['DIV']
                    }

                    edge_idx = int(self.local_transmit_task[ue_index]['EDGE'])
                    # 🔧 FIFO：直接append
                    self.edge_computation_queue[ue_index][edge_idx].append(tmp_dict)
                    self.task_count_edge += 1

                    # 🔧 安全违规检测（仅记录，不阻止）
                    task_sensitivity = tmp_dict['SENS']
                    edge_trust = self.en_trust_levels[edge_idx]
                    if task_sensitivity >= 2 and edge_trust < 0.9:
                        self.security_violation_count += 1
                    elif task_sensitivity >= 1 and edge_trust < 0.6:
                        self.security_violation_count += 1

                    self.b_edge_comp[ue_index, edge_idx] += self.local_transmit_task[ue_index]['SIZE']
                    self.process_delay_trans[self.local_transmit_task[ue_index]['TIME'], ue_index] = \
                        self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1
                    self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                elif self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1 == self.max_delay:
                    self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                    self.process_delay[self.local_transmit_task[ue_index]['TIME'], ue_index] = self.max_delay
                    self.unfinish_task[self.local_transmit_task[ue_index]['TIME'], ue_index] = 1
                    self.drop_trans_count += 1

        # 更新拥塞状态
        self.edge_ue_m_observe = self.edge_ue_m
        self.edge_ue_m = np.zeros(self.n_edge)
        for edge_index in range(self.n_edge):
            for ue_index in range(self.n_ue):
                if (len(self.edge_computation_queue[ue_index][edge_index]) > 0) or \
                        (isinstance(self.edge_process_task[ue_index][edge_index]['REMAIN'], float) and
                         self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0):
                    self.edge_ue_m[edge_index] += 1

        # 更新负载因子（仅用于观测）
        self.edge_load_factors = self.get_edge_load_factor()

        # 保存动作
        self.previous_actions = action

        # 时间更新
        self.time_count += 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            for time_index in range(self.n_time):
                for ue_index in range(self.n_ue):
                    if self.process_delay[time_index, ue_index] == 0 and self.arrive_task_size[
                        time_index, ue_index] != 0:
                        self.process_delay[time_index, ue_index] = (self.time_count - 1) - time_index + 1
                        self.unfinish_task[time_index, ue_index] = 1

        # 观测
        UEs_OBS_ = np.zeros([self.n_ue, self.n_features])
        UEs_lstm_state_ = np.zeros([self.n_ue, self.n_lstm_state])
        if not done:
            for ue_index in range(self.n_ue):
                next_time_slot = self.time_count
                if next_time_slot < self.n_time and self.arrive_task_size[next_time_slot, ue_index] != 0:
                    scalar_part = np.array([
                        self.arrive_task_size[next_time_slot, ue_index],
                        self.t_ue_comp[ue_index] - next_time_slot + 1,
                        self.t_ue_tran[ue_index] - next_time_slot + 1,
                        self.ue_energy_state[ue_index],
                        self.arrive_task_sens[next_time_slot, ue_index]
                    ])
                    array_part = self.b_edge_comp[ue_index, :]
                    local_processing_ratio = np.sum(self.previous_actions == 0) / self.n_ue
                    total_offload_count = np.sum(self.previous_actions > 0)
                    coordination_features = np.array([local_processing_ratio, total_offload_count])
                    edge_loads = self.edge_load_factors
                    UEs_OBS_[ue_index, :] = np.concatenate(
                        (scalar_part[:3], array_part, scalar_part[3:], coordination_features, edge_loads))
                UEs_lstm_state_[ue_index, :] = np.hstack(self.edge_ue_m_observe)

        return UEs_OBS_, UEs_lstm_state_, done, self.security_violation_count