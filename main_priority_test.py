#
# 文件名: main_priority_test.py (实验专用)
# 描述:
# 1. 导入 MEC_Env (实验组A) 或 MEC_Env_FIFO (对照组B)
# 2. 增加了按任务敏感度分类统计丢弃率的功能
#
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil

# --- 导入环境和算法 ---
# !!!!!!!!!!!! 注意 !!!!!!!!!!!!
# 默认使用 "A组" (有优先级)
from MEC_Env import MEC
# 在 "B组" 实验中, 您需要手动将上面这行注释掉
# 然后取消注释下面这行
#from MEC_Env_FIFO import MEC
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

from D3QN import DuelingDoubleDeepQNetwork
from Config import Config


def normalize(parameter, minimum, maximum):
    normalized_parameter = (parameter - minimum) / (maximum - minimum)
    return normalized_parameter


def QoE_Function_SeCO_v2(delay, max_delay, unfinish_task, ue_energy_state,
                         ue_comp_energy, ue_trans_energy, edge_comp_energy,
                         ue_idle_energy, action, task_sensitivity, en_trust_levels,
                         edge_load_factors=None):
    """
    改进的QoE函数，包含负载均衡奖励
    """
    # 能量计算
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10

    # 性能成本（考虑能量状态）
    performance_cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))

    # 安全惩罚
    security_penalty = 0
    if action > 0:
        chosen_en_index = int(action - 1)
        if 0 <= chosen_en_index < len(en_trust_levels):
            chosen_en_trust = en_trust_levels[chosen_en_index]
            if chosen_en_trust < 0.5 and task_sensitivity >= 2:
                security_penalty = Config.SECURITY_PENALTY_MULTIPLIER * (1 - chosen_en_trust) * task_sensitivity

    # 负载均衡奖励/惩罚
    load_balance_bonus = 0
    if action > 0 and edge_load_factors is not None:
        chosen_edge = int(action - 1)
        avg_load = np.mean(edge_load_factors)

        if edge_load_factors[chosen_edge] < avg_load:
            # 选择低负载服务器给予奖励
            load_balance_bonus = 10 * (1 - edge_load_factors[chosen_edge])
        else:
            # 选择高负载服务器给予惩罚
            load_balance_bonus = -10 * edge_load_factors[chosen_edge]

    # 时延感知奖励
    delay_ratio = delay / max_delay
    if delay_ratio < 0.3:  # 快速完成任务
        delay_bonus = 20 * (1 - delay_ratio)
    elif delay_ratio > 0.7:  # 接近截止时间
        delay_bonus = -20 * delay_ratio
    else:
        delay_bonus = 0

    # 总成本
    total_cost = performance_cost + security_penalty - load_balance_bonus - delay_bonus

    # 基础奖励
    Reward = max_delay * 4

    if unfinish_task:
        # 未完成任务的严重惩罚（根据任务敏感度加权）
        QoE = -total_cost - 50 * (1 + task_sensitivity)
    else:
        QoE = Reward - total_cost

    return QoE


def Cal_QoE(ue_RL_list, episode):
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    return avg_episode_sum_reward


def Cal_Delay(ue_RL_list, episode):
    avg_delay_in_episode = []
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] != 0:
                avg_delay_in_episode.append(ue_RL_list[i].delay_store[episode][j])
    if not avg_delay_in_episode: return 0
    return sum(avg_delay_in_episode) / len(avg_delay_in_episode)


def Cal_Energy(ue_RL_list, episode):
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    return avg_energy_in_episode


def train(ue_RL_list, NUM_EPISODE):
    avg_QoE_list = []
    avg_delay_list = []
    energy_cons_list = []
    num_drop_list = []
    load_balance_list = []

    # --- 新增：用于存储所有回合的分类丢弃率 ---
    all_drop_rates_sens_0 = []
    all_drop_rates_sens_1 = []
    all_drop_rates_sens_2 = []

    RL_step = 0

    for episode in range(NUM_EPISODE):
        print("\n-*-**-***-*****-********-*************-********-*****-***-**-*-")
        print("Episode  :", episode, )
        print("Epsilon  :", ue_RL_list[0].epsilon)

        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        has_task_mask = np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob
        bitarrive_size = bitarrive_size * has_task_mask
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        bitarrive_dens = np.zeros([env.n_time, env.n_ue])
        for i in range(env.n_time):
            for j in range(env.n_ue):
                if bitarrive_size[i, j] != 0:
                    bitarrive_dens[i, j] = random.choice(Config.TASK_COMP_DENS)

        bitarrive_sens = np.zeros([env.n_time, env.n_ue])
        for i in range(env.n_time):
            for j in range(env.n_ue):
                if bitarrive_size[i, j] != 0:
                    bitarrive_sens[i, j] = random.choice(Config.TASK_SENSITIVITY_LEVELS)

        # --- 新增：统计本回合生成的各优先级任务总数 ---
        # 我们只关心有任务到达的地方 (bitarrive_size != 0)
        mask_has_task = (bitarrive_size != 0)
        total_tasks_sens_0 = np.sum(bitarrive_sens[mask_has_task] == 0)
        total_tasks_sens_1 = np.sum(bitarrive_sens[mask_has_task] == 1)
        total_tasks_sens_2 = np.sum(bitarrive_sens[mask_has_task] == 2)
        print(
            f"Tasks Generated: Sens-0: {total_tasks_sens_0}, Sens-1: {total_tasks_sens_1}, Sens-2: {total_tasks_sens_2}")

        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features), 'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan, 'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens, bitarrive_sens)

        episode_load_balance_values = []

        while True:
            action_all = np.zeros([env.n_ue])
            edge_loads = env.get_edge_load_factor() if hasattr(env, 'get_edge_load_factor') else None

            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    action_all[ue_index] = 0
                else:
                    if hasattr(ue_RL_list[ue_index], 'choose_action_with_coordination'):
                        action_all[ue_index] = ue_RL_list[ue_index].choose_action_with_coordination(observation,
                                                                                                    edge_loads)
                    else:
                        action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)

                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            observation_all_, lstm_state_all_, done = env.step(action_all)

            if hasattr(env, 'edge_load_factors'):
                current_load_balance = np.std(env.edge_load_factors)
                episode_load_balance_values.append(current_load_balance)

            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index, :])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task
            edge_loads_for_reward = env.get_edge_load_factor() if hasattr(env, 'get_edge_load_factor') else None

            for ue_index in range(env.n_ue):
                history[env.time_count - 1][ue_index]['observation'] = observation_all[ue_index, :]
                history[env.time_count - 1][ue_index]['lstm'] = np.squeeze(lstm_state_all[ue_index, :])
                history[env.time_count - 1][ue_index]['action'] = action_all[ue_index]
                history[env.time_count - 1][ue_index]['observation_'] = observation_all_[ue_index]
                history[env.time_count - 1][ue_index]['lstm_'] = np.squeeze(lstm_state_all_[ue_index, :])

                update_index = np.where((1 - reward_indicator[:, ue_index]) * process_delay[:, ue_index] > 0)[0]
                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        action_taken = history[time_index][ue_index]['action']
                        task_sensitivity = env.arrive_task_sens[time_index, ue_index]

                        reward = QoE_Function_SeCO_v2(process_delay[time_index, ue_index], env.max_delay,
                                                      unfinish_task[time_index, ue_index],
                                                      env.ue_energy_state[ue_index],
                                                      env.ue_comp_energy[time_index, ue_index],
                                                      env.ue_tran_energy[time_index, ue_index],
                                                      env.edge_comp_energy[time_index, ue_index],
                                                      env.ue_idle_energy[time_index, ue_index], action_taken,
                                                      task_sensitivity, env.en_trust_levels,
                                                      edge_loads_for_reward)

                        ue_RL_list[ue_index].store_transition(history[time_index][ue_index]['observation'],
                                                              history[time_index][ue_index]['lstm'], action_taken,
                                                              reward, history[time_index][ue_index]['observation_'],
                                                              history[time_index][ue_index]['lstm_'])
                        ue_RL_list[ue_index].do_store_reward(episode, time_index, reward)
                        ue_RL_list[ue_index].do_store_delay(episode, time_index, process_delay[time_index, ue_index])
                        ue_RL_list[ue_index].do_store_energy(episode, time_index,
                                                             env.ue_comp_energy[time_index, ue_index],
                                                             env.ue_tran_energy[time_index, ue_index],
                                                             env.edge_comp_energy[time_index, ue_index],
                                                             env.ue_idle_energy[time_index, ue_index])
                        reward_indicator[time_index, ue_index] = 1

            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            if done:
                avg_delay = Cal_Delay(ue_RL_list, episode)
                avg_energy = Cal_Energy(ue_RL_list, episode)
                avg_QoE = Cal_QoE(ue_RL_list, episode)
                total_dropped = env.drop_trans_count + env.drop_edge_count + env.drop_ue_count
                ue_comp_energy = sum(sum(env.ue_comp_energy))
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_tran_energy = sum(sum(env.ue_tran_energy))
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                edge_comp_energy = sum(sum(sum(env.edge_comp_energy)))
                ue_idle_energy = sum(sum(sum(env.ue_idle_energy)))
                total_edge_energy = edge_comp_energy + ue_idle_energy
                edge_bit_processed = sum(sum(sum(env.edge_bit_processed)))

                if episode_load_balance_values:
                    avg_load_balance = np.mean(episode_load_balance_values)
                else:
                    avg_load_balance = 0.0

                edge_workloads = []
                for edge_idx in range(env.n_edge):
                    edge_work = sum(env.edge_bit_processed[:, :, edge_idx].flatten())
                    edge_workloads.append(edge_work)

                if edge_workloads and np.mean(edge_workloads) > 0:
                    load_cv = np.std(edge_workloads) / np.mean(edge_workloads)
                else:
                    load_cv = 0.0

                print("SystemPerformance: -----------------")
                print(
                    f"{'Num_Dropped':<15} : {total_dropped} [Trans_Drop: {env.drop_trans_count} Edge_Drop: {env.drop_edge_count} UE_Drop: {env.drop_ue_count} ]")
                print(f"{'Avg_Delay':<15} : {avg_delay:.1f}")
                print(f"{'Avg_Energy':<15} : {avg_energy:.1f}")
                print(f"{'Avg_QoE':<15} : {avg_QoE:.1f}")
                print(f"{'Load_Balance':<15} : {avg_load_balance:.3f} (CV: {load_cv:.3f})")
                print("EnergyCosumption: ------------------")
                print(f"{'Local':<15} : {ue_comp_energy:.1f} [ue_bit_processed: {int(ue_bit_processed)} ]")
                print(f"{'Trans':<15} : {ue_tran_energy:.1f} [ue_bit_transmitted: {int(ue_bit_transmitted)} ]")
                print(f"{'Edges':<15} : {total_edge_energy:.1f} [edge_bit_processed : {int(edge_bit_processed)} ]")

                # --- 新增：计算并打印分类丢弃率 ---
                mask_was_dropped = (env.unfinish_task == 1)
                dropped_tasks_sens_0 = np.sum(env.arrive_task_sens[mask_was_dropped] == 0)
                dropped_tasks_sens_1 = np.sum(env.arrive_task_sens[mask_was_dropped] == 1)
                dropped_tasks_sens_2 = np.sum(env.arrive_task_sens[mask_was_dropped] == 2)

                drop_rate_sens_0 = (dropped_tasks_sens_0 / total_tasks_sens_0) if total_tasks_sens_0 > 0 else 0
                drop_rate_sens_1 = (dropped_tasks_sens_1 / total_tasks_sens_1) if total_tasks_sens_1 > 0 else 0
                drop_rate_sens_2 = (dropped_tasks_sens_2 / total_tasks_sens_2) if total_tasks_sens_2 > 0 else 0

                print("Priority Performance: ---------------")
                print(f"Sens-0 Drop Rate: {drop_rate_sens_0:.2%} ({dropped_tasks_sens_0}/{total_tasks_sens_0})")
                print(f"Sens-1 Drop Rate: {drop_rate_sens_1:.2%} ({dropped_tasks_sens_1}/{total_tasks_sens_1})")
                print(f"Sens-2 Drop Rate: {drop_rate_sens_2:.2%} ({dropped_tasks_sens_2}/{total_tasks_sens_2})")

                # 存储用于后续平均
                all_drop_rates_sens_0.append(drop_rate_sens_0)
                all_drop_rates_sens_1.append(drop_rate_sens_1)
                all_drop_rates_sens_2.append(drop_rate_sens_2)
                # --- 结束新增 ---

                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(total_dropped)
                load_balance_list.append(avg_load_balance)

                if episode > 0 and episode % 10 == 0:
                    # (绘图部分不变)
                    fig, axs = plt.subplots(5, 1, figsize=(10, 25))
                    fig.suptitle(f'Performance Metrics Over Episodes (Episode {episode})', fontsize=16, y=0.92)
                    axs[0].plot(avg_QoE_list, marker='o', linestyle='-', color='b', label='Avg QoE', markersize=4)
                    axs[0].set_ylabel('Average QoE')
                    axs[0].set_title('Average QoE')
                    axs[0].grid(True)
                    axs[0].legend()
                    axs[1].plot(avg_delay_list, marker='s', linestyle='-', color='g', label='Avg Delay', markersize=4)
                    axs[1].set_ylabel('Average Delay')
                    axs[1].set_title('Average Delay')
                    axs[1].grid(True)
                    axs[1].legend()
                    axs[2].plot(energy_cons_list, marker='^', linestyle='-', color='r', label='Energy Cons.',
                                markersize=4)
                    axs[2].set_ylabel('Energy Consumption')
                    axs[2].set_title('Energy Consumption')
                    axs[2].grid(True)
                    axs[2].legend()
                    axs[3].plot(num_drop_list, marker='x', linestyle='-', color='m', label='Num Drops', markersize=6)
                    axs[3].set_ylabel('Number Drops')  # 修正了这里的拼写错误
                    axs[3].set_title('Number of Dropped Tasks')
                    axs[3].grid(True)
                    axs[3].legend()
                    axs[4].plot(load_balance_list, marker='d', linestyle='-', color='orange', label='Load Balance',
                                markersize=4)
                    axs[4].set_ylabel('Load Balance (Std Dev)')
                    axs[4].set_xlabel('Episode')
                    axs[4].set_title('Load Balance (Lower is Better)')
                    axs[4].grid(True)
                    axs[4].legend()
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig('Performance_Chart_Improved.png', dpi=100, bbox_inches='tight')
                    print(f"\n>>> Performance chart updated at episode {episode} <<<\n")
                    plt.close(fig)

                break

    # --- 新增：打印所有回合的平均丢弃率 ---
    print("\n\n--- FINAL AVERAGE RESULTS ---")
    print(f"Avg Sens-0 Drop Rate: {np.mean(all_drop_rates_sens_0):.2%}")
    print(f"Avg Sens-1 Drop Rate: {np.mean(all_drop_rates_sens_1):.2%}")
    print(f"Avg Sens-2 Drop Rate: {np.mean(all_drop_rates_sens_2):.2%}")
    print("-----------------------------\n")


if __name__ == "__main__":
    # --- MODIFICATION: 确保env是MEC类的一个实例 ---
    # 根据 main_priority_test.py 顶部的 import 语句，
    # env 将是 MEC_Env.MEC 或 MEC_Env_FIFO.MEC 的实例
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate=Config.LEARNING_RATE,
                                                    reward_decay=Config.REWARD_DECAY,
                                                    replace_target_iter=Config.N_NETWORK_UPDATE,
                                                    memory_size=Config.MEMORY_SIZE
                                                    ))
    train(ue_RL_list, Config.N_EPISODE)
