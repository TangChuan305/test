# run_comparison_fixed.py (完整修复版)
# 🔧 核心修复：
# 1. RoSCo使用优先级队列 + 强制安全约束
# 2. QECO使用FIFO队列 + 不考虑安全性
# 3. 正确对比物理指标

import sys
import os
import numpy as np
import random

# 复制修复版的环境文件到当前目录
import shutil

shutil.copy('/home/claude/MEC_Env_Fixed.py', './MEC_Env_RoSCo.py')
shutil.copy('/home/claude/MEC_Env_FIFO_Fixed.py', './MEC_Env_Baseline.py')

# 导入配置和算法
from Config import Config
from D3QN import DuelingDoubleDeepQNetwork

print("=" * 70)
print("🔧 实验设置说明")
print("=" * 70)
print("QECO Baseline:")
print("  - 使用FIFO队列（先进先出）")
print("  - 不考虑安全性（允许任何动作）")
print("  - QoE函数：只关注延迟和能耗")
print("\nRoSCo (您的算法):")
print("  - 使用优先级队列（紧急任务优先）")
print("  - 强制执行安全约束（不安全动作自动改为本地处理）")
print("  - QoE函数：考虑延迟、能耗、安全性和负载均衡")
print("=" * 70)
print()


def normalize(parameter, minimum, maximum):
    return (parameter - minimum) / (maximum - minimum)


def QoE_Function_Baseline(delay, max_delay, unfinish_task, ue_energy_state,
                          ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    """
    QECO Baseline的QoE函数：只关心延迟和能耗，不考虑安全性
    """
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10

    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))
    Reward = max_delay * 4

    if unfinish_task:
        QoE = -cost - max_delay * 4
    else:
        QoE = Reward - cost

    return QoE


def QoE_Function_RoSCo(delay, max_delay, unfinish_task, ue_energy_state,
                       ue_comp_energy, ue_trans_energy, edge_comp_energy,
                       ue_idle_energy, action, task_sensitivity, en_trust_levels,
                       edge_load_factors=None):
    """
    RoSCo的QoE函数：考虑延迟、能耗、安全性和负载均衡
    """
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    performance_cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))

    # 安全惩罚
    security_penalty = 0
    if action > 0:
        chosen_en_index = int(action - 1)
        if 0 <= chosen_en_index < len(en_trust_levels):
            chosen_en_trust = en_trust_levels[chosen_en_index]
            if chosen_en_trust < 0.9 and task_sensitivity >= 2:
                security_penalty = Config.SECURITY_PENALTY_MULTIPLIER * (1 - chosen_en_trust) * task_sensitivity
            elif chosen_en_trust < 0.6 and task_sensitivity >= 1:
                security_penalty = Config.SECURITY_PENALTY_MULTIPLIER * 0.5 * (1 - chosen_en_trust) * task_sensitivity

    # 负载均衡奖励
    load_balance_bonus = 0
    if action > 0 and edge_load_factors is not None:
        chosen_edge = int(action - 1)
        if chosen_edge < len(edge_load_factors):
            chosen_load = edge_load_factors[chosen_edge]
            avg_load = np.mean(edge_load_factors)

            SCALE = Config.LOAD_BALANCE_REWARD_SCALE * 5
            THRESHOLD = avg_load + Config.LOAD_THRESHOLD_OFFSET

            if chosen_load < avg_load:
                load_balance_bonus = SCALE * (1 - chosen_load)
            else:
                excess_load = chosen_load - avg_load
                if chosen_load > THRESHOLD:
                    load_balance_bonus = -SCALE * (1 + excess_load) ** 2
                else:
                    load_balance_bonus = -SCALE * chosen_load * Config.LOAD_PENALTY_FACTOR

    # 延迟奖励
    delay_ratio = delay / max_delay
    if delay_ratio < 0.3:
        delay_bonus = 20 * (1 - delay_ratio)
    elif delay_ratio > 0.7:
        delay_bonus = -20 * delay_ratio
    else:
        delay_bonus = 0

    total_cost = performance_cost + security_penalty - load_balance_bonus - delay_bonus

    Reward = max_delay * 4
    if unfinish_task:
        QoE = -total_cost - 50 * (1 + task_sensitivity)
    else:
        QoE = Reward - total_cost
    return QoE


def train_model(ue_RL_list, env, mode="RoSCo", num_episodes=100):
    """
    训练模型并收集统计数据
    """
    print(f"\n{'=' * 70}")
    print(f"开始训练: {mode}")
    print(f"{'=' * 70}\n")

    # 统计指标
    physical_delays = []
    physical_energies = []
    physical_drop_rates = []
    physical_security_violations = []
    physical_load_cvs = []

    RL_step = 0

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"[{mode}] Episode: {episode}/{num_episodes}")

        # 生成任务
        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        has_task_mask = np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob
        bitarrive_size = bitarrive_size * has_task_mask
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        bitarrive_dens = np.random.choice(Config.TASK_COMP_DENS, size=[env.n_time, env.n_ue])
        bitarrive_sens = np.random.choice(Config.TASK_SENSITIVITY_LEVELS, size=[env.n_time, env.n_ue])

        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens, bitarrive_sens)

        # 存储任务和奖励
        reward_indicator = np.zeros([env.n_time, env.n_ue])
        process_delay = env.process_delay
        unfinish_task = env.unfinish_task

        history = [[{'observation': None, 'lstm': None, 'action': None, 'observation_': None, 'lstm_': None}
                    for _ in range(env.n_ue)] for _ in range(env.n_time)]

        episode_delays = []
        episode_energies = []
        total_tasks_generated = np.sum(bitarrive_size > 0)
        security_violations_total = 0

        done = False
        while not done:
            action_all = np.zeros(env.n_ue, dtype=int)
            edge_loads_for_reward = env.get_edge_load_factor()

            for ue_index in range(env.n_ue):
                if env.time_count < env.n_time and env.arrive_task_size[env.time_count, ue_index] > 0:
                    if mode == "RoSCo":
                        action_all[ue_index] = ue_RL_list[ue_index].choose_action_with_coordination(
                            observation_all[ue_index], edge_loads=edge_loads_for_reward)
                    else:  # Baseline
                        action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation_all[ue_index])
                else:
                    action_all[ue_index] = 0

            observation_all_, lstm_state_all_, done, security_count = env.step(action_all)
            security_violations_total = security_count

            for ue_index in range(env.n_ue):
                if env.time_count - 1 < env.n_time and env.arrive_task_size[env.time_count - 1, ue_index] > 0:
                    ue_RL_list[ue_index].update_lstm(np.squeeze(lstm_state_all_[ue_index, :]))
                    history[env.time_count - 1][ue_index]['observation'] = observation_all[ue_index]
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

                            if mode == "RoSCo":
                                reward = QoE_Function_RoSCo(
                                    process_delay[time_index, ue_index], env.max_delay,
                                    unfinish_task[time_index, ue_index],
                                    env.ue_energy_state[ue_index],
                                    env.ue_comp_energy[time_index, ue_index],
                                    env.ue_tran_energy[time_index, ue_index],
                                    env.edge_comp_energy[time_index, ue_index],
                                    env.ue_idle_energy[time_index, ue_index],
                                    action_taken, task_sensitivity, env.en_trust_levels,
                                    edge_loads_for_reward)
                            else:  # Baseline
                                reward = QoE_Function_Baseline(
                                    process_delay[time_index, ue_index], env.max_delay,
                                    unfinish_task[time_index, ue_index],
                                    env.ue_energy_state[ue_index],
                                    env.ue_comp_energy[time_index, ue_index],
                                    env.ue_tran_energy[time_index, ue_index],
                                    env.edge_comp_energy[time_index, ue_index],
                                    env.ue_idle_energy[time_index, ue_index])

                            ue_RL_list[ue_index].store_transition(
                                history[time_index][ue_index]['observation'],
                                history[time_index][ue_index]['lstm'], action_taken,
                                reward, history[time_index][ue_index]['observation_'],
                                history[time_index][ue_index]['lstm_'])

                            if unfinish_task[time_index, ue_index] == 0:
                                episode_delays.append(process_delay[time_index, ue_index])
                                edge_e = next((e for e in env.edge_comp_energy[time_index, ue_index] if e != 0), 0)
                                idle_e = next((e for e in env.ue_idle_energy[time_index, ue_index] if e != 0), 0)
                                total_e = (env.ue_comp_energy[time_index, ue_index] +
                                           env.ue_tran_energy[time_index, ue_index] + edge_e + idle_e)
                                episode_energies.append(total_e)

                            reward_indicator[time_index, ue_index] = 1

            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            if done:
                total_dropped = env.drop_trans_count + env.drop_edge_count + env.drop_ue_count
                drop_rate = (total_dropped / total_tasks_generated) if total_tasks_generated > 0 else 0
                avg_delay = np.mean(episode_delays) if episode_delays else 0
                avg_energy = np.mean(episode_energies) if episode_energies else 0

                edge_workloads = []
                for edge_idx in range(env.n_edge):
                    edge_work = sum(env.edge_bit_processed[:, :, edge_idx].flatten())
                    edge_workloads.append(edge_work)

                load_cv = (np.std(edge_workloads) / np.mean(edge_workloads)) if (
                        edge_workloads and np.mean(edge_workloads) > 0) else 0

                if episode % 10 == 0:
                    print(f"  Delay: {avg_delay:.3f} | Drop: {drop_rate:.2%} | "
                          f"Security Violations: {security_violations_total} | Load CV: {load_cv:.3f}")

                physical_delays.append(avg_delay)
                physical_energies.append(avg_energy)
                physical_drop_rates.append(drop_rate)
                physical_security_violations.append(security_violations_total)
                physical_load_cvs.append(load_cv)

                break

    # 计算最终平均值（去掉前100个episode的不稳定数据）
    start_episode = min(100, num_episodes // 2)

    results = {
        "Algorithm": mode,
        "Avg Delay": np.mean(physical_delays[start_episode:]),
        "Avg Energy": np.mean(physical_energies[start_episode:]),
        "Avg Drop Rate": np.mean(physical_drop_rates[start_episode:]),
        "Avg Security Violations": np.mean(physical_security_violations[start_episode:]),
        "Avg Load CV": np.mean(physical_load_cvs[start_episode:])
    }
    return results


if __name__ == "__main__":
    print("\n🚀 开始对比实验...\n")

    NUM_EPISODES = Config.N_EPISODE  # 使用Config中定义的回合数

    # ======================================================================
    # 实验一: QECO Baseline (FIFO队列，不考虑安全)
    # ======================================================================
    print("\n" + "=" * 70)
    print("实验一: QECO Baseline")
    print("=" * 70)

    import MEC_Env_Baseline as MEC_Baseline

    env_baseline = MEC_Baseline.MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME,
                                    Config.N_COMPONENT, Config.MAX_DELAY)

    ue_RL_list_baseline = []
    for ue in range(Config.N_UE):
        ue_RL_list_baseline.append(
            DuelingDoubleDeepQNetwork(
                env_baseline.n_actions, env_baseline.n_features,
                env_baseline.n_lstm_state, env_baseline.n_time,
                learning_rate=Config.LEARNING_RATE,
                reward_decay=Config.REWARD_DECAY,
                replace_target_iter=Config.N_NETWORK_UPDATE,
                memory_size=Config.MEMORY_SIZE))

    baseline_results = train_model(ue_RL_list_baseline, env_baseline,
                                   mode="Baseline", num_episodes=NUM_EPISODES)

    # ======================================================================
    # 实验二: RoSCo (优先级队列，强制安全约束)
    # ======================================================================
    print("\n" + "=" * 70)
    print("实验二: RoSCo")
    print("=" * 70)

    import MEC_Env_RoSCo as MEC_RoSCo

    env_rosco = MEC_RoSCo.MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME,
                              Config.N_COMPONENT, Config.MAX_DELAY)

    ue_RL_list_rosco = []
    for ue in range(Config.N_UE):
        ue_RL_list_rosco.append(
            DuelingDoubleDeepQNetwork(
                env_rosco.n_actions, env_rosco.n_features,
                env_rosco.n_lstm_state, env_rosco.n_time,
                learning_rate=Config.LEARNING_RATE,
                reward_decay=Config.REWARD_DECAY,
                replace_target_iter=Config.N_NETWORK_UPDATE,
                memory_size=Config.MEMORY_SIZE))

    rosco_results = train_model(ue_RL_list_rosco, env_rosco,
                                mode="RoSCo", num_episodes=NUM_EPISODES)

    # ======================================================================
    # 最终结果对比
    # ======================================================================
    print("\n\n" + "=" * 70)
    print("           FINAL COMPARISON (Physical Metrics)")
    print("=" * 70)

    print(f"{'Metric':<30} | {'Baseline (QECO)':<18} | {'RoSCo (Yours)':<15}")
    print("-" * 70)
    print(f"{'Avg Delay (ms)':<30} | {baseline_results['Avg Delay'] * 100:<18.2f} | "
          f"{rosco_results['Avg Delay'] * 100:<15.2f}")
    print(f"{'Avg Drop Rate (%)':<30} | {baseline_results['Avg Drop Rate'] * 100:<18.2f} | "
          f"{rosco_results['Avg Drop Rate'] * 100:<15.2f}")
    print(f"{'Avg Load CV (低=好)':<30} | {baseline_results['Avg Load CV']:<18.3f} | "
          f"{rosco_results['Avg Load CV']:<15.3f}")
    print(f"{'Avg Security Violations':<30} | {baseline_results['Avg Security Violations']:<18.2f} | "
          f"{rosco_results['Avg Security Violations']:<15.2f}")

    print("\n" + "=" * 70)
    print("📊 预期结果分析:")
    print("=" * 70)
    print("✅ QECO (Baseline):")
    print("   - 较低的延迟（不顾一切追求性能）")
    print("   - 非常高的安全违规数（>100，因为不考虑安全）")
    print("   - 较高的负载CV（负载不均衡）")
    print("\n✅ RoSCo (您的算法):")
    print("   - 极低的安全违规数（应接近0，因为强制执行安全约束）")
    print("   - 较低的负载CV（负载更均衡）")
    print("   - 较低的丢弃率（优先级队列保护紧急任务）")
    print("   - 延迟稍高（为了安全性和负载均衡做出的合理权衡）")
    print("=" * 70)

    # 额外的诊断信息
    print("\n📋 诊断信息:")
    print(f"RoSCo被阻止的不安全动作数: {env_rosco.security_blocked_count}")
    print(f"RoSCo实际发生的安全违规数: {env_rosco.security_violation_count}")
    print("（如果安全约束正确工作，实际违规数应该为0或接近0）")
    print("=" * 70)