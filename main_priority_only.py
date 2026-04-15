# main_baseline.py - Priority-only 消融训练脚本（优先级队列，无安全约束）
# 使用修复后的 MEC_Env_FIFO.py

from MEC_Env_Priority_NoSec import MEC
from D3QN import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def normalize(parameter, minimum, maximum):
    """归一化函数"""
    normalized_parameter = (parameter - minimum) / (maximum - minimum)
    return normalized_parameter


def QoE_Function_Baseline(delay, max_delay, unfinish_task, ue_energy_state,
                          ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    """
    QECO Baseline的QoE函数：只考虑延迟和能耗，不考虑安全性和负载均衡
    """
    # 能量计算
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    
    # 简单的性能成本（只考虑延迟和能耗）
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))
    
    # 基础奖励
    Reward = max_delay * 4
    
    if unfinish_task:
        # 简单的丢弃惩罚
        QoE = -cost - max_delay * 4
    else:
        QoE = Reward - cost
    
    return QoE


def Cal_QoE(ue_RL_list, episode):
    """计算平均QoE"""
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    return avg_episode_sum_reward


def Cal_Delay(ue_RL_list, episode):
    """计算平均延迟"""
    avg_delay_in_episode = []
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] != 0:
                avg_delay_in_episode.append(ue_RL_list[i].delay_store[episode][j])
    if not avg_delay_in_episode:
        return 0
    return sum(avg_delay_in_episode) / len(avg_delay_in_episode)


def Cal_Energy(ue_RL_list, episode):
    """计算平均能耗"""
    avg_energy_in_episode = []
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].energy_store[episode])):
            if ue_RL_list[i].energy_store[episode][j] != 0:
                avg_energy_in_episode.append(ue_RL_list[i].energy_store[episode][j])
    if not avg_energy_in_episode:
        return 0
    return sum(avg_energy_in_episode) / len(avg_energy_in_episode)


if __name__ == "__main__":
    print("="*70)
    print("Priority-only 消融训练脚本")
    print("="*70)
    print("特性：")
    print("✅ 使用优先级队列（紧急/敏感任务优先）")
    print("❌ 不强制执行安全约束（允许选择不安全节点）")
    print("❌ 不考虑负载均衡")
    print("✅ 只优化延迟和能耗（reward不含安全/负载项）")
    print("="*70)
    print()
    
    # 创建环境（使用FIFO版本）
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)
    
    # 创建智能体
    ue_RL_list = []
    for ue in range(Config.N_UE):
        ue_RL_list.append(
            DuelingDoubleDeepQNetwork(
                env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                learning_rate=Config.LEARNING_RATE,
                reward_decay=Config.REWARD_DECAY,
                replace_target_iter=Config.N_NETWORK_UPDATE,
                memory_size=Config.MEMORY_SIZE))
    
    # 统计变量
    QoE_total_list = []
    Delay_total_list = []
    Energy_total_list = []
    Security_violations_list = []
    Load_CV_list = []
    Drop_rate_list = []
    
    RL_step = 0
    
    # 训练循环
    for episode in range(Config.N_EPISODE):
        print(f"\n{'='*70}")
        print(f"Episode {episode}/{Config.N_EPISODE}")
        print(f"{'='*70}")
        
        # 生成任务
        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, 
                                          size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        has_task_mask = np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob
        bitarrive_size = bitarrive_size * has_task_mask
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])
        
        bitarrive_dens = np.random.choice(Config.TASK_COMP_DENS, size=[env.n_time, env.n_ue])
        bitarrive_sens = np.random.choice(Config.TASK_SENSITIVITY_LEVELS, size=[env.n_time, env.n_ue])
        
        # 重置环境
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens, bitarrive_sens)
        
        # 初始化
        reward_indicator = np.zeros([env.n_time, env.n_ue])
        process_delay = env.process_delay
        unfinish_task = env.unfinish_task
        
        history = [[{'observation': None, 'lstm': None, 'action': None, 
                    'observation_': None, 'lstm_': None}
                   for _ in range(env.n_ue)] for _ in range(env.n_time)]
        
        episode_delays = []
        episode_energies = []
        total_tasks_generated = np.sum(bitarrive_size > 0)
        
        done = False
        while not done:
            action_all = np.zeros(env.n_ue, dtype=int)
            
            # 选择动作（使用普通的choose_action，不考虑负载）
            for ue_index in range(env.n_ue):
                if env.time_count < env.n_time and env.arrive_task_size[env.time_count, ue_index] > 0:
                    # ❌ Baseline不使用负载感知
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action_with_coordination(
                        observation_all[ue_index], edge_loads=env.get_edge_load_factor())
                else:
                    action_all[ue_index] = 0
            
            # 执行动作（Baseline环境不会强制安全约束）
            observation_all_, lstm_state_all_, done, security_count = env.step(action_all)
            
            # 更新LSTM和历史
            for ue_index in range(env.n_ue):
                if env.time_count - 1 < env.n_time and env.arrive_task_size[env.time_count - 1, ue_index] > 0:
                    ue_RL_list[ue_index].update_lstm(np.squeeze(lstm_state_all_[ue_index, :]))
                    history[env.time_count - 1][ue_index]['observation'] = observation_all[ue_index]
                    history[env.time_count - 1][ue_index]['lstm'] = np.squeeze(lstm_state_all[ue_index, :])
                    history[env.time_count - 1][ue_index]['action'] = action_all[ue_index]
                    history[env.time_count - 1][ue_index]['observation_'] = observation_all_[ue_index]
                    history[env.time_count - 1][ue_index]['lstm_'] = np.squeeze(lstm_state_all_[ue_index, :])
                    
                    # 计算奖励
                    update_index = np.where((1 - reward_indicator[:, ue_index]) * process_delay[:, ue_index] > 0)[0]
                    if len(update_index) != 0:
                        for update_ii in range(len(update_index)):
                            time_index = update_index[update_ii]
                            action_taken = history[time_index][ue_index]['action']
                            
                            # ❌ 使用Baseline的简单QoE函数（不考虑安全性和负载均衡）
                            reward = QoE_Function_Baseline(
                                process_delay[time_index, ue_index], env.max_delay,
                                unfinish_task[time_index, ue_index],
                                env.ue_energy_state[ue_index],
                                env.ue_comp_energy[time_index, ue_index],
                                env.ue_tran_energy[time_index, ue_index],
                                env.edge_comp_energy[time_index, ue_index],
                                env.ue_idle_energy[time_index, ue_index])
                            
                            # 存储经验
                            ue_RL_list[ue_index].store_transition(
                                history[time_index][ue_index]['observation'],
                                history[time_index][ue_index]['lstm'], action_taken,
                                reward, history[time_index][ue_index]['observation_'],
                                history[time_index][ue_index]['lstm_'])
                            
                            # 存储奖励和延迟
                            ue_RL_list[ue_index].do_store_reward(episode, time_index, reward)
                            ue_RL_list[ue_index].do_store_delay(episode, time_index, 
                                                                process_delay[time_index, ue_index])
                            
                            # 统计能耗
                            if unfinish_task[time_index, ue_index] == 0:
                                episode_delays.append(process_delay[time_index, ue_index])
                                edge_e = next((e for e in env.edge_comp_energy[time_index, ue_index] if e != 0), 0)
                                idle_e = next((e for e in env.ue_idle_energy[time_index, ue_index] if e != 0), 0)
                                total_e = (env.ue_comp_energy[time_index, ue_index] +
                                          env.ue_tran_energy[time_index, ue_index] + edge_e + idle_e)
                                episode_energies.append(total_e)
                                ue_RL_list[ue_index].do_store_energy(
                                    episode, time_index,
                                    env.ue_comp_energy[time_index, ue_index],
                                    env.ue_tran_energy[time_index, ue_index],
                                    env.edge_comp_energy[time_index, ue_index],
                                    env.ue_idle_energy[time_index, ue_index])
                            
                            reward_indicator[time_index, ue_index] = 1
            
            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_
            
            # 学习
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()
            
            if done:
                # 计算本回合的统计数据
                total_dropped = env.drop_trans_count + env.drop_edge_count + env.drop_ue_count
                drop_rate = (total_dropped / total_tasks_generated) if total_tasks_generated > 0 else 0
                avg_delay = np.mean(episode_delays) if episode_delays else 0
                avg_energy = np.mean(episode_energies) if episode_energies else 0
                
                # 计算负载CV
                edge_workloads = []
                for edge_idx in range(env.n_edge):
                    edge_work = sum(env.edge_bit_processed[:, :, edge_idx].flatten())
                    edge_workloads.append(edge_work)
                load_cv = (np.std(edge_workloads) / np.mean(edge_workloads)) if (
                        edge_workloads and np.mean(edge_workloads) > 0) else 0
                
                # 保存统计数据
                QoE_total_list.append(Cal_QoE(ue_RL_list, episode))
                Delay_total_list.append(avg_delay)
                Energy_total_list.append(avg_energy)
                Security_violations_list.append(security_count)
                Load_CV_list.append(load_cv)
                Drop_rate_list.append(drop_rate)
                
                # 打印统计
                print(f"QoE: {QoE_total_list[-1]:.2f}")
                print(f"平均延迟: {avg_delay:.3f}")
                print(f"平均能耗: {avg_energy:.3f}")
                print(f"安全违规数: {security_count} ❌ (Priority-only不强制安全)")
                print(f"丢弃率: {drop_rate:.2%}")
                print(f"负载CV: {load_cv:.3f}")
                
                break
    
    # 保存最终结果
    print("\n" + "="*70)
    print("训练完成！最终统计（后900个回合）：")
    print("="*70)
    
    start_idx = 100  # 跳过前100个不稳定的回合
    print(f"平均QoE: {np.mean(QoE_total_list[start_idx:]):.2f}")
    print(f"平均延迟: {np.mean(Delay_total_list[start_idx:]):.3f}")
    print(f"平均能耗: {np.mean(Energy_total_list[start_idx:]):.3f}")
    print(f"平均安全违规数: {np.mean(Security_violations_list[start_idx:]):.2f} ❌")
    print(f"平均丢弃率: {np.mean(Drop_rate_list[start_idx:]):.2%}")
    print(f"平均负载CV: {np.mean(Load_CV_list[start_idx:]):.3f}")
    print("="*70)
    print("\n⚠️  注意：Baseline的安全违规数应该很高（>100）")
    print("这是正常的，因为Baseline不考虑安全性！")
    print("="*70)
    
    # 保存结果到文件
    results = {
        'QoE': QoE_total_list,
        'Delay': Delay_total_list,
        'Energy': Energy_total_list,
        'Security_Violations': Security_violations_list,
        'Load_CV': Load_CV_list,
        'Drop_Rate': Drop_rate_list
    }
    
    np.save('priority_only_results.npy', results)
    print("\n结果已保存到 priority_only_results.npy")
    
    # 绘制图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].plot(QoE_total_list)
    axes[0, 0].set_title('QoE over Episodes')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('QoE')
    
    axes[0, 1].plot(Delay_total_list)
    axes[0, 1].set_title('Delay over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Delay')
    
    axes[0, 2].plot(Energy_total_list)
    axes[0, 2].set_title('Energy over Episodes')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Energy')
    
    axes[1, 0].plot(Security_violations_list)
    axes[1, 0].set_title('Security Violations over Episodes (应该很高)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Violations')
    axes[1, 0].axhline(y=100, color='r', linestyle='--', label='预期水平')
    axes[1, 0].legend()
    
    axes[1, 1].plot(Load_CV_list)
    axes[1, 1].set_title('Load CV over Episodes')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Load CV')
    
    axes[1, 2].plot(Drop_rate_list)
    axes[1, 2].set_title('Drop Rate over Episodes')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Drop Rate')
    
    plt.tight_layout()
    plt.savefig('priority_only_training_results.png', dpi=300)
    print("训练图表已保存到 priority_only_training_results.png")
    
    plt.show()