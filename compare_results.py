# compare_results.py - 对比RoSCo和QECO Baseline的实验结果
# 读取 rosco_results.npy 和 baseline_results.npy，生成对比报告和图表

import numpy as np
import matplotlib.pyplot as plt
import os


def load_results():
    """加载两个算法的结果"""
    if not os.path.exists('rosco_results.npy'):
        print("❌ 错误：找不到 rosco_results.npy")
        print("请先运行 main_rosco.py")
        return None, None
    
    if not os.path.exists('baseline_results.npy'):
        print("❌ 错误：找不到 baseline_results.npy")
        print("请先运行 main_baseline.py")
        return None, None
    
    rosco_results = np.load('rosco_results.npy', allow_pickle=True).item()
    baseline_results = np.load('baseline_results.npy', allow_pickle=True).item()
    
    print("✅ 成功加载结果文件")
    return rosco_results, baseline_results


def calculate_statistics(results, start_idx=100):
    """计算统计数据（跳过前100个不稳定的回合）"""
    stats = {}
    stats['Avg_QoE'] = np.mean(results['QoE'][start_idx:])
    stats['Avg_Delay'] = np.mean(results['Delay'][start_idx:])
    stats['Avg_Energy'] = np.mean(results['Energy'][start_idx:])
    stats['Avg_Security_Violations'] = np.mean(results['Security_Violations'][start_idx:])
    stats['Avg_Load_CV'] = np.mean(results['Load_CV'][start_idx:])
    stats['Avg_Drop_Rate'] = np.mean(results['Drop_Rate'][start_idx:])
    
    stats['Std_QoE'] = np.std(results['QoE'][start_idx:])
    stats['Std_Delay'] = np.std(results['Delay'][start_idx:])
    stats['Std_Energy'] = np.std(results['Energy'][start_idx:])
    stats['Std_Security_Violations'] = np.std(results['Security_Violations'][start_idx:])
    stats['Std_Load_CV'] = np.std(results['Load_CV'][start_idx:])
    stats['Std_Drop_Rate'] = np.std(results['Drop_Rate'][start_idx:])
    
    return stats


def print_comparison_table(rosco_stats, baseline_stats):
    """打印对比表格"""
    print("\n" + "="*80)
    print("           FINAL COMPARISON (Physical Metrics)")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline (QECO)':<20} | {'RoSCo (Yours)':<20}")
    print("-"*80)
    
    # 延迟（转换为毫秒）
    print(f"{'Avg Delay (ms)':<30} | {baseline_stats['Avg_Delay']*100:<20.2f} | "
          f"{rosco_stats['Avg_Delay']*100:<20.2f}")
    
    # 丢弃率（转换为百分比）
    print(f"{'Avg Drop Rate (%)':<30} | {baseline_stats['Avg_Drop_Rate']*100:<20.2f} | "
          f"{rosco_stats['Avg_Drop_Rate']*100:<20.2f}")
    
    # 负载CV
    print(f"{'Avg Load CV (低=好)':<30} | {baseline_stats['Avg_Load_CV']:<20.3f} | "
          f"{rosco_stats['Avg_Load_CV']:<20.3f}")
    
    # 安全违规数 ⭐ 关键指标
    print(f"{'Avg Security Violations ⭐':<30} | {baseline_stats['Avg_Security_Violations']:<20.2f} | "
          f"{rosco_stats['Avg_Security_Violations']:<20.2f}")
    
    # 能耗
    print(f"{'Avg Energy':<30} | {baseline_stats['Avg_Energy']:<20.3f} | "
          f"{rosco_stats['Avg_Energy']:<20.3f}")
    
    # QoE
    print(f"{'Avg QoE':<30} | {baseline_stats['Avg_QoE']:<20.2f} | "
          f"{rosco_stats['Avg_QoE']:<20.2f}")
    
    print("="*80)


def print_improvement_analysis(rosco_stats, baseline_stats):
    """打印改进分析"""
    print("\n" + "="*80)
    print("           IMPROVEMENT ANALYSIS")
    print("="*80)
    
    # 计算改进百分比
    security_improvement = (baseline_stats['Avg_Security_Violations'] - 
                           rosco_stats['Avg_Security_Violations']) / \
                           baseline_stats['Avg_Security_Violations'] * 100
    
    load_improvement = (baseline_stats['Avg_Load_CV'] - rosco_stats['Avg_Load_CV']) / \
                       baseline_stats['Avg_Load_CV'] * 100
    
    drop_improvement = (baseline_stats['Avg_Drop_Rate'] - rosco_stats['Avg_Drop_Rate']) / \
                       baseline_stats['Avg_Drop_Rate'] * 100
    
    delay_change = (rosco_stats['Avg_Delay'] - baseline_stats['Avg_Delay']) / \
                   baseline_stats['Avg_Delay'] * 100
    
    print(f"🔒 安全性改进：{security_improvement:+.1f}%")
    if security_improvement > 90:
        print("   ✅ 优秀！安全违规减少超过90%")
    elif security_improvement > 70:
        print("   ✅ 良好！安全违规显著减少")
    else:
        print("   ⚠️  警告：安全改进不足，请检查代码")
    
    print(f"\n⚖️  负载均衡改进：{load_improvement:+.1f}%")
    if load_improvement > 40:
        print("   ✅ 优秀！负载分布显著改善")
    elif load_improvement > 20:
        print("   ✅ 良好！负载分布有所改善")
    else:
        print("   ⚠️  负载均衡改进较小")
    
    print(f"\n📉 丢弃率改进：{drop_improvement:+.1f}%")
    if drop_improvement > 20:
        print("   ✅ 优秀！任务丢弃率显著降低")
    elif drop_improvement > 0:
        print("   ✅ 良好！任务丢弃率有所降低")
    else:
        print("   ⚠️  丢弃率未改善")
    
    print(f"\n⏱️  延迟变化：{delay_change:+.1f}%")
    if abs(delay_change) < 5:
        print("   ✅ 优秀！延迟增加在可接受范围内（<5%）")
    elif delay_change < 10:
        print("   ✅ 可接受！延迟略有增加但换来了安全性")
    else:
        print("   ⚠️  延迟增加较多，可能需要调整参数")
    
    print("="*80)


def print_verification_checklist(rosco_stats, baseline_stats):
    """打印验证清单"""
    print("\n" + "="*80)
    print("           VERIFICATION CHECKLIST")
    print("="*80)
    
    checks_passed = 0
    total_checks = 5
    
    # 检查1：RoSCo的安全违规数应该很低
    check1 = rosco_stats['Avg_Security_Violations'] < 10
    print(f"{'✅' if check1 else '❌'} 检查1: RoSCo安全违规数 < 10")
    print(f"   实际值: {rosco_stats['Avg_Security_Violations']:.2f}")
    if check1:
        checks_passed += 1
    else:
        print("   ⚠️  RoSCo的安全违规数应该接近0，请检查MEC_Env.py")
    
    # 检查2：Baseline的安全违规数应该很高
    check2 = baseline_stats['Avg_Security_Violations'] > 80
    print(f"\n{'✅' if check2 else '❌'} 检查2: Baseline安全违规数 > 80")
    print(f"   实际值: {baseline_stats['Avg_Security_Violations']:.2f}")
    if check2:
        checks_passed += 1
    else:
        print("   ⚠️  Baseline的安全违规数应该很高，请检查MEC_Env_FIFO.py")
    
    # 检查3：RoSCo的负载CV应该更低
    check3 = rosco_stats['Avg_Load_CV'] < baseline_stats['Avg_Load_CV']
    print(f"\n{'✅' if check3 else '❌'} 检查3: RoSCo负载CV < Baseline负载CV")
    print(f"   RoSCo: {rosco_stats['Avg_Load_CV']:.3f} vs Baseline: {baseline_stats['Avg_Load_CV']:.3f}")
    if check3:
        checks_passed += 1
    else:
        print("   ⚠️  RoSCo应该有更好的负载均衡")
    
    # 检查4：RoSCo的丢弃率应该不高于Baseline
    check4 = rosco_stats['Avg_Drop_Rate'] <= baseline_stats['Avg_Drop_Rate'] * 1.05
    print(f"\n{'✅' if check4 else '❌'} 检查4: RoSCo丢弃率 ≤ Baseline丢弃率")
    print(f"   RoSCo: {rosco_stats['Avg_Drop_Rate']*100:.2f}% vs Baseline: {baseline_stats['Avg_Drop_Rate']*100:.2f}%")
    if check4:
        checks_passed += 1
    else:
        print("   ⚠️  优先级队列应该降低丢弃率")
    
    # 检查5：延迟增加应该可接受
    check5 = rosco_stats['Avg_Delay'] < baseline_stats['Avg_Delay'] * 1.15
    print(f"\n{'✅' if check5 else '❌'} 检查5: RoSCo延迟增加 < 15%")
    delay_increase = (rosco_stats['Avg_Delay'] - baseline_stats['Avg_Delay']) / baseline_stats['Avg_Delay'] * 100
    print(f"   延迟增加: {delay_increase:+.2f}%")
    if check5:
        checks_passed += 1
    else:
        print("   ⚠️  延迟增加过多")
    
    print(f"\n总体结果: {checks_passed}/{total_checks} 项检查通过")
    if checks_passed == total_checks:
        print("🎉 全部通过！实验结果符合预期！")
    elif checks_passed >= 3:
        print("✅ 大部分通过！结果基本符合预期")
    else:
        print("⚠️  多项检查未通过，请检查代码或重新训练")
    
    print("="*80)


def plot_comparison_charts(rosco_results, baseline_results):
    """绘制对比图表"""
    start_idx = 100
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 安全违规数对比（最重要！）
    ax1 = plt.subplot(2, 3, 1)
    episodes = range(len(rosco_results['Security_Violations']))
    ax1.plot(episodes, baseline_results['Security_Violations'], 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax1.plot(episodes, rosco_results['Security_Violations'], 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax1.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5, label='稳定期开始')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Security Violations', fontsize=12)
    ax1.set_title('⭐ 安全违规数对比 (越低越好)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 负载CV对比
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episodes, baseline_results['Load_CV'], 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax2.plot(episodes, rosco_results['Load_CV'], 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax2.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Load CV', fontsize=12)
    ax2.set_title('负载均衡对比 (越低越好)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 丢弃率对比
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(episodes, np.array(baseline_results['Drop_Rate'])*100, 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax3.plot(episodes, np.array(rosco_results['Drop_Rate'])*100, 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax3.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Drop Rate (%)', fontsize=12)
    ax3.set_title('任务丢弃率对比 (越低越好)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 延迟对比
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(episodes, np.array(baseline_results['Delay'])*100, 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax4.plot(episodes, np.array(rosco_results['Delay'])*100, 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax4.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Delay (ms)', fontsize=12)
    ax4.set_title('平均延迟对比', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 能耗对比
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(episodes, baseline_results['Energy'], 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax5.plot(episodes, rosco_results['Energy'], 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax5.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_ylabel('Energy', fontsize=12)
    ax5.set_title('平均能耗对比', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. QoE对比
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(episodes, baseline_results['QoE'], 
             label='QECO Baseline', color='red', linewidth=2, alpha=0.7)
    ax6.plot(episodes, rosco_results['QoE'], 
             label='RoSCo', color='green', linewidth=2, alpha=0.7)
    ax6.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_ylabel('QoE', fontsize=12)
    ax6.set_title('QoE对比 (越高越好)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 对比图表已保存到 comparison_results.png")
    
    return fig


def plot_bar_comparison(rosco_stats, baseline_stats):
    """绘制柱状对比图（用于论文）"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. 安全违规数
    ax1 = axes[0]
    x = ['QECO\nBaseline', 'RoSCo']
    y = [baseline_stats['Avg_Security_Violations'], rosco_stats['Avg_Security_Violations']]
    bars = ax1.bar(x, y, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Security Violations', fontsize=12)
    ax1.set_title('⭐ 安全违规数\n(越低越好)', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{y[i]:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 负载CV
    ax2 = axes[1]
    y = [baseline_stats['Avg_Load_CV'], rosco_stats['Avg_Load_CV']]
    bars = ax2.bar(x, y, color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Load CV', fontsize=12)
    ax2.set_title('负载均衡\n(越低越好)', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{y[i]:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 丢弃率
    ax3 = axes[2]
    y = [baseline_stats['Avg_Drop_Rate']*100, rosco_stats['Avg_Drop_Rate']*100]
    bars = ax3.bar(x, y, color=['red', 'green'], alpha=0.7)
    ax3.set_ylabel('Drop Rate (%)', fontsize=12)
    ax3.set_title('任务丢弃率\n(越低越好)', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{y[i]:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. 延迟
    ax4 = axes[3]
    y = [baseline_stats['Avg_Delay']*100, rosco_stats['Avg_Delay']*100]
    bars = ax4.bar(x, y, color=['red', 'green'], alpha=0.7)
    ax4.set_ylabel('Delay (ms)', fontsize=12)
    ax4.set_title('平均延迟', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{y[i]:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comparison_bars.png', dpi=300, bbox_inches='tight')
    print("📊 柱状对比图已保存到 comparison_bars.png")
    
    return fig


def save_latex_table(rosco_stats, baseline_stats):
    """生成LaTeX表格代码（用于论文）"""
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison between QECO Baseline and RoSCo}
\\label{tab:comparison}
\\begin{tabular}{lcc}
\\hline
\\textbf{Metric} & \\textbf{QECO Baseline} & \\textbf{RoSCo} \\\\
\\hline
Avg Delay (ms) & {:.2f} & {:.2f} \\\\
Avg Drop Rate (\\%) & {:.2f} & {:.2f} \\\\
Avg Load CV & {:.3f} & {:.3f} \\\\
Avg Security Violations & {:.2f} & {:.2f} \\\\
Avg Energy & {:.3f} & {:.3f} \\\\
\\hline
\\end{tabular}
\\end{table}
""".format(
        baseline_stats['Avg_Delay']*100, rosco_stats['Avg_Delay']*100,
        baseline_stats['Avg_Drop_Rate']*100, rosco_stats['Avg_Drop_Rate']*100,
        baseline_stats['Avg_Load_CV'], rosco_stats['Avg_Load_CV'],
        baseline_stats['Avg_Security_Violations'], rosco_stats['Avg_Security_Violations'],
        baseline_stats['Avg_Energy'], rosco_stats['Avg_Energy']
    )
    
    with open('comparison_table.tex', 'w') as f:
        f.write(latex_code)
    
    print("📝 LaTeX表格代码已保存到 comparison_table.tex")


if __name__ == "__main__":
    print("="*80)
    print("           RoSCo vs QECO Baseline - 结果对比分析")
    print("="*80)
    print()
    
    # 加载结果
    rosco_results, baseline_results = load_results()
    
    if rosco_results is None or baseline_results is None:
        print("\n❌ 无法加载结果文件，程序退出")
        exit(1)
    
    # 计算统计数据
    print("\n📊 正在计算统计数据...")
    rosco_stats = calculate_statistics(rosco_results)
    baseline_stats = calculate_statistics(baseline_results)
    
    # 打印对比表格
    print_comparison_table(rosco_stats, baseline_stats)
    
    # 打印改进分析
    print_improvement_analysis(rosco_stats, baseline_stats)
    
    # 打印验证清单
    print_verification_checklist(rosco_stats, baseline_stats)
    
    # 绘制对比图表
    print("\n📊 正在生成对比图表...")
    plot_comparison_charts(rosco_results, baseline_results)
    plot_bar_comparison(rosco_stats, baseline_stats)
    
    # 生成LaTeX表格
    save_latex_table(rosco_stats, baseline_stats)
    
    print("\n" + "="*80)
    print("✅ 对比分析完成！")
    print("="*80)
    print("\n生成的文件：")
    print("  📊 comparison_results.png - 详细对比图表")
    print("  📊 comparison_bars.png - 柱状对比图（适合论文）")
    print("  📝 comparison_table.tex - LaTeX表格代码（适合论文）")
    print("\n这些文件可以直接用于您的论文！")
    print("="*80)
    
    # 显示图表
    plt.show()
