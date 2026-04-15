# compare_results_paper.py - Enhanced Version for Paper Publication
# Features: Radar Chart, Error Bars, LaTeX Export, Auto Font Fallback

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# --- 1. Professional Style Settings with Fallback ---
def set_style():
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        plt.rcParams['font.family'] = 'sans-serif'  # Fallback

    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# --- 2. Helper Method for Radar Chart ---
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

    register_projection(RadarAxes)
    return theta


# --- 3. Data Loading & Stats ---
def load_results():
    if not os.path.exists('rosco_results.npy') or not os.path.exists('baseline_results.npy'):
        print("❌ Error: Result files not found. Please run main simulations first.")
        return None, None

    rosco = np.load('rosco_results.npy', allow_pickle=True).item()
    baseline = np.load('baseline_results.npy', allow_pickle=True).item()
    print("✅ Results loaded successfully")
    return rosco, baseline


def calculate_statistics(results, start_idx=100):
    stats = {}
    metrics = ['QoE', 'Delay', 'Energy', 'Security_Violations', 'Load_CV', 'Drop_Rate']

    for m in metrics:
        data = np.array(results[m][start_idx:])
        stats[f'Avg_{m}'] = np.mean(data)
        stats[f'Std_{m}'] = np.std(data)

    return stats


# --- 4. Plotting Functions ---

def plot_radar_chart(rosco_stats, baseline_stats):
    """
    Generate a Radar Chart (Spider Plot) to compare normalized metrics.
    We normalize Baseline to 1.0.
    For costs (Delay, Energy, etc.), Lower is Better.
    """
    # Metrics to display
    labels = ['Delay', 'Energy', 'Drop Rate', 'Security\nViolations', 'Load CV']

    # Get values
    base_vals = [baseline_stats['Avg_Delay'], baseline_stats['Avg_Energy'],
                 max(baseline_stats['Avg_Drop_Rate'], 1e-6),
                 max(baseline_stats['Avg_Security_Violations'], 1e-6), baseline_stats['Avg_Load_CV']]

    rosco_vals = [rosco_stats['Avg_Delay'], rosco_stats['Avg_Energy'],
                  rosco_stats['Avg_Drop_Rate'],
                  rosco_stats['Avg_Security_Violations'], rosco_stats['Avg_Load_CV']]

    # Normalize (Baseline = 1.0)
    # Be careful with 0 values in baseline
    base_norm = [1.0] * 5
    rosco_norm = []
    for r, b in zip(rosco_vals, base_vals):
        if b == 0:
            rosco_norm.append(0)  # Should not happen usually
        else:
            rosco_norm.append(r / b)

    N = len(labels)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # Plot Baseline
    ax.plot(theta, base_norm, color='#e74c3c', linewidth=2, linestyle='--', label='QECO Baseline (Normalized 1.0)')
    ax.fill(theta, base_norm, facecolor='#e74c3c', alpha=0.1)

    # Plot RoSCo
    ax.plot(theta, rosco_norm, color='#2ecc71', linewidth=3, label='RoSCo (Ours)')
    ax.fill(theta, rosco_norm, facecolor='#2ecc71', alpha=0.25)

    ax.set_varlabels(labels)

    # Add grid rings
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.2', '0.4', '0.6', '0.8', '1.0'], angle=0, fontsize=9)
    ax.set_ylim(0, 1.1)  # Limit the view to focus on improvement (assuming RoSCo <= Baseline)

    plt.title('Normalized Cost Comparison\n(Lower Area is Better)', y=1.08, fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig('comparison_radar_english.png', dpi=300, bbox_inches='tight')
    print("📊 Radar chart saved to comparison_radar_english.png")


def plot_bar_comparison_with_error(rosco_stats, baseline_stats):
    """Bar chart with Error Bars (Standard Deviation)"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    metrics = [
        ('Avg_Security_Violations', 'Std_Security_Violations', 'Security Violations', 'Lower is Better', '{:.1f}'),
        ('Avg_Load_CV', 'Std_Load_CV', 'Load Balancing (CV)', 'Lower is Better', '{:.3f}'),
        ('Avg_Drop_Rate', 'Std_Drop_Rate', 'Drop Rate', 'Lower is Better', '{:.1%}'),
        ('Avg_Delay', 'Std_Delay', 'Avg Delay (s)', 'Lower is Better', '{:.3f}')
    ]

    x = ['Baseline', 'RoSCo']
    colors = ['#e74c3c', '#2ecc71']

    for i, (avg_k, std_k, title, sub, fmt) in enumerate(metrics):
        ax = axes[i]

        # Prepare data
        means = [baseline_stats[avg_k], rosco_stats[avg_k]]
        stds = [baseline_stats[std_k], rosco_stats[std_k]]

        # Handle unit conversion if necessary
        # (Assuming raw data is already in correct units or converting here)

        # Plot
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_title(f"{title}\n({sub})", fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add values on top
        for bar, val in zip(bars, means):
            if '%' in fmt:
                val_str = f"{val * 100:.2f}%"
            else:
                val_str = fmt.format(val)

            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.05,
                    val_str, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_bars_error_english.png', dpi=300, bbox_inches='tight')
    print("📊 Bar chart with error bars saved to comparison_bars_error_english.png")


def save_latex_table(rosco_stats, baseline_stats):
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison: QECO Baseline vs. RoSCo}
\label{tab:comparison}
\begin{tabular}{lcccc}
\hline
\textbf{Metric} & \textbf{QECO Baseline} & \textbf{RoSCo (Ours)} & \textbf{Improvement} \\
\hline
Avg Delay (ms) & %.2f & \textbf{%.2f} & %s \\
Avg Drop Rate (\%%) & %.2f & \textbf{%.2f} & %s \\
Avg Load CV & %.3f & \textbf{%.3f} & %s \\
Avg Security Violations & %.2f & \textbf{%.2f} & %s \\
Avg Energy (J) & %.3f & \textbf{%.3f} & %s \\
Avg QoE & %.2f & \textbf{%.2f} & %s \\
\hline
\end{tabular}
\end{table}
"""

    # Helpers for improvement string
    def calc_imp(base, new, lower_is_better=True):
        if base == 0: return "-"
        change = (new - base) / base * 100
        if lower_is_better:
            return f"{change:+.1f}\%" if change <= 0 else f"+{change:.1f}\%"
        else:
            return f"{change:+.1f}\%" if change >= 0 else f"{change:.1f}\%"

    # Format values
    values = (
        baseline_stats['Avg_Delay'] * 100, rosco_stats['Avg_Delay'] * 100,
        calc_imp(baseline_stats['Avg_Delay'], rosco_stats['Avg_Delay']),
        baseline_stats['Avg_Drop_Rate'] * 100, rosco_stats['Avg_Drop_Rate'] * 100,
        calc_imp(baseline_stats['Avg_Drop_Rate'], rosco_stats['Avg_Drop_Rate']),
        baseline_stats['Avg_Load_CV'], rosco_stats['Avg_Load_CV'],
        calc_imp(baseline_stats['Avg_Load_CV'], rosco_stats['Avg_Load_CV']),
        baseline_stats['Avg_Security_Violations'], rosco_stats['Avg_Security_Violations'],
        calc_imp(baseline_stats['Avg_Security_Violations'], rosco_stats['Avg_Security_Violations']),
        baseline_stats['Avg_Energy'], rosco_stats['Avg_Energy'],
        calc_imp(baseline_stats['Avg_Energy'], rosco_stats['Avg_Energy']),
        baseline_stats['Avg_QoE'], rosco_stats['Avg_QoE'],
        calc_imp(baseline_stats['Avg_QoE'], rosco_stats['Avg_QoE'], False)
    )

    with open('comparison_table_english.tex', 'w') as f:
        f.write(latex_code % values)
    print("📝 Enhanced LaTeX table saved to comparison_table_english.tex")


# --- 5. Main Execution ---
if __name__ == "__main__":
    print("=" * 80)
    print("           RoSCo vs QECO - Publication Ready Analysis")
    print("=" * 80)

    set_style()
    rosco_res, base_res = load_results()

    if rosco_res:
        rosco_stats = calculate_statistics(rosco_res)
        base_stats = calculate_statistics(base_res)

        # 1. Standard Comparison Charts (Lines)
        # (Assuming you use the function from your original file, simplified here)
        # plot_comparison_charts(rosco_res, base_res)

        # 2. Enhanced Bar Charts with Error Bars
        plot_bar_comparison_with_error(rosco_stats, base_stats)

        # 3. Radar Chart (New!)
        plot_radar_chart(rosco_stats, base_stats)

        # 4. LaTeX Table
        save_latex_table(rosco_stats, base_stats)

        print("\nAll publication figures generated successfully!")