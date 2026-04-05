import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from matplotlib.colors import LightSource


def load_and_process_csv(csv_path):
    """加载并处理CSV文件，返回网格数据"""
    df = pd.read_csv(csv_path)

    # 过滤掉零值或负值
    df = df[df['sq_diff'] > 0]

    if df.empty:
        print(f"Error: No positive values found in {csv_path}")
        return None, None, None, None

    # 排除异常值：超过同层中位数 10 倍的点视为异常
    layer_medians = df.groupby('layer')['sq_diff'].transform('median')
    df_clean = df[df['sq_diff'] <= layer_medians * 10]
    outlier_text = ''
    if len(df_clean) < len(df):
        outliers = df[df['sq_diff'] > layer_medians * 10]
        lines = [f"Outliers removed ({len(df) - len(df_clean)}):"]
        for _, row in outliers.iterrows():
            lines.append(f"  L{int(row['layer'])}H{int(row['head'])} = {row['sq_diff']:.2e}")
        outlier_text = '\n'.join(lines)
        df = df_clean

    layers = df['layer'].values
    heads = df['head'].values
    sq_diffs = df['sq_diff'].values

    return layers, heads, sq_diffs, outlier_text


def main():
    parser = argparse.ArgumentParser(description='Plot QKT squared difference from GPTAQ quantization')
    parser.add_argument('--csv', type=str, default='QK1.csv', help='Path to QK.csv file (single mode)')
    parser.add_argument('--csv1', type=str, default=None, help='Path to first QK.csv file (baseline, e.g., without QK_quant)')
    parser.add_argument('--csv2', type=str, default=None, help='Path to second QK.csv file (comparison, e.g., with QK_quant)')
    parser.add_argument('--label1', type=str, default='Baseline', help='Label for first surface')
    parser.add_argument('--label2', type=str, default='QK Quant', help='Label for second surface')
    parser.add_argument('--log', action='store_true', default=False, help='Use log scale for z-axis')
    parser.add_argument('--output', type=str, default='qkt_plot_comparison.png', help='Output filename')
    args = parser.parse_args()

    # 双CSV对比模式
    if args.csv1 and args.csv2:
        layers1, heads1, sq_diffs1, outlier_text1 = load_and_process_csv(args.csv1)
        layers2, heads2, sq_diffs2, outlier_text2 = load_and_process_csv(args.csv2)

        if layers1 is None or layers2 is None:
            return

        # 合并获取统一的网格
        all_layers = np.concatenate([layers1, layers2])
        all_heads = np.concatenate([heads1, heads2])
        unique_layers = np.unique(all_layers)
        unique_heads = np.unique(all_heads)

        layer_grid, head_grid = np.meshgrid(unique_layers, unique_heads)

        # 插值第一个CSV
        points1 = np.column_stack((layers1, heads1))
        sq_diff_grid1 = griddata(points1, sq_diffs1, (layer_grid, head_grid), method='linear', fill_value=np.nan)

        # 插值第二个CSV
        points2 = np.column_stack((layers2, heads2))
        sq_diff_grid2 = griddata(points2, sq_diffs2, (layer_grid, head_grid), method='linear', fill_value=np.nan)

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 第一个曲面：蓝色系，较低位置
        surf1 = ax.plot_surface(layer_grid, head_grid, sq_diff_grid1,
                               cmap='Blues', alpha=0.7, linewidth=0, antialiased=True,
                               label=args.label1)

        # 第二个曲面：红色系，较高位置
        surf2 = ax.plot_surface(layer_grid, head_grid, sq_diff_grid2,
                               cmap='Reds', alpha=0.7, linewidth=0, antialiased=True,
                               label=args.label2)

        # 创建假的proxy artists用于图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', edgecolor='blue', alpha=0.7, label=args.label1),
                          Patch(facecolor='lightcoral', edgecolor='red', alpha=0.7, label=args.label2)]
        ax.legend(handles=legend_elements, loc='upper left')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Head')
        ax.set_zlabel('SQ Diff (||QK^T - Q\'K\'^T||_F^2)')

        # 计算z轴范围用于对数刻度
        z_min = min(np.nanmin(sq_diff_grid1), np.nanmin(sq_diff_grid2))
        z_max = max(np.nanmax(sq_diff_grid1), np.nanmax(sq_diff_grid2))

        if args.log:
            ax.set_zscale('log')
            z_ticks = np.logspace(np.log10(z_min), np.log10(z_max), num=8)
            ax.set_zticks(z_ticks)
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            ax.tick_params(axis='z', labelsize=8)

        ax.set_title(f'QKT Quantization Error Comparison\n{args.label1} vs {args.label2}')

        num_layers = len(unique_layers)
        num_heads = len(unique_heads)

        x_ticks = np.linspace(min(unique_layers), max(unique_layers),
                             min(10, num_layers), dtype=int)
        y_ticks = np.linspace(min(unique_heads), max(unique_heads),
                             min(8, num_heads), dtype=int)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)

        # 显示异常值信息
        outlier_text_combined = ''
        if outlier_text1:
            outlier_text_combined += f"[{args.label1}]\n{outlier_text1}\n"
        if outlier_text2:
            outlier_text_combined += f"\n[{args.label2}]\n{outlier_text2}"

        if outlier_text_combined:
            fig.text(0.02, 0.02, outlier_text_combined.strip(), fontsize=7, fontfamily='monospace',
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Comparison plot saved to {args.output}')

    else:
        # 单CSV模式（原有功能）
        csv_path = args.csv
        layers, heads, sq_diffs, outlier_text = load_and_process_csv(csv_path)

        if layers is None:
            return

        unique_layers = np.unique(layers)
        unique_heads = np.unique(heads)

        layer_grid, head_grid = np.meshgrid(unique_layers, unique_heads)

        points = np.column_stack((layers, heads))
        sq_diff_grid = griddata(points, sq_diffs, (layer_grid, head_grid), method='linear', fill_value=np.nan)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(layer_grid, head_grid, sq_diff_grid,
                              cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)

        fig.colorbar(surf, ax=ax, label='Squared Difference')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Head')
        ax.set_zlabel('SQ Diff (||QK^T - Q\'K\'^T||_F^2)')

        if args.log:
            ax.set_zscale('log')
            z_min = np.nanmin(sq_diff_grid)
            z_max = np.nanmax(sq_diff_grid)
            z_ticks = np.logspace(np.log10(z_min), np.log10(z_max), num=8)
            ax.set_zticks(z_ticks)
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            ax.tick_params(axis='z', labelsize=8)

        ax.set_title('Per-Head QKT Quantization Error Surface')

        num_layers = len(unique_layers)
        num_heads = len(unique_heads)

        x_ticks = np.linspace(min(unique_layers), max(unique_layers),
                             min(10, num_layers), dtype=int)
        y_ticks = np.linspace(min(unique_heads), max(unique_heads),
                             min(8, num_heads), dtype=int)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)

        if outlier_text:
            fig.text(0.02, 0.02, outlier_text, fontsize=8, fontfamily='monospace',
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
