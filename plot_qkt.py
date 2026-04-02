import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import matplotlib.ticker as ticker


def main():
    parser = argparse.ArgumentParser(description='Plot QKT squared difference from GPTAQ quantization')
    parser.add_argument('--csv', type=str, default='QK1.csv', help='Path to QK.csv file')
    parser.add_argument('--log', action='store_true', default=False, help='Use log scale for z-axis')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # 过滤掉零值或负值
    df = df[df['sq_diff'] > 0]

    if df.empty:
        print("Error: No positive values found in sq_diff column")
        return

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

    unique_layers = np.unique(layers)
    unique_heads = np.unique(heads)

    layer_grid, head_grid = np.meshgrid(unique_layers, unique_heads)

    points = np.column_stack((layers, heads))
    values = sq_diffs

    sq_diff_grid = griddata(points, values, (layer_grid, head_grid), method='linear', fill_value=np.nan)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(layer_grid, head_grid, sq_diff_grid.T,
                          cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)

    fig.colorbar(surf, ax=ax, label='Squared Difference')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_zlabel('SQ Diff (||QK^T - Q\'K\'^T||_F^2)')

    if args.log:
        ax.set_zscale('log')
        # 手动设置 z 轴刻度：在对数空间均匀分布
        z_min = np.nanmin(sq_diff_grid)
        z_max = np.nanmax(sq_diff_grid)
        # 生成 6~8 个对数等间距刻度
        z_ticks = np.logspace(np.log10(z_min), np.log10(z_max), num=8)
        ax.set_zticks(z_ticks)
        # 用科学计数法格式化刻度标签，避免重叠
        ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        # 减小 z 轴标签字体
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
    plt.savefig('qkt_plot_surface1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved to qkt_plot_surface2.png')


if __name__ == '__main__':
    main()
