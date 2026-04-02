import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    parser = argparse.ArgumentParser(description='Plot QKT squared difference from GPTAQ quantization')
    parser.add_argument('--csv', type=str, default='QK.csv', help='Path to QK.csv file')
    parser.add_argument('--log', action='store_true', help='Use log scale for z-axis')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(df['layer'], df['head'], df['sq_diff'],
                    c=df['sq_diff'], cmap='viridis', alpha=0.8, s=20)
    fig.colorbar(sc, ax=ax, label='Squared Difference')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_zlabel('SQ Diff (||QK^T - Q\'K\'^T||_F^2)')

    if args.log:
        ax.set_yscale('log')
        ax.set_zscale('log')

    ax.set_title('Per-Head QKT Quantization Error')

    num_layers = df['layer'].nunique()
    num_heads = df['head'].nunique()
    ax.set_xticks(range(0, num_layers + 1, max(1, num_layers // 10)))
    ax.set_yticks(range(0, num_heads + 1, max(1, num_heads // 4)))

    plt.tight_layout()
    plt.savefig('qkt_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved to qkt_plot.png')


if __name__ == '__main__':
    main()
