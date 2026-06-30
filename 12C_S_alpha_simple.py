#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12C_S_alpha_simple.py

C_labeled.csv から最後の S_alpha スペクトルだけを作るための簡略版です。
元Notebookの途中確認プロットやフィット処理は省き、S_alpha計算に必要な部分だけ残しています。

使い方:
    python 12C_S_alpha_simple.py C_labeled.csv

画像保存もしたい場合:
    python 12C_S_alpha_simple.py C_labeled.csv --save s_alpha.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 解析で使う固定値
# =========================

# GRQ2charge の proton gate
GRQ2_CENTER = 213.276
GRQ2_SIGMA = 21.194
GRQ2_NSIGMA = 5

# LAS の粒子識別用 gate
# corrected_tot = LASTOT + a*LASX + b*LASX^2
LAS_TOT_A = 0.0430496
LAS_TOT_B = -1.3842e-5
LAS_TOT_MIN = 140
LAS_TOT_MAX = 170

# true coincidence gate
# 元Notebookで最終的に使っていた値をそのまま使用
TDIFF_MEAN = 86.947
TDIFF_SIGMA = 3.377
TDIFF_NSIGMA = 4

# 運動エネルギー再構成用の係数
BEAM_KINEMATIC_ENERGY = 392.0  # MeV
PROTON_MASS = 938.27           # MeV/c^2
GR_MOMENTUM_REF = 832.0        # MeV/c

# GRX -> Tp
GRX_COEF = -0.0000454772
GRX_OFFSET = 0.0027089081

# LASX -> T_alpha
LASX_COEF = 0.02746
LASX_OFFSET = 63.804


def calculate_s_alpha(df: pd.DataFrame) -> pd.Series:
    """元Notebookと同じ条件で event を選び、S_alpha を計算する。"""

    # 1. tracking gate
    # GRX, GRY が -9999 の event は tracking できていないものとして除く。
    tracking_gate = (df["GRX"] != -9999) & (df["GRY"] != -9999)

    # 2. proton gate
    # GRQ2charge が proton らしい範囲に入る event だけ残す。
    grq2_min = GRQ2_CENTER - GRQ2_NSIGMA * GRQ2_SIGMA
    grq2_max = GRQ2_CENTER + GRQ2_NSIGMA * GRQ2_SIGMA
    proton_gate = (df["GRQ2charge"] > grq2_min) & (df["GRQ2charge"] < grq2_max)

    # 3. LAS の TOT gate
    # LASX 依存性を補正した LASTOT が 140〜170 の event だけ残す。
    corrected_tot = df["LASTOT"] + LAS_TOT_A * df["LASX"] + LAS_TOT_B * df["LASX"] ** 2
    las_tot_gate = (corrected_tot > LAS_TOT_MIN) & (corrected_tot < LAS_TOT_MAX)

    # 4. true coincidence gate
    # TDiff = LASQ1timing - GRQ2timing が、元Notebookで決めたピーク周りに入る event だけ残す。
    tdiff = df["LASQ1timing"] - df["GRQ2timing"]
    tdiff_min = TDIFF_MEAN - TDIFF_NSIGMA * TDIFF_SIGMA
    tdiff_max = TDIFF_MEAN + TDIFF_NSIGMA * TDIFF_SIGMA
    true_coin_gate = (tdiff >= tdiff_min) & (tdiff <= tdiff_max)

    # すべての gate を同時に満たす event だけ使う。
    gate = tracking_gate & proton_gate & las_tot_gate & true_coin_gate
    selected = df[gate].copy()

    print(f"all events              : {len(df)}")
    print(f"selected events for Sα  : {len(selected)}")

    # 5. Tp の再構成
    # 元Notebookの式：
    # Tp = sqrt( [ (GRX*a + offset + 1) * p_ref ]^2 + mp^2 ) - mp
    proton_momentum = (selected["GRX"] * GRX_COEF + GRX_OFFSET + 1.0) * GR_MOMENTUM_REF
    tp = np.sqrt(proton_momentum ** 2 + PROTON_MASS ** 2) - PROTON_MASS

    # 6. T_alpha の再構成
    # 元Notebookの式：T_alpha = 0.02746 * LASX + 63.804
    t_alpha = LASX_COEF * selected["LASX"] + LASX_OFFSET

    # 7. S_alpha の計算
    # 元Notebookの式：S_alpha = 392 - (Tp + T_alpha)
    s_alpha = BEAM_KINEMATIC_ENERGY - (tp + t_alpha)

    return s_alpha


def plot_s_alpha(s_alpha, save_path=None):
    """S_alpha スペクトルを表示する。"""
    bins = np.arange(-50, 50 + 1, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(s_alpha, bins=bins, histtype="step", linewidth=0.8)
    ax.set_xlabel(r"$S_\alpha$ [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title(r"$S_\alpha$ spectrum")
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Make S_alpha spectrum from C_labeled.csv")
    parser.add_argument("csv", nargs="?", default="C_labeled.csv", help="input CSV file")
    parser.add_argument("--save", default=None, help="output image path, e.g. s_alpha.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    df = pd.read_csv(csv_path)
    s_alpha = calculate_s_alpha(df)
    plot_s_alpha(s_alpha, save_path=args.save)


if __name__ == "__main__":
    main()
