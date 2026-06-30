#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12C_S_alpha_analysis.py

Notebook「test4_12C_S_alpha.ipynb」を、最後の S_alpha スペクトルを表示する
1本の Python スクリプトに整理したものです。

基本の使い方:
    python 12C_S_alpha_analysis.py C_labeled.csv

画像として保存もしたい場合:
    python 12C_S_alpha_analysis.py C_labeled.csv --save s_alpha.png

途中確認用のプロットも表示したい場合:
    python 12C_S_alpha_analysis.py C_labeled.csv --show-intermediate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import Normalize


# ============================================================
# 1. 汎用関数
# ============================================================

def gaussian(x, amp, mean, sigma):
    """ガウシアン関数。ヒストグラムのピークをフィットするために使う。"""
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian_hist(
    variable,
    hist_range_min,
    hist_range_max,
    bin_width,
    fit_range_min,
    fit_range_max,
    initial_guess,
    bound_min,
    bound_max,
):
    """
    1次元データをヒストグラム化し、指定した範囲だけを使ってガウシアンフィットする。

    戻り値:
        popt: フィットパラメータ [Amp, Mean, Sigma]
        pcov: 共分散行列
        bins: ヒストグラムのbin境界
        bin_centers: bin中心
        bin_counts: 各binのcount
    """
    variable = np.asarray(variable)

    bins = np.arange(hist_range_min, hist_range_max + bin_width, bin_width)
    bin_counts, bin_edges = np.histogram(variable, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = (bin_centers >= fit_range_min) & (bin_centers <= fit_range_max)
    x_fit = bin_centers[mask]
    y_fit = bin_counts[mask]

    popt, pcov = curve_fit(
        gaussian,
        x_fit,
        y_fit,
        p0=initial_guess,
        bounds=(bound_min, bound_max),
    )

    return popt, pcov, bins, bin_centers, bin_counts


def print_fit_result(title, popt, pcov):
    """フィット結果を見やすく表示する。"""
    print(f"\n{title}")
    print("-" * len(title))
    param_names = ["Amp", "Mean", "Sigma"]
    for i, param in enumerate(popt):
        error = np.sqrt(pcov[i, i])
        print(f"{param_names[i]}: {param:.3f} ± {error:.3f}")


def plot_1d_hist(variable, range_min, range_max, bin_width, xlabel, title=None):
    """1次元ヒストグラムを表示する。"""
    bins = np.arange(range_min, range_max + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(variable, bins=bins, histtype="step", linewidth=0.8)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_1d_hist_with_fit(
    variable,
    bins,
    popt,
    hist_range_min,
    hist_range_max,
    xlabel,
    title=None,
):
    """1次元ヒストグラムとガウシアンフィット結果を重ねて表示する。"""
    x_plot = np.linspace(hist_range_min, hist_range_max, 1000)
    y_plot = gaussian(x_plot, *popt)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(variable, bins=bins, histtype="step", linewidth=0.8, label="Data")
    ax.plot(x_plot, y_plot, linewidth=1.5, label="Gaussian fit")
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    if title is not None:
        ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_lasx_tot(LASX, TOT):
    """LASX vs LASTOT の2次元ヒストグラムを確認用に表示する。"""
    x_axis_range_min = -1000
    x_axis_range_max = 1000
    y_axis_range_min = 0
    y_axis_range_max = 300
    x_bin_width = 1
    y_bin_width = 1

    bins = [
        np.arange(x_axis_range_min, x_axis_range_max, x_bin_width),
        np.arange(y_axis_range_min, y_axis_range_max, y_bin_width),
    ]
    hist, xedges, yedges = np.histogram2d(LASX, TOT, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=0, vmax=max(1, np.max(hist) // 2))
    mesh = ax.pcolormesh(xedges, yedges, hist.T, shading="auto", norm=norm)

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Counts")

    ax.grid()
    ax.set_xlabel("LASX")
    ax.set_ylabel("LASTOT")
    ax.set_title("LASX vs LASTOT")

    return fig, ax


# ============================================================
# 2. 解析本体
# ============================================================

def analyze(csv_path, show_intermediate=False, save_path=None, no_fit=False):
    """
    CSVを読み込み、ゲートをかけ、最後に S_alpha スペクトルを表示する。

    Parameters
    ----------
    csv_path : str or Path
        入力CSVファイル。Notebookでは C_labeled.csv を読んでいた。
    show_intermediate : bool
        Trueなら途中の確認プロットも表示する。
    save_path : str or Path or None
        指定した場合、最後のS_alphaスペクトルを画像として保存する。
    no_fit : bool
        Trueなら最後のS_alphaフィットを行わず、ヒストグラムだけ表示する。
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    # 必要な列を取り出す。
    # 元Notebookで使っていた列:
    # GRQ2timing, LASX, LASQ1timing, LASTOT, GRX, GRA, GRY, GRB, GRQ2charge
    GRtiming = df["GRQ2timing"]
    LASX = df["LASX"]
    LAStiming = df["LASQ1timing"]
    TOT = df["LASTOT"]

    xdp = df["GRX"]
    adp = df["GRA"]
    ydp = df["GRY"]
    bdp = df["GRB"]
    GRQ2 = df["GRQ2charge"]

    if show_intermediate:
        plot_lasx_tot(LASX, TOT)

    # ------------------------------------------------------------
    # tracking gate
    # ------------------------------------------------------------
    # GRX, GRY が -9999 でないイベントだけを使う。
    tracking_mask = (xdp != -9999) & (ydp != -9999)

    # ------------------------------------------------------------
    # proton gate + LAS TOT gate
    # ------------------------------------------------------------
    # 元Notebookでは GRQ2charge の中心 213.276、幅 21.194*5 を使っている。
    # さらに LASTOT に LASX 依存の補正をかけた量が 140〜170 の範囲に入るイベントを選ぶ。
    corrected_tot = TOT + 0.0430496 * LASX - 1.3842e-5 * (LASX ** 2)

    proton_mask = (GRQ2 > 213.276 - 21.194 * 5) & (GRQ2 < 213.276 + 21.194 * 5)
    las_tot_mask = (corrected_tot > 140) & (corrected_tot < 170)

    gated = df[tracking_mask & proton_mask & las_tot_mask].copy()

    print(f"読み込んだイベント数: {len(df)}")
    print(f"tracking gate後: {tracking_mask.sum()}")
    print(f"proton gate + LAS TOT gate後: {len(gated)}")

    # ------------------------------------------------------------
    # TDiff spectrum
    # ------------------------------------------------------------
    TDiff = gated["LASQ1timing"] - gated["GRQ2timing"]

    if show_intermediate:
        plot_1d_hist(
            TDiff,
            range_min=-500,
            range_max=500,
            bin_width=1,
            xlabel="TDiff = LASQ1timing - GRQ2timing",
            title="TDiff spectrum before true coincidence gate",
        )

    # TDiffをガウシアンフィットする。
    # この結果を使えば true coincidence の中心とsigmaを自動で決められる。
    popt_tdiff, pcov_tdiff, bins_tdiff, _, _ = fit_gaussian_hist(
        variable=TDiff,
        hist_range_min=-500,
        hist_range_max=500,
        bin_width=1,
        fit_range_min=0,
        fit_range_max=200,
        initial_guess=[2000, 90, 5],
        bound_min=[0, 0, 0],
        bound_max=[np.inf, 200, np.inf],
    )
    print_fit_result("TDiff Gaussian fit", popt_tdiff, pcov_tdiff)

    if show_intermediate:
        plot_1d_hist_with_fit(
            TDiff,
            bins=bins_tdiff,
            popt=popt_tdiff,
            hist_range_min=-500,
            hist_range_max=500,
            xlabel="TDiff = LASQ1timing - GRQ2timing",
            title="TDiff Gaussian fit",
        )

    # ------------------------------------------------------------
    # true coincidence gate
    # ------------------------------------------------------------
    # 元Notebookでは手で mean=86.947, sigma=3.377 と入れていた。
    # ここではフィット結果を使う形にしている。
    # 固定値を使いたい場合は、下の2行をコメントアウトして、
    # mean_tdiff = 86.947
    # sigma_tdiff = 3.377
    # に戻せばよい。
    mean_tdiff = popt_tdiff[1]
    sigma_tdiff = abs(popt_tdiff[2])

    true_mask = (TDiff >= mean_tdiff - 4 * sigma_tdiff) & (TDiff <= mean_tdiff + 4 * sigma_tdiff)
    true = gated[true_mask].copy()

    print(f"true coincidence gate後: {len(true)}")
    print(f"TDiff gate: {mean_tdiff:.3f} ± 4 * {sigma_tdiff:.3f}")

    TDiff_true = true["LASQ1timing"] - true["GRQ2timing"]

    if show_intermediate:
        plot_1d_hist(
            TDiff_true,
            range_min=-500,
            range_max=100,
            bin_width=1,
            xlabel="TDiff = LASQ1timing - GRQ2timing",
            title="TDiff spectrum after true coincidence gate",
        )

    # ------------------------------------------------------------
    # Tp, T_alpha の再構成
    # ------------------------------------------------------------
    GRX_true = true["GRX"]
    LASX_true = true["LASX"]

    # 元Notebookの式をそのまま使う。
    # GRX -> proton kinetic energy
    Tp = (((GRX_true * -0.0000454772 + 0.0027089081 + 1) * 832) ** 2 + 938.27 ** 2) ** 0.5 - 938.27

    # LASX -> alpha kinetic energy
    T_alpha = 0.02746 * LASX_true + 63.804

    # ------------------------------------------------------------
    # S_alpha spectrum
    # ------------------------------------------------------------
    # 元Notebookでは S_alpha = 392 - (Tp + T_alpha)
    S_alpha = 392 - (Tp + T_alpha)

    if no_fit:
        fig, ax = plot_1d_hist(
            S_alpha,
            range_min=-50,
            range_max=50,
            bin_width=1,
            xlabel="S_alpha",
            title="S_alpha spectrum",
        )
    else:
        popt_s, pcov_s, bins_s, _, _ = fit_gaussian_hist(
            variable=S_alpha,
            hist_range_min=-50,
            hist_range_max=50,
            bin_width=1,
            fit_range_min=-20,
            fit_range_max=40,
            initial_guess=[2500, 10, 5],
            bound_min=[0, -20, 0],
            bound_max=[np.inf, 40, np.inf],
        )
        print_fit_result("S_alpha Gaussian fit", popt_s, pcov_s)

        fig, ax = plot_1d_hist_with_fit(
            S_alpha,
            bins=bins_s,
            popt=popt_s,
            hist_range_min=-50,
            hist_range_max=50,
            xlabel="S_alpha",
            title="S_alpha spectrum",
        )

    # 保存指定があれば最後の図を保存する。
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n最後の S_alpha スペクトルを保存しました: {save_path}")

    # 最後に全ての図を表示する。
    plt.show()


# ============================================================
# 3. コマンドラインから実行する部分
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="C_labeled.csv から最後の S_alpha スペクトルを表示する解析スクリプト"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="C_labeled.csv",
        help="入力CSVファイル。省略時は C_labeled.csv",
    )
    parser.add_argument(
        "--show-intermediate",
        action="store_true",
        help="途中のLASX-TOT、TDiffスペクトルも表示する",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="最後の S_alpha スペクトルを画像として保存するパス。例: s_alpha.png",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="S_alpha のガウシアンフィットをせず、ヒストグラムだけ表示する",
    )

    args = parser.parse_args()

    analyze(
        csv_path=args.csv_path,
        show_intermediate=args.show_intermediate,
        save_path=args.save,
        no_fit=args.no_fit,
    )


if __name__ == "__main__":
    main()
