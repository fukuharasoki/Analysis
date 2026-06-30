// 12C_S_alpha_simple.C
//
// C_labeled.csv から最後の S_alpha スペクトルだけを作る簡略ROOTマクロです。
// 元Notebook / Python版と同じく、途中確認プロットは省略しています。
//
// 使い方:
//   root -l 12C_S_alpha_simple.C
//
// または、保存ファイル名も指定する場合:
//   root -l '12C_S_alpha_simple.C("C_labeled.csv", "s_alpha.png")'

#include <TCanvas.h>
#include <TH1D.h>
#include <TStyle.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// =========================
// 解析で使う固定値
// =========================

// GRQ2charge の proton gate
const double GRQ2_CENTER = 213.276;
const double GRQ2_SIGMA  = 21.194;
const double GRQ2_NSIGMA = 5.0;

// LAS の粒子識別用 gate
// corrected_tot = LASTOT + a*LASX + b*LASX^2
const double LAS_TOT_A   = 0.0430496;
const double LAS_TOT_B   = -1.3842e-5;
const double LAS_TOT_MIN = 140.0;
const double LAS_TOT_MAX = 170.0;

// true coincidence gate
const double TDIFF_MEAN   = 86.947;
const double TDIFF_SIGMA  = 3.377;
const double TDIFF_NSIGMA = 4.0;

// 運動エネルギー再構成用の係数
const double BEAM_KINEMATIC_ENERGY = 392.0;  // MeV
const double PROTON_MASS           = 938.27; // MeV/c^2
const double GR_MOMENTUM_REF       = 832.0;  // MeV/c

// GRX -> Tp
const double GRX_COEF   = -0.0000454772;
const double GRX_OFFSET = 0.0027089081;

// LASX -> T_alpha
const double LASX_COEF   = 0.02746;
const double LASX_OFFSET = 63.804;

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\"";
    const auto begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    // 今回の数値CSV用の簡易パーサです。
    // ダブルクォート内にカンマがある複雑なCSVは想定していません。
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(line);
    while (std::getline(ss, item, ',')) out.push_back(trim(item));
    return out;
}

bool has_column(const std::map<std::string, int>& col, const std::string& name) {
    return col.find(name) != col.end();
}

} // namespace

void S_alpha_simple(const char* csv_file = "C_labeled.csv", const char* save_file = "") {
    std::ifstream fin(csv_file);
    if (!fin) {
        std::cerr << "CSV file was not found: " << csv_file << std::endl;
        return;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        std::cerr << "CSV file is empty: " << csv_file << std::endl;
        return;
    }

    // header から列番号を作る
    const auto header = split_csv_line(line);
    std::map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        col[header[i]] = i;
    }

    const std::vector<std::string> required = {
        "GRX", "GRY", "GRQ2charge", "LASTOT", "LASX", "LASQ1timing", "GRQ2timing"
    };
    for (const auto& name : required) {
        if (!has_column(col, name)) {
            std::cerr << "Required column was not found: " << name << std::endl;
            return;
        }
    }

    auto get_value = [&](const std::vector<std::string>& row, const std::string& name) -> double {
        const int idx = col[name];
        if (idx < 0 || idx >= static_cast<int>(row.size()) || row[idx].empty()) return NAN;
        return std::stod(row[idx]);
    };

    TH1D* h = new TH1D("h_S_alpha", "S_{#alpha} spectrum;S_{#alpha} [MeV];Counts", 100, -50.0, 50.0);

    long long n_all = 0;
    long long n_selected = 0;

    const double grq2_min = GRQ2_CENTER - GRQ2_NSIGMA * GRQ2_SIGMA;
    const double grq2_max = GRQ2_CENTER + GRQ2_NSIGMA * GRQ2_SIGMA;
    const double tdiff_min = TDIFF_MEAN - TDIFF_NSIGMA * TDIFF_SIGMA;
    const double tdiff_max = TDIFF_MEAN + TDIFF_NSIGMA * TDIFF_SIGMA;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        ++n_all;

        const auto row = split_csv_line(line);
        if (row.size() < header.size()) continue;

        double GRX         = get_value(row, "GRX");
        double GRY         = get_value(row, "GRY");
        double GRQ2charge  = get_value(row, "GRQ2charge");
        double LASTOT      = get_value(row, "LASTOT");
        double LASX        = get_value(row, "LASX");
        double LASQ1timing = get_value(row, "LASQ1timing");
        double GRQ2timing  = get_value(row, "GRQ2timing");

        if (!std::isfinite(GRX) || !std::isfinite(GRY) || !std::isfinite(GRQ2charge) ||
            !std::isfinite(LASTOT) || !std::isfinite(LASX) ||
            !std::isfinite(LASQ1timing) || !std::isfinite(GRQ2timing)) {
            continue;
        }

        // 1. tracking gate
        const bool tracking_gate = (GRX != -9999.0) && (GRY != -9999.0);

        // 2. proton gate
        const bool proton_gate = (GRQ2charge > grq2_min) && (GRQ2charge < grq2_max);

        // 3. LAS TOT gate
        const double corrected_tot = LASTOT + LAS_TOT_A * LASX + LAS_TOT_B * LASX * LASX;
        const bool las_tot_gate = (corrected_tot > LAS_TOT_MIN) && (corrected_tot < LAS_TOT_MAX);

        // 4. true coincidence gate
        const double tdiff = LASQ1timing - GRQ2timing;
        const bool true_coin_gate = (tdiff >= tdiff_min) && (tdiff <= tdiff_max);

        if (!(tracking_gate && proton_gate && las_tot_gate && true_coin_gate)) continue;
        ++n_selected;

        // 5. Tp の再構成
        const double proton_momentum = (GRX * GRX_COEF + GRX_OFFSET + 1.0) * GR_MOMENTUM_REF;
        const double Tp = std::sqrt(proton_momentum * proton_momentum + PROTON_MASS * PROTON_MASS) - PROTON_MASS;

        // 6. T_alpha の再構成
        const double T_alpha = LASX_COEF * LASX + LASX_OFFSET;

        // 7. S_alpha の計算
        const double S_alpha = BEAM_KINEMATIC_ENERGY - (Tp + T_alpha);
        h->Fill(S_alpha);
    }

    std::cout << "all events             : " << n_all << std::endl;
    std::cout << "selected events for Sa : " << n_selected << std::endl;

    gStyle->SetOptStat(1110);

    TCanvas* c = new TCanvas("c_S_alpha", "S_alpha spectrum", 800, 600);
    h->SetLineWidth(2);
    h->Draw("hist");
    c->Update();

    if (std::string(save_file).size() > 0) {
        c->SaveAs(save_file);
        std::cout << "saved: " << save_file << std::endl;
    }
}

// root -l 12C_S_alpha_simple.C でそのまま実行したい場合用
void _12C_S_alpha_simple() {
    S_alpha_simple();
}
