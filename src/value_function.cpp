#include "value_function.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "features.h"
#include "tetris_state.h"

namespace tcore {

bool ValueFunction::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[vf] cannot open %s\n", path.c_str());
        return false;
    }
    theta_.clear();
    std::string line;
    while (std::getline(f, line)) {
        // trim
        auto a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        auto b = line.find_last_not_of(" \t\r\n");
        std::string s = line.substr(a, b - a + 1);
        try { theta_.push_back(std::stod(s)); }
        catch (...) { /* skip non-numeric lines */ }
    }
    std::cerr << "[vf] loaded " << theta_.size() << " weights from " << path << "\n";
    return !theta_.empty();
}

void ValueFunction::set_theta(const std::vector<double>& theta) {
    theta_ = theta;
}

void ValueFunction::set_theta(std::vector<double>&& theta) {
    theta_ = std::move(theta);
}

double ValueFunction::value(const State& s, int action) const {
    if (action_list(s) == 0) return 0.0;
    double base[kFeatureDim];
    compute_features(s, action, base);
    return value_from_features(base);
}

double ValueFunction::value_from_features(const double* base) const {
    double x = 0;
    for (int i = 0; i < kFeatureDim; ++i) x += theta_[i] * base[i];
    return x;
}

} // namespace tcore
