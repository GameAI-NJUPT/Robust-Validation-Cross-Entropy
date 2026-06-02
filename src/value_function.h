// Linear value function over the nine signed CBMPI features. The value is a
// single dot product theta^T phi.

#pragma once

#include <string>
#include <vector>

namespace tcore {

struct State;

class ValueFunction {
public:
    // Returns true on success.
    bool load(const std::string& path);
    void set_theta(const std::vector<double>& theta);
    void set_theta(std::vector<double>&& theta);
    const std::vector<double>& theta() const { return theta_; }

    int  dim() const { return static_cast<int>(theta_.size()); }

    // value(s, action) returns 0 for terminal states, otherwise computes the
    // signed features and returns theta^T phi.
    double value(const State& s, int action) const;

    // value_from_features uses an already-computed 9-d signed feature vector.
    double value_from_features(const double* base) const;

private:
    std::vector<double> theta_;
};

} // namespace tcore
