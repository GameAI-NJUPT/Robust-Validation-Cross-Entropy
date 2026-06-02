#include "ce_final/ce_final.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "features.h"
#include "pieces.h"
#include "rng.h"
#include "tetris_state.h"
#include "value_function.h"

namespace fs = std::filesystem;

namespace tcore::ce_final {
namespace {

constexpr const char* kDefaultConfigPath = "src/ce_final/final-ce.properties";
constexpr const char* kCheckpointVersion = "final-ce-cpp-checkpoint-v3";
constexpr const char* kCheckpointStateFile = "checkpoint_state.txt";
constexpr const char* kCheckpointManifestFile = "checkpoint_manifest.properties";
constexpr const char* kCheckpointFingerprintFile = "checkpoint_fingerprint.txt";
constexpr int kParameterNum = kFeatureDim;
constexpr int kMaxSteps = 100000;

std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::string lower_no_underscore(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != '_' && c != '-' && c != ' ') {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
    }
    return out;
}

std::vector<std::string> split_list(const std::string& value) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : value) {
        if (c == ',' || c == ';') {
            cur = trim(cur);
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    cur = trim(cur);
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::string timestamp(const char* fmt) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    std::ostringstream os;
    os << std::put_time(&tm, fmt);
    return os.str();
}

std::string fmt6(double value) {
    if (!std::isfinite(value)) return "";
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << value;
    return os.str();
}

std::string csv_escape(const std::string& value) {
    bool quote = false;
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '"' || c == ',' || c == '\n' || c == '\r') quote = true;
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    return quote ? "\"" + escaped + "\"" : escaped;
}

uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string join_doubles(const std::vector<double>& xs) {
    std::ostringstream os;
    os << std::setprecision(17);
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ',';
        os << xs[i];
    }
    return os.str();
}

std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> out;
    if (trim(s).empty()) return out;
    for (const std::string& part : split_csv_simple(s)) {
        out.push_back(std::stod(trim(part)));
    }
    return out;
}

void atomic_replace(const fs::path& tmp, const fs::path& target) {
    std::error_code ec;
    fs::remove(target, ec);
    ec.clear();
    fs::rename(tmp, target, ec);
    if (ec) throw std::runtime_error("could not replace " + target.string() + ": " + ec.message());
}

void copy_latest(const fs::path& source, const fs::path& target) {
    std::error_code ec;
    fs::copy_file(source, target, fs::copy_options::overwrite_existing, ec);
    if (ec) throw std::runtime_error("could not copy " + source.string() + " to " + target.string() + ": " + ec.message());
}

class Log {
public:
    void open(const fs::path& path) {
        file_.open(path, std::ios::app);
        if (!file_) throw std::runtime_error("cannot open log file: " + path.string());
    }

    void info(const std::string& msg) { write("INFO", msg); }
    void warn(const std::string& msg) { write("WARN", msg); }

private:
    void write(const char* level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mu_);
        std::string line = timestamp("%Y-%m-%d %H:%M:%S") + " " + level + " - " + msg;
        std::cerr << line << "\n";
        if (file_) file_ << line << "\n";
    }

    std::mutex mu_;
    std::ofstream file_;
};

struct Config {
    long long base_seed = 42;
    long long run_stride = 1000003;
    int runs = 5;
    int run_start = 1;
    int run_end = 5;
    int population = 100;
    double elite_ratio = 0.1;
    int iterations = 50;
    bool checkpoint_enabled = true;
    int checkpoint_every_iterations = 1;
    bool resume_enabled = false;
    std::string resume_batch_dir;
    std::string benchmark_methods = "OriginalCE,PoolCE,PoolDiversityCE,CEMRL,ProportionalCE";
    std::string robust_ablation_methods = "RobustValidationRerankDiversityCE,RobustValidationStdDiversityCE,RobustValidationMemoryDiversityCE";
    std::string benchmark_noise_levels = "noise_off,noise_on";
    std::string ratio_methods = "PoolCE,PoolDiversityCE";
    std::string ratio_selection_modes = "train_elite,val_elite";
    int train_episodes = 100;
    int validation_episodes = 100;
    int validation_top_k = 10;
    int global_top_n = 10;
    int test_episodes = 10000;
    int thread_reserve = 2;
    std::string results_root = "runs/ce_final";
    std::string episode_protocol = "seeded_stream_iid";
    double initial_noise = 5.0;
    double proportional_max_weight_share = 0.35;
    double pool_size_ratio = 0.1;
    double pool_elite_weight_share = 0.7;
    double pool_history_weight_share = 0.3;
    double diversity_lambda = 0.3;
    int replay_max_size = 50000;
    int td_updates_per_actor = 200;
    int row_num = 10;
    std::vector<double> ratios = {0.6, 0.7, 0.8};

    int robust_candidate_train_top_k = 30;
    int robust_candidate_history_top_k = 10;
    int robust_candidate_previous_val_top_k = 10;
    int robust_candidate_random_k = 10;
    int robust_validation_episodes = 300;
    double robust_validation_std_penalty = 0.35;
    double robust_train_score_weight = 0.10;
    double robust_diversity_weight = 0.15;
    int robust_deep_every = 10;
    int robust_deep_episodes = 1000;
    int behavior_probe_states = 512;
    int robust_pool_size = 30;
    double robust_pool_current_share_early = 0.50;
    double robust_pool_history_share_early = 0.50;
    double robust_pool_current_share_late = 0.65;
    double robust_pool_history_share_late = 0.35;
    int robust_pool_late_after_iteration = 30;
    int robust_final_top_n = 20;
    bool early_stop_enabled = true;
    int early_stop_min_iterations = 50;
    int early_stop_patience_deep_checks = 3;
    double early_stop_min_delta = 25.0;
    double early_stop_min_avg_std = 0.05;

    int elite_num() const {
        return std::max(1, static_cast<int>(std::llround(population * elite_ratio)));
    }
    int pool_size() const {
        return std::max(1, static_cast<int>(std::llround(population * pool_size_ratio)));
    }
    int robust_history_pool_size() const {
        return std::max(1, robust_pool_size);
    }
    int run_count() const { return run_end - run_start + 1; }
};

long long to_long(const std::map<std::string, std::string>& props, const std::string& key, long long fallback) {
    auto it = props.find(key);
    if (it == props.end()) return fallback;
    try { return std::stoll(trim(it->second)); } catch (...) { return fallback; }
}

int to_int(const std::map<std::string, std::string>& props, const std::string& key, int fallback) {
    auto it = props.find(key);
    if (it == props.end()) return fallback;
    try { return std::stoi(trim(it->second)); } catch (...) { return fallback; }
}

double to_double(const std::map<std::string, std::string>& props, const std::string& key, double fallback) {
    auto it = props.find(key);
    if (it == props.end()) return fallback;
    try { return std::stod(trim(it->second)); } catch (...) { return fallback; }
}

bool to_bool(const std::map<std::string, std::string>& props, const std::string& key, bool fallback) {
    auto it = props.find(key);
    if (it == props.end()) return fallback;
    std::string v = lower_no_underscore(trim(it->second));
    if (v == "true" || v == "yes" || v == "1" || v == "on") return true;
    if (v == "false" || v == "no" || v == "0" || v == "off") return false;
    return fallback;
}

std::string to_string_prop(const std::map<std::string, std::string>& props,
                           const std::string& key, const std::string& fallback) {
    auto it = props.find(key);
    return it == props.end() ? fallback : trim(it->second);
}

void read_properties(const std::string& path, std::map<std::string, std::string>& props) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        props[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
}

Config load_config(const std::string& path, const std::map<std::string, std::string>& overrides) {
    Config c;
    std::map<std::string, std::string> props;
    read_properties(path, props);
    for (const auto& kv : overrides) props[kv.first] = kv.second;

    c.base_seed = to_long(props, "base.seed", c.base_seed);
    c.run_stride = to_long(props, "run.stride", c.run_stride);
    c.runs = std::max(1, to_int(props, "runs", c.runs));
    c.run_start = std::clamp(to_int(props, "run.start", c.run_start), 1, c.runs);
    c.run_end = to_int(props, "run.end", c.runs);
    c.run_end = std::max(c.run_start, std::min(c.run_end, c.runs));
    c.population = std::max(1, to_int(props, "population", c.population));
    c.elite_ratio = std::clamp(to_double(props, "elite.ratio", c.elite_ratio), 0.0, 1.0);
    c.iterations = std::max(1, to_int(props, "iterations", c.iterations));
    c.checkpoint_enabled = to_bool(props, "checkpoint.enabled", c.checkpoint_enabled);
    c.checkpoint_every_iterations = std::max(1, to_int(props, "checkpoint.every.iterations", c.checkpoint_every_iterations));
    c.resume_enabled = to_bool(props, "resume.enabled", c.resume_enabled);
    c.resume_batch_dir = to_string_prop(props, "resume.batch.dir", c.resume_batch_dir);
    c.benchmark_methods = to_string_prop(props, "benchmark.methods", c.benchmark_methods);
    c.robust_ablation_methods = to_string_prop(props, "robust.ablation.methods", c.robust_ablation_methods);
    c.benchmark_noise_levels = to_string_prop(props, "benchmark.noise.levels", c.benchmark_noise_levels);
    c.ratio_methods = to_string_prop(props, "ratio.methods", c.ratio_methods);
    c.ratio_selection_modes = to_string_prop(props, "ratio.selection.modes", c.ratio_selection_modes);
    c.train_episodes = std::max(1, to_int(props, "train.episodes", c.train_episodes));
    c.validation_episodes = std::max(1, to_int(props, "validation.episodes", c.validation_episodes));
    c.validation_top_k = std::max(1, to_int(props, "validation.top.k", c.validation_top_k));
    c.global_top_n = std::max(1, to_int(props, "global.top.n", c.global_top_n));
    c.test_episodes = std::max(1, to_int(props, "test.episodes", c.test_episodes));
    c.thread_reserve = std::max(0, to_int(props, "thread.reserve", c.thread_reserve));
    c.results_root = to_string_prop(props, "results.root", c.results_root);
    c.episode_protocol = to_string_prop(props, "episode.protocol", c.episode_protocol);
    c.initial_noise = std::max(0.0, to_double(props, "initial.noise", c.initial_noise));
    c.proportional_max_weight_share = std::clamp(to_double(props, "proportional.max.weight.share", c.proportional_max_weight_share), 0.0, 1.0);
    c.pool_size_ratio = std::clamp(to_double(props, "pool.size.ratio", c.pool_size_ratio), 0.0, 1.0);
    c.pool_elite_weight_share = std::clamp(to_double(props, "pool.elite.weight.share", c.pool_elite_weight_share), 0.0, 1.0);
    c.pool_history_weight_share = std::clamp(to_double(props, "pool.history.weight.share", c.pool_history_weight_share), 0.0, 1.0);
    c.diversity_lambda = std::max(0.0, to_double(props, "diversity.lambda", c.diversity_lambda));
    c.replay_max_size = std::max(1, to_int(props, "cemrl.replay.max.size", c.replay_max_size));
    c.td_updates_per_actor = std::max(0, to_int(props, "cemrl.td.updates.per.actor", c.td_updates_per_actor));
    c.row_num = std::max(1, to_int(props, "row.num", c.row_num));
    c.robust_candidate_train_top_k = std::max(1, to_int(props, "robust.validation.candidate.train.top.k", c.robust_candidate_train_top_k));
    c.robust_candidate_history_top_k = std::max(0, to_int(props, "robust.validation.candidate.history.top.k", c.robust_candidate_history_top_k));
    c.robust_candidate_previous_val_top_k = std::max(0, to_int(props, "robust.validation.candidate.previous.val.top.k", c.robust_candidate_previous_val_top_k));
    c.robust_candidate_random_k = std::max(0, to_int(props, "robust.validation.candidate.random.k", c.robust_candidate_random_k));
    c.robust_validation_episodes = std::max(1, to_int(props, "robust.validation.episodes", c.robust_validation_episodes));
    c.robust_validation_std_penalty = std::max(0.0, to_double(props, "robust.validation.std.penalty", c.robust_validation_std_penalty));
    c.robust_train_score_weight = std::max(0.0, to_double(props, "robust.train.score.weight", c.robust_train_score_weight));
    c.robust_diversity_weight = std::max(0.0, to_double(props, "robust.diversity.weight", c.robust_diversity_weight));
    c.robust_deep_every = std::max(1, to_int(props, "robust.deep.every", c.robust_deep_every));
    c.robust_deep_episodes = std::max(1, to_int(props, "robust.deep.episodes", c.robust_deep_episodes));
    c.behavior_probe_states = std::max(1, to_int(props, "behavior.probe.states", c.behavior_probe_states));
    c.robust_pool_size = std::max(1, to_int(props, "robust.pool.size", c.robust_pool_size));
    c.robust_pool_current_share_early = std::clamp(to_double(props, "robust.pool.current.share.early", c.robust_pool_current_share_early), 0.0, 1.0);
    c.robust_pool_history_share_early = std::clamp(to_double(props, "robust.pool.history.share.early", c.robust_pool_history_share_early), 0.0, 1.0);
    c.robust_pool_current_share_late = std::clamp(to_double(props, "robust.pool.current.share.late", c.robust_pool_current_share_late), 0.0, 1.0);
    c.robust_pool_history_share_late = std::clamp(to_double(props, "robust.pool.history.share.late", c.robust_pool_history_share_late), 0.0, 1.0);
    c.robust_pool_late_after_iteration = std::max(1, to_int(props, "robust.pool.late.after.iteration", c.robust_pool_late_after_iteration));
    c.robust_final_top_n = std::max(1, to_int(props, "robust.final.top.n", c.robust_final_top_n));
    c.early_stop_enabled = to_bool(props, "early.stop.enabled", c.early_stop_enabled);
    c.early_stop_min_iterations = std::max(1, to_int(props, "early.stop.min.iterations", c.early_stop_min_iterations));
    c.early_stop_patience_deep_checks = std::max(1, to_int(props, "early.stop.patience.deep.checks", c.early_stop_patience_deep_checks));
    c.early_stop_min_delta = std::max(0.0, to_double(props, "early.stop.min.delta", c.early_stop_min_delta));
    c.early_stop_min_avg_std = std::max(0.0, to_double(props, "early.stop.min.avg.std", c.early_stop_min_avg_std));

    auto ratio_it = props.find("ratio.grid");
    if (ratio_it != props.end()) {
        std::vector<double> ratios;
        for (const std::string& token : split_list(ratio_it->second)) {
            try {
                double r = std::stod(token);
                if (r > 0.0 && r < 1.0) ratios.push_back(r);
            } catch (...) {}
        }
        if (!ratios.empty()) c.ratios = ratios;
    }
    return c;
}

int thread_count(const Config& config) {
    unsigned hc = std::thread::hardware_concurrency();
    int base = hc == 0 ? 1 : static_cast<int>(hc);
    return std::max(1, base - config.thread_reserve);
}

enum class Algorithm {
    Original,
    Pool,
    PoolDiversity,
    RobustPoolDiversity,
    RobustValidationRerankDiversity,
    RobustValidationStdDiversity,
    RobustValidationMemoryDiversity,
    Proportional,
};

std::string method_name(Algorithm a) {
    switch (a) {
        case Algorithm::Original: return "OriginalCE";
        case Algorithm::Pool: return "PoolCE";
        case Algorithm::PoolDiversity: return "PoolDiversityCE";
        case Algorithm::RobustPoolDiversity: return "RobustPoolDiversityCE";
        case Algorithm::RobustValidationRerankDiversity: return "RobustValidationRerankDiversityCE";
        case Algorithm::RobustValidationStdDiversity: return "RobustValidationStdDiversityCE";
        case Algorithm::RobustValidationMemoryDiversity: return "RobustValidationMemoryDiversityCE";
        case Algorithm::Proportional: return "ProportionalCE";
    }
    return "Unknown";
}

bool uses_pool(Algorithm a) {
    return a == Algorithm::Pool || a == Algorithm::PoolDiversity || a == Algorithm::RobustPoolDiversity
        || a == Algorithm::RobustValidationRerankDiversity
        || a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

bool uses_diversity(Algorithm a) {
    return a == Algorithm::PoolDiversity || a == Algorithm::RobustPoolDiversity
        || a == Algorithm::RobustValidationRerankDiversity
        || a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

bool uses_proportional(Algorithm a) {
    return a == Algorithm::Proportional;
}

bool uses_robust_pool_diversity(Algorithm a) {
    return a == Algorithm::RobustPoolDiversity
        || a == Algorithm::RobustValidationRerankDiversity
        || a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

bool is_robust_ablation_method(Algorithm a) {
    return a == Algorithm::RobustValidationRerankDiversity
        || a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

bool robust_uses_std_penalty(Algorithm a) {
    return a == Algorithm::RobustPoolDiversity
        || a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

bool robust_uses_validation_memory(Algorithm a) {
    return a == Algorithm::RobustPoolDiversity
        || a == Algorithm::RobustValidationMemoryDiversity;
}

std::string robust_score_formula(Algorithm a) {
    if (a == Algorithm::RobustValidationRerankDiversity) return "validation_mean";
    if (a == Algorithm::RobustValidationStdDiversity
        || a == Algorithm::RobustValidationMemoryDiversity) {
        return "validation_mean - std_penalty * validation_std";
    }
    return "validation_mean - std_penalty * validation_std + train_weight * normalized_train_score + diversity_weight * behavior_distance";
}

std::vector<Algorithm> parse_algorithms(const std::string& value, Log& log, bool ratio_mode) {
    std::vector<Algorithm> out;
    for (const std::string& token : split_list(value)) {
        std::string n = lower_no_underscore(token);
        if (n == "originalce" || n == "original") {
            if (ratio_mode) {
                log.warn("ratio experiment skips non-pool method: " + token);
            } else {
                out.push_back(Algorithm::Original);
            }
        } else if (n == "poolce" || n == "pool") {
            out.push_back(Algorithm::Pool);
        } else if (n == "pooldiversityce" || n == "pooldiversity") {
            out.push_back(Algorithm::PoolDiversity);
        } else if (n == "robustpooldiversityce" || n == "robustpooldiversity" || n == "robustpdce") {
            if (ratio_mode) {
                log.warn("ratio experiment skips robust benchmark method: " + token);
            } else {
                out.push_back(Algorithm::RobustPoolDiversity);
            }
        } else if (n == "robustvalidationrerankdiversityce" || n == "robustvalidationrerankdiversity"
                   || n == "robustrerankdiversityce" || n == "robustrerankdiversity") {
            if (ratio_mode) {
                log.warn("ratio experiment skips robust ablation method: " + token);
            } else {
                out.push_back(Algorithm::RobustValidationRerankDiversity);
            }
        } else if (n == "robustvalidationstddiversityce" || n == "robustvalidationstddiversity"
                   || n == "robuststddiversityce" || n == "robuststddiversity") {
            if (ratio_mode) {
                log.warn("ratio experiment skips robust ablation method: " + token);
            } else {
                out.push_back(Algorithm::RobustValidationStdDiversity);
            }
        } else if (n == "robustvalidationmemorydiversityce" || n == "robustvalidationmemorydiversity"
                   || n == "robustmemorydiversityce" || n == "robustmemorydiversity") {
            if (ratio_mode) {
                log.warn("ratio experiment skips robust ablation method: " + token);
            } else {
                out.push_back(Algorithm::RobustValidationMemoryDiversity);
            }
        } else if (n == "proportionalce" || n == "proportional") {
            if (ratio_mode) {
                log.warn("ratio experiment skips non-pool method: " + token);
            } else {
                out.push_back(Algorithm::Proportional);
            }
        } else if (n == "cemrl") {
            log.warn("CEMRL is intentionally not implemented in C++; skipping token: " + token);
        } else {
            throw std::runtime_error("unknown CE method: " + token);
        }
    }
    if (out.empty()) throw std::runtime_error("no runnable CE methods after parsing: " + value);
    return out;
}

enum class SelectionMode { TrainElite, ValElite };

std::string selection_label(SelectionMode s) {
    return s == SelectionMode::ValElite ? "val_elite" : "train_elite";
}

std::vector<SelectionMode> parse_selection_modes(const std::string& value) {
    std::vector<SelectionMode> out;
    for (const std::string& token : split_list(value)) {
        std::string n = lower_no_underscore(token);
        if (n == "trainelite") out.push_back(SelectionMode::TrainElite);
        else if (n == "valelite" || n == "validationelite") out.push_back(SelectionMode::ValElite);
        else throw std::runtime_error("unknown selection mode: " + token);
    }
    if (out.empty()) throw std::runtime_error("ratio.selection.modes must not be empty");
    return out;
}

std::vector<bool> parse_noise_levels(const std::string& value) {
    std::vector<bool> out;
    for (const std::string& token : split_list(value)) {
        std::string n = lower_no_underscore(token);
        if (n == "noiseoff" || n == "off" || n == "false") out.push_back(false);
        else if (n == "noiseon" || n == "on" || n == "true") out.push_back(true);
        else throw std::runtime_error("unknown noise level: " + token);
    }
    if (out.empty()) throw std::runtime_error("benchmark.noise.levels must not be empty");
    return out;
}

std::string noise_status(bool on) {
    return on ? "noise_on" : "noise_off";
}

std::string ratio_label(double train_share) {
    int train = static_cast<int>(std::llround(train_share * 100.0));
    int validation = 100 - train;
    return std::to_string(train) + "_" + std::to_string(validation);
}

int32_t java_hash_code(const std::string& s) {
    uint32_t h = 0;
    for (unsigned char c : s) h = h * 31u + c;
    return static_cast<int32_t>(h);
}

struct SeedManager {
    static constexpr long long TRAIN_OFFSET = 11000000LL;
    static constexpr long long VALIDATION_OFFSET = 33000000LL;
    static constexpr long long TEST_OFFSET = 44000000LL;
    static constexpr long long THETA_OFFSET = 55000000LL;
    static constexpr long long ROBUST_VALIDATION_OFFSET = 77000000LL;
    static constexpr long long ROBUST_DEEP_OFFSET = 88000000LL;
    static constexpr long long ROBUST_BEHAVIOR_OFFSET = 99000000LL;
    static constexpr long long ROBUST_CANDIDATE_OFFSET = 111000000LL;
    static constexpr long long ITERATION_STRIDE = 10000019LL;

    static long long run_seed(const Config& c, int run_id) {
        return c.base_seed + std::max(0, run_id - 1) * c.run_stride;
    }
    static long long theta_seed(const Config& c, int run_id) {
        return run_seed(c, run_id) + THETA_OFFSET;
    }
    static std::vector<int64_t> build_seeds(int64_t seed, int count) {
        JavaRandom rng(seed);
        std::vector<int64_t> seeds(count);
        for (int64_t& x : seeds) x = rng.next_long();
        return seeds;
    }
    static std::vector<int64_t> train_seeds(const Config& c, int run_id, int iteration) {
        int64_t seed = run_seed(c, run_id) + TRAIN_OFFSET + iteration * ITERATION_STRIDE;
        return build_seeds(seed, c.train_episodes);
    }
    static std::vector<int64_t> validation_seeds(const Config& c, int count, const std::string& scope) {
        int64_t h = java_hash_code(scope);
        int64_t salt = h < 0 ? -h : h;
        int64_t seed = c.base_seed + VALIDATION_OFFSET + salt * 1009LL;
        return build_seeds(seed, count);
    }
    static std::vector<int64_t> robust_validation_seeds(const Config& c, int run_id, int count) {
        return build_seeds(run_seed(c, run_id) + ROBUST_VALIDATION_OFFSET, count);
    }
    static std::vector<int64_t> robust_deep_seeds(const Config& c, int run_id, int count) {
        return build_seeds(run_seed(c, run_id) + ROBUST_DEEP_OFFSET, count);
    }
    static int64_t robust_behavior_seed(const Config& c, int run_id) {
        return run_seed(c, run_id) + ROBUST_BEHAVIOR_OFFSET;
    }
    static int64_t robust_candidate_seed(const Config& c, int run_id, int iteration) {
        return run_seed(c, run_id) + ROBUST_CANDIDATE_OFFSET + iteration * ITERATION_STRIDE;
    }
    static int64_t test_seed(const Config& c, int episode_index) {
        return c.base_seed + TEST_OFFSET + episode_index * ITERATION_STRIDE;
    }
};

struct EpisodeResult {
    int score = 0;
    int steps = 0;
};

int greedy_action(const State& s, const ValueFunction& vf) {
    uint64_t mask = action_list(s);
    if (mask == 0) return -1;
    int best_a = -1;
    double best_v = -std::numeric_limits<double>::infinity();
    int n = action_count(s.piece);
    for (int a = 0; a < n; ++a) {
        if (!action_valid(a, mask)) continue;
        double base[kFeatureDim];
        compute_features(s, a, base);
        double v = vf.value_from_features(base);
        if (v > best_v) {
            best_v = v;
            best_a = a;
        }
    }
    return best_a;
}

EpisodeResult run_episode(const ValueFunction& vf, int64_t seed) {
    JavaRandom rng(seed);
    State s = initial_state(rng.next_int(7));
    EpisodeResult out;
    for (int step = 0; step < kMaxSteps; ++step) {
        int a = greedy_action(s, vf);
        if (a < 0) break;
        s = after_state(s, a);
        s.score += s.reward;
        s.piece = rng.next_int(7);
        out.steps++;
        if (is_final(s)) break;
    }
    out.score = s.score;
    return out;
}

int random_valid_action(const State& s, JavaRandom& rng) {
    uint64_t mask = action_list(s);
    if (mask == 0) return -1;
    std::vector<int> actions;
    int n = action_count(s.piece);
    actions.reserve(n);
    for (int a = 0; a < n; ++a) {
        if (action_valid(a, mask)) actions.push_back(a);
    }
    if (actions.empty()) return -1;
    return actions[static_cast<size_t>(rng.next_int(static_cast<int32_t>(actions.size())))];
}

std::vector<State> build_behavior_probe_states(const Config& config, int run_id) {
    std::vector<State> states;
    states.reserve(static_cast<size_t>(config.behavior_probe_states));
    JavaRandom rng(SeedManager::robust_behavior_seed(config, run_id));
    State s = initial_state(rng.next_int(7));
    int attempts = 0;
    int max_attempts = std::max(1000, config.behavior_probe_states * 50);
    while (static_cast<int>(states.size()) < config.behavior_probe_states && attempts++ < max_attempts) {
        if (action_list(s) == 0 || is_final(s)) {
            s = initial_state(rng.next_int(7));
            continue;
        }
        states.push_back(s);
        int a = random_valid_action(s, rng);
        if (a < 0) {
            s = initial_state(rng.next_int(7));
            continue;
        }
        s = after_state(s, a);
        s.score += s.reward;
        s.piece = rng.next_int(7);
        if (is_final(s)) s = initial_state(rng.next_int(7));
    }
    if (states.empty()) states.push_back(initial_state(0));
    while (static_cast<int>(states.size()) < config.behavior_probe_states) {
        states.push_back(states.back());
    }
    return states;
}

double evaluate_theta_parallel(const std::vector<double>& theta,
                               const std::vector<int64_t>& seeds,
                               int threads) {
    if (seeds.empty()) return 0.0;
    if (threads <= 1 || seeds.size() == 1) {
        ValueFunction vf;
        vf.set_theta(theta);
        double total = 0.0;
        for (int64_t seed : seeds) total += run_episode(vf, seed).score;
        return total / static_cast<double>(seeds.size());
    }
    int n_threads = std::max(1, std::min<int>(threads, static_cast<int>(seeds.size())));
    std::atomic<size_t> next(0);
    std::vector<double> partial(n_threads, 0.0);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            ValueFunction vf;
            vf.set_theta(theta);
            double local = 0.0;
            while (true) {
                size_t i = next.fetch_add(1);
                if (i >= seeds.size()) break;
                local += run_episode(vf, seeds[i]).score;
            }
            partial[t] = local;
        });
    }
    for (auto& worker : workers) worker.join();
    double total = std::accumulate(partial.begin(), partial.end(), 0.0);
    return total / static_cast<double>(seeds.size());
}

struct EvalStats {
    double mean = 0.0;
    double stddev = 0.0;
};

EvalStats evaluate_theta_stats_parallel(const std::vector<double>& theta,
                                        const std::vector<int64_t>& seeds,
                                        int threads) {
    if (seeds.empty()) return {};
    int n_threads = std::max(1, std::min<int>(threads, static_cast<int>(seeds.size())));
    std::atomic<size_t> next(0);
    std::vector<double> sums(n_threads, 0.0);
    std::vector<double> squares(n_threads, 0.0);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            ValueFunction vf;
            vf.set_theta(theta);
            double local_sum = 0.0;
            double local_square = 0.0;
            while (true) {
                size_t i = next.fetch_add(1);
                if (i >= seeds.size()) break;
                double score = static_cast<double>(run_episode(vf, seeds[i]).score);
                local_sum += score;
                local_square += score * score;
            }
            sums[t] = local_sum;
            squares[t] = local_square;
        });
    }
    for (auto& worker : workers) worker.join();
    double sum = std::accumulate(sums.begin(), sums.end(), 0.0);
    double square = std::accumulate(squares.begin(), squares.end(), 0.0);
    double n = static_cast<double>(seeds.size());
    EvalStats stats;
    stats.mean = sum / n;
    stats.stddev = std::sqrt(std::max(0.0, square / n - stats.mean * stats.mean));
    return stats;
}

struct CandidateScore {
    int population_index = -1;
    int train_rank = -1;
    double train_score = 0.0;
    double validation_score = std::numeric_limits<double>::quiet_NaN();
    double validation_std = 0.0;
    double robust_score = std::numeric_limits<double>::quiet_NaN();
    double behavior_distance = 0.0;
    double validation_component = 0.0;
    double std_penalty_component = 0.0;
    double train_component_diagnostic = 0.0;
    double behavior_component_diagnostic = 0.0;
    std::vector<double> theta;
    std::vector<int> behavior_signature;
    std::string source = "population";
    int iteration = -1;
};

struct ScoredTheta {
    int source_iteration = -1;
    int source_rank = -1;
    double train_score = 0.0;
    double validation_score = 0.0;
    double validation_std = 0.0;
    double robust_score = 0.0;
    double behavior_distance = 0.0;
    std::vector<double> theta;
    std::vector<int> behavior_signature;
};

struct PoolEntry {
    double selection_score = 0.0;
    double train_score = 0.0;
    double validation_score = std::numeric_limits<double>::quiet_NaN();
    double validation_std = 0.0;
    double robust_score = std::numeric_limits<double>::quiet_NaN();
    double min_distance = 0.0;
    int source_iteration = -1;
    int source_rank = -1;
    std::vector<double> theta;
    std::vector<int> behavior_signature;
};

struct TrainTheta {
    int source_iteration = -1;
    int population_index = -1;
    double train_score = 0.0;
    std::vector<double> theta;
};

struct WeightedTheta {
    std::vector<double> theta;
    double weight = 0.0;
};

struct SummaryRow {
    std::string experiment;
    std::string method;
    std::string noise;
    std::string ratio;
    std::string selection;
    std::string selected_theta;
    std::string run_id;
    long long run_seed = 0;
    int source_iteration = 0;
    int source_rank = 0;
    double train_score = 0.0;
    double validation_mean = 0.0;
    double test_mean = 0.0;
    int train_episodes = 0;
    int validation_episodes = 0;
    int test_episodes = 0;
    int completed_iterations = 0;
    std::string status = "OK";
    std::string artifact_path;

    static std::string header() {
        return "experiment,method,noise_status,ratio,selection_mode,selected_theta,run_id,run_seed,source_iteration,source_rank,train_score,validation_mean,test_mean,train_episodes,validation_episodes,test_episodes,completed_iterations,status,artifact_path";
    }

    std::string to_csv() const {
        std::vector<std::string> cols = {
            csv_escape(experiment), csv_escape(method), csv_escape(noise), csv_escape(ratio),
            csv_escape(selection), csv_escape(selected_theta), csv_escape(run_id),
            std::to_string(run_seed), std::to_string(source_iteration), std::to_string(source_rank),
            fmt6(train_score), fmt6(validation_mean), fmt6(test_mean),
            std::to_string(train_episodes), std::to_string(validation_episodes),
            std::to_string(test_episodes), std::to_string(completed_iterations),
            csv_escape(status), csv_escape(artifact_path)
        };
        std::ostringstream os;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (i) os << ',';
            os << cols[i];
        }
        return os.str();
    }
};

class Trainer {
public:
    Trainer(const Config& config, std::string experiment_name, Algorithm algorithm,
            SelectionMode selection_mode, std::string ratio_label, int validation_episodes,
            int run_id, bool with_noise, fs::path result_dir, Log& log)
        : config_(config),
          experiment_name_(std::move(experiment_name)),
          algorithm_(algorithm),
          selection_mode_(selection_mode),
          ratio_label_(std::move(ratio_label)),
          validation_episodes_(std::max(1, validation_episodes)),
          run_id_(run_id),
          with_noise_(with_noise),
          result_dir_(std::move(result_dir)),
          log_(log),
          elite_num_(config.elite_num()),
          threads_(thread_count(config)),
          scores_(config.population, 0.0),
          thetas_(config.population),
          means_(kParameterNum, 0.0),
          stds_(kParameterNum, 10.0),
          theta_rng_(SeedManager::theta_seed(config, run_id)),
          validation_seeds_(SeedManager::validation_seeds(config, validation_episodes_, validation_scope())),
          robust_validation_seeds_(SeedManager::robust_validation_seeds(config, run_id, config.robust_validation_episodes)),
          robust_deep_seeds_(SeedManager::robust_deep_seeds(config, run_id, config.robust_deep_episodes)) {
        fs::create_directories(result_dir_);
        if (is_robust()) behavior_probe_states_ = build_behavior_probe_states(config_, run_id_);
        if (config_.resume_enabled) load_checkpoint();
        else init_outputs();
    }

    std::vector<SummaryRow> train() {
        auto run_start = std::chrono::steady_clock::now();
        log_.info("[" + label() + "] start targetIterations=" + std::to_string(config_.iterations)
                  + " completedIterations=" + std::to_string(completed_iterations_)
                  + " population=" + std::to_string(config_.population)
                  + " elite=" + std::to_string(elite_num_)
                  + " trainEpisodes=" + std::to_string(config_.train_episodes)
                  + " validationEpisodes=" + std::to_string(validation_episodes_)
                  + " testEpisodes=" + std::to_string(config_.test_episodes)
                  + " threads=" + std::to_string(threads_)
                  + " output=" + result_dir_.string());

        int start_iteration = completed_iterations_;
        for (int iteration = start_iteration; iteration < config_.iterations; ++iteration) {
            auto iter_start = std::chrono::steady_clock::now();
            auto train_seeds = SeedManager::train_seeds(config_, run_id_, iteration);
            double noise = calculate_noise(iteration);
            sample_population();
            evaluate_population(train_seeds);
            auto sorted = sorted_indices_by_score();
            double iter_best = scores_[sorted[0]];
            double iter_avg = std::accumulate(scores_.begin(), scores_.end(), 0.0) / scores_.size();
            update_best_train(iteration, sorted[0], iter_best);
            update_global_train_pool(iteration);
            write_global_top_weights();

            std::vector<CandidateScore> validated;
            std::vector<CandidateScore> selected;
            if (is_robust()) {
                validated = validate_robust_candidates(iteration, sorted);
                selected = select_robust_elites(validated);
            } else {
                validated = validate_top_candidates(iteration, sorted);
                selected = select_elites(sorted, validated);
            }
            log_population_scores(iteration);
            log_elites(iteration, selected);
            log_elite_weights(iteration, selected);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - iter_start).count();
            log_progress(iteration, iter_best, iter_avg, best_validation_score(), selected.size(), elapsed);
            if (is_robust()) {
                update_robust_distribution(selected, noise, iteration);
            } else {
                update_distribution(sorted, selected, noise);
            }
            if (uses_pool(algorithm_) && !is_robust()) {
                update_history_pool(selected);
                write_history_pool();
            }
            if (is_robust()) {
                update_robust_history_pool(selected);
                write_history_pool();
                write_behavior_pool();
            }
            write_validation_pool();
            auto best_validation = best_validation_theta();
            if (best_validation) write_theta(best_validation->theta, result_dir_ / "best_validation_theta.txt");
            completed_iterations_ = iteration + 1;
            if (is_robust()) {
                run_deep_validation_if_due(iteration);
                if (should_stop_early(iteration)) {
                    write_early_stop_file();
                    maybe_save_checkpoint();
                    break;
                }
            }
            maybe_save_checkpoint();

            auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - run_start).count();
            log_.info("[" + label() + "] iter " + std::to_string(iteration + 1) + "/"
                      + std::to_string(config_.iterations)
                      + " trainBest=" + fmt6(iter_best)
                      + " globalTrainBest=" + fmt6(best_train_score_)
                      + " avg=" + fmt6(iter_avg)
                      + " valBest=" + fmt6(best_validation_score())
                      + " elites=" + std::to_string(selected.size())
                      + " validationPool=" + std::to_string(validation_pool_.size())
                      + "/" + std::to_string(config_.global_top_n)
                      + " historyPool=" + std::to_string(history_pool_.size())
                      + "/" + std::to_string(is_robust() ? config_.robust_history_pool_size() : config_.pool_size())
                      + " noise=" + fmt6(noise)
                      + " avgStd=" + fmt6(average_std())
                      + " iterMs=" + std::to_string(elapsed)
                      + " elapsedMs=" + std::to_string(total_elapsed));
        }

        maybe_save_checkpoint();
        auto rows = write_final_outputs();
        log_.info("[" + label() + "] completed bestTrain=" + fmt6(best_train_score_)
                  + " bestValidation=" + fmt6(best_validation_score())
                  + " rows=" + std::to_string(rows.size()));
        return rows;
    }

private:
    std::string label() const {
        return method_name(algorithm_) + " " + noise_status(with_noise_) + " "
             + effective_selection_label() + " run_" + std::to_string(run_id_);
    }

    bool is_robust() const {
        return uses_robust_pool_diversity(algorithm_);
    }

    std::string effective_selection_label() const {
        return is_robust() ? "robust_validation" : selection_label(selection_mode_);
    }

    std::string validation_scope() const {
        return experiment_name_ + ":" + ratio_label_;
    }

    void init_outputs() {
        write_header("progress.csv", "iteration,best_train,avg_train,best_validation,elite_count,elapsed_ms");
        std::ostringstream pop;
        pop << "iteration";
        for (int i = 0; i < config_.population; ++i) pop << ",id_" << (i + 1);
        write_header("population_scores.csv", pop.str());
        write_header("elites.csv", "iteration,rank,population_index,train_rank,train_score,validation_score,selection_score");
        write_header("elite_weights.csv", theta_header("iteration,elite_rank,population_index,train_rank,train_score,validation_score,selection_score,selection_mode"));
        write_header("global_top_weights.csv", theta_header("rank,score,source_iteration"));
        write_header("validation_pool.csv", theta_header(is_robust()
            ? "rank,source_iteration,source_rank,train_score,validation_score,validation_std,robust_score,behavior_distance"
            : "rank,source_iteration,source_rank,train_score,validation_score"));
        if (uses_pool(algorithm_)) {
            write_header("pool_history.csv", theta_header(is_robust()
                ? "rank,selection_score,train_score,validation_score,validation_std,robust_score,behavior_distance,source_iteration,source_rank"
                : "rank,selection_score,train_score,validation_score,min_distance"));
        }
        if (is_robust()) {
            if (is_robust_ablation_method(algorithm_)) {
                write_header("candidate_validation.csv",
                             theta_header("iteration,rank,source,population_index,train_rank,train_score,validation_mean,validation_std,score_formula,validation_component,std_penalty_component,train_component_diagnostic,behavior_component_diagnostic,behavior_distance,selection_score,previous_val_enabled,robust_score"));
            } else {
                write_header("candidate_validation.csv",
                             theta_header("iteration,rank,source,population_index,train_rank,train_score,validation_mean,validation_std,behavior_distance,robust_score"));
            }
            write_header("deep_validation.csv",
                         theta_header("iteration,rank,source_iteration,source_rank,train_score,validation_mean,validation_std,deep_mean,deep_std,robust_score,improved"));
            write_header("behavior_pool.csv",
                         theta_header("rank,robust_score,validation_mean,validation_std,train_score,behavior_distance,source_iteration,source_rank,signature_hash"));
            write_header("early_stop.txt", "status=running");
        }
    }

    std::string theta_header(const std::string& prefix) const {
        std::ostringstream os;
        os << prefix;
        for (int i = 0; i < kParameterNum; ++i) os << ",weight_" << (i + 1);
        return os.str();
    }

    void write_header(const std::string& name, const std::string& header) {
        std::ofstream f(result_dir_ / name, std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + (result_dir_ / name).string());
        f << header << "\n";
    }

    void sample_population() {
        for (int i = 0; i < config_.population; ++i) {
            std::vector<double> theta(kParameterNum);
            for (int d = 0; d < kParameterNum; ++d) {
                theta[d] = theta_rng_.next_gaussian() * stds_[d] + means_[d];
            }
            thetas_[i] = std::move(theta);
        }
    }

    void evaluate_population(const std::vector<int64_t>& train_seeds) {
        std::atomic<int> next(0);
        int n_threads = std::max(1, std::min(threads_, config_.population));
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            workers.emplace_back([&]() {
                while (true) {
                    int idx = next.fetch_add(1);
                    if (idx >= config_.population) break;
                    scores_[idx] = evaluate_theta_parallel(thetas_[idx], train_seeds, 1);
                }
            });
        }
        for (auto& worker : workers) worker.join();
    }

    std::vector<int> sorted_indices_by_score() const {
        std::vector<int> sorted(config_.population);
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return scores_[a] > scores_[b];
        });
        return sorted;
    }

    bool theta_close(const std::vector<double>& a, const std::vector<double>& b) const {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6) return false;
        }
        return true;
    }

    bool candidate_exists(const std::vector<CandidateScore>& candidates,
                          const std::vector<double>& theta) const {
        for (const auto& candidate : candidates) {
            if (theta_close(candidate.theta, theta)) return true;
        }
        return false;
    }

    std::vector<int> behavior_signature(const std::vector<double>& theta) const {
        ValueFunction vf;
        vf.set_theta(theta);
        std::vector<int> signature;
        signature.reserve(behavior_probe_states_.size());
        for (const State& state : behavior_probe_states_) {
            signature.push_back(greedy_action(state, vf));
        }
        return signature;
    }

    static uint64_t signature_hash_value(const std::vector<int>& signature) {
        uint64_t h = 1469598103934665603ULL;
        for (int action : signature) {
            uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(action) + 129);
            for (int i = 0; i < 8; ++i) {
                h ^= static_cast<unsigned char>((v >> (i * 8)) & 0xffu);
                h *= 1099511628211ULL;
            }
        }
        return h;
    }

    double signature_distance(const std::vector<int>& a, const std::vector<int>& b) const {
        if (a.empty() || b.empty() || a.size() != b.size()) return 1.0;
        int diff = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) ++diff;
        }
        return static_cast<double>(diff) / static_cast<double>(a.size());
    }

    double behavior_distance_to_pool(const std::vector<int>& signature,
                                     const std::vector<double>* exclude_theta = nullptr) const {
        if (history_pool_.empty()) return 1.0;
        double best = 1.0;
        bool compared = false;
        for (const auto& entry : history_pool_) {
            if (exclude_theta && theta_close(entry.theta, *exclude_theta)) continue;
            compared = true;
            best = std::min(best, signature_distance(signature, entry.behavior_signature));
        }
        return compared ? best : 1.0;
    }

    double normalized_train_score(double train_score) const {
        if (!std::isfinite(train_score)) return 0.0;
        auto mm = std::minmax_element(scores_.begin(), scores_.end());
        double lo = *mm.first;
        double hi = *mm.second;
        if (hi - lo <= 1e-8) return 0.0;
        return std::clamp((train_score - lo) / (hi - lo), 0.0, 1.0);
    }

    double robust_score(double validation_mean, double validation_std,
                        double train_score, double behavior_distance) const {
        if (algorithm_ == Algorithm::RobustValidationRerankDiversity) {
            return validation_mean;
        }
        if (algorithm_ == Algorithm::RobustValidationStdDiversity
            || algorithm_ == Algorithm::RobustValidationMemoryDiversity) {
            return validation_mean - config_.robust_validation_std_penalty * validation_std;
        }
        return validation_mean
             - config_.robust_validation_std_penalty * validation_std
             + config_.robust_train_score_weight * normalized_train_score(train_score)
             + config_.robust_diversity_weight * behavior_distance;
    }

    void fill_robust_components(CandidateScore& candidate) const {
        candidate.validation_component = candidate.validation_score;
        candidate.std_penalty_component = robust_uses_std_penalty(algorithm_)
            ? -config_.robust_validation_std_penalty * candidate.validation_std
            : 0.0;
        candidate.train_component_diagnostic =
            config_.robust_train_score_weight * normalized_train_score(candidate.train_score);
        candidate.behavior_component_diagnostic =
            config_.robust_diversity_weight * candidate.behavior_distance;
    }

    bool previous_val_candidates_enabled() const {
        return robust_uses_validation_memory(algorithm_) && config_.robust_candidate_previous_val_top_k > 0;
    }

    CandidateScore current_candidate(int iteration, int population_index, int train_rank,
                                     const std::string& source) const {
        CandidateScore c;
        c.population_index = population_index;
        c.train_rank = train_rank;
        c.train_score = scores_[population_index];
        c.theta = thetas_[population_index];
        c.source = source;
        c.iteration = iteration;
        return c;
    }

    CandidateScore scored_candidate(const ScoredTheta& s, const std::string& source) const {
        CandidateScore c;
        c.population_index = -1;
        c.train_rank = s.source_rank;
        c.train_score = s.train_score;
        c.validation_score = s.validation_score;
        c.validation_std = s.validation_std;
        c.robust_score = s.robust_score;
        c.behavior_distance = s.behavior_distance;
        c.theta = s.theta;
        c.behavior_signature = s.behavior_signature;
        c.source = source;
        c.iteration = s.source_iteration;
        return c;
    }

    CandidateScore pool_candidate(const PoolEntry& p, const std::string& source) const {
        CandidateScore c;
        c.population_index = -1;
        c.train_rank = p.source_rank;
        c.train_score = p.train_score;
        c.validation_score = p.validation_score;
        c.validation_std = p.validation_std;
        c.robust_score = p.robust_score;
        c.behavior_distance = p.min_distance;
        c.theta = p.theta;
        c.behavior_signature = p.behavior_signature;
        c.source = source;
        c.iteration = p.source_iteration;
        return c;
    }

    void add_candidate(std::vector<CandidateScore>& candidates, const CandidateScore& candidate) const {
        if (!candidate_exists(candidates, candidate.theta)) candidates.push_back(candidate);
    }

    std::vector<CandidateScore> build_robust_candidate_set(int iteration,
                                                           const std::vector<int>& sorted) const {
        std::vector<CandidateScore> candidates;
        int train_count = std::min<int>(config_.robust_candidate_train_top_k, sorted.size());
        for (int rank = 0; rank < train_count; ++rank) {
            add_candidate(candidates, current_candidate(iteration, sorted[rank], rank, "train_top"));
        }

        if (previous_val_candidates_enabled()) {
            auto val_entries = validation_pool_sorted();
            int val_count = std::min<int>(config_.robust_candidate_previous_val_top_k, val_entries.size());
            for (int i = 0; i < val_count; ++i) {
                add_candidate(candidates, scored_candidate(val_entries[i], "previous_val_top"));
            }
        }

        std::vector<PoolEntry> hist = history_pool_;
        std::sort(hist.begin(), hist.end(), [](const PoolEntry& a, const PoolEntry& b) {
            return a.robust_score > b.robust_score;
        });
        int hist_count = std::min<int>(config_.robust_candidate_history_top_k, hist.size());
        for (int i = 0; i < hist_count; ++i) {
            add_candidate(candidates, pool_candidate(hist[i], "history_top"));
        }

        if (config_.robust_candidate_random_k > 0 && !sorted.empty()) {
            JavaRandom rng(SeedManager::robust_candidate_seed(config_, run_id_, iteration));
            int added = 0;
            int attempts = 0;
            int max_attempts = std::max(config_.population * 4, config_.robust_candidate_random_k * 20);
            while (added < config_.robust_candidate_random_k && attempts++ < max_attempts) {
                int idx = rng.next_int(config_.population);
                size_t rank_pos = static_cast<size_t>(std::find(sorted.begin(), sorted.end(), idx) - sorted.begin());
                int rank = rank_pos < sorted.size() ? static_cast<int>(rank_pos) : -1;
                size_t before = candidates.size();
                add_candidate(candidates, current_candidate(iteration, idx, rank, "random"));
                if (candidates.size() > before) ++added;
            }
        }
        return candidates;
    }

    std::vector<CandidateScore> validate_robust_candidates(int iteration,
                                                           const std::vector<int>& sorted) {
        std::vector<CandidateScore> candidates = build_robust_candidate_set(iteration, sorted);
        for (auto& candidate : candidates) {
            if (candidate.behavior_signature.empty()) {
                candidate.behavior_signature = behavior_signature(candidate.theta);
            }
            candidate.behavior_distance = behavior_distance_to_pool(candidate.behavior_signature,
                                                                    &candidate.theta);
            EvalStats stats = evaluate_theta_stats_parallel(candidate.theta, robust_validation_seeds_, threads_);
            candidate.validation_score = stats.mean;
            candidate.validation_std = stats.stddev;
            candidate.robust_score = robust_score(stats.mean, stats.stddev,
                                                  candidate.train_score,
                                                  candidate.behavior_distance);
            fill_robust_components(candidate);
            offer_validation_pool({candidate.iteration, candidate.train_rank, candidate.train_score,
                                   candidate.validation_score, candidate.validation_std,
                                   candidate.robust_score, candidate.behavior_distance,
                                   candidate.theta, candidate.behavior_signature});
        }
        std::sort(candidates.begin(), candidates.end(), [](const CandidateScore& a, const CandidateScore& b) {
            if (a.robust_score != b.robust_score) return a.robust_score > b.robust_score;
            return a.validation_score > b.validation_score;
        });
        log_candidate_validation(iteration, candidates);
        return candidates;
    }

    std::vector<CandidateScore> select_robust_elites(const std::vector<CandidateScore>& validated) const {
        std::vector<CandidateScore> selected = validated;
        if (static_cast<int>(selected.size()) > elite_num_) selected.resize(elite_num_);
        return selected;
    }

    std::vector<CandidateScore> validate_top_candidates(int iteration, const std::vector<int>& sorted) {
        int count = std::min<int>(std::max(config_.validation_top_k, elite_num_), sorted.size());
        std::vector<CandidateScore> out;
        out.reserve(count);
        for (int rank = 0; rank < count; ++rank) {
            int idx = sorted[rank];
            double val = evaluate_theta_parallel(thetas_[idx], validation_seeds_, threads_);
            CandidateScore c;
            c.population_index = idx;
            c.train_rank = rank;
            c.train_score = scores_[idx];
            c.validation_score = val;
            c.validation_std = 0.0;
            c.robust_score = val;
            c.theta = thetas_[idx];
            c.iteration = iteration;
            out.push_back(c);
            ScoredTheta scored;
            scored.source_iteration = iteration;
            scored.source_rank = rank;
            scored.train_score = scores_[idx];
            scored.validation_score = val;
            scored.robust_score = val;
            scored.theta = thetas_[idx];
            offer_validation_pool(scored);
        }
        return out;
    }

    std::vector<CandidateScore> select_elites(const std::vector<int>& sorted,
                                              const std::vector<CandidateScore>& validated) const {
        if (selection_mode_ == SelectionMode::ValElite) {
            std::vector<CandidateScore> by_val = validated;
            std::sort(by_val.begin(), by_val.end(), [](const CandidateScore& a, const CandidateScore& b) {
                if (a.validation_score != b.validation_score) return a.validation_score > b.validation_score;
                return a.train_score > b.train_score;
            });
            if (static_cast<int>(by_val.size()) > elite_num_) by_val.resize(elite_num_);
            return by_val;
        }

        std::vector<CandidateScore> elites;
        int count = std::min<int>(elite_num_, sorted.size());
        elites.reserve(count);
        for (int rank = 0; rank < count; ++rank) {
            int idx = sorted[rank];
            auto it = std::find_if(validated.begin(), validated.end(), [&](const CandidateScore& c) {
                return c.population_index == idx;
            });
            if (it != validated.end()) {
                elites.push_back(*it);
            } else {
                elites.push_back(current_candidate(-1, idx, rank, "train_top"));
            }
        }
        return elites;
    }

    void update_distribution(const std::vector<int>& sorted,
                             const std::vector<CandidateScore>& selected,
                             double noise) {
        if (uses_proportional(algorithm_)) {
            update_proportional(sorted, noise);
            return;
        }
        std::vector<WeightedTheta> samples;
        if (uses_pool(algorithm_)) {
            samples = build_pool_samples(selected);
        } else {
            for (int i = 0; i < elite_num_; ++i) {
                samples.push_back({thetas_[sorted[i]], 1.0 / elite_num_});
            }
        }
        update_from_weighted_samples(samples, noise);
    }

    void update_proportional(const std::vector<int>& sorted, double noise) {
        std::vector<double> weights(elite_num_, 0.0);
        double baseline = scores_[sorted[elite_num_ - 1]];
        double total = 0.0;
        for (int i = 0; i < elite_num_; ++i) {
            weights[i] = std::max(0.0, scores_[sorted[i]] - baseline);
            total += weights[i];
        }
        if (total <= 1e-8) {
            std::fill(weights.begin(), weights.end(), 1.0 / elite_num_);
        } else {
            for (double& w : weights) w /= total;
            bool clipped = false;
            for (double& w : weights) {
                if (w > config_.proportional_max_weight_share) {
                    w = config_.proportional_max_weight_share;
                    clipped = true;
                }
            }
            if (clipped) {
                double clipped_total = std::accumulate(weights.begin(), weights.end(), 0.0);
                for (double& w : weights) w = clipped_total <= 1e-8 ? 1.0 / elite_num_ : w / clipped_total;
            }
        }
        std::vector<WeightedTheta> samples;
        for (int i = 0; i < elite_num_; ++i) samples.push_back({thetas_[sorted[i]], weights[i]});
        update_from_weighted_samples(samples, noise);
    }

    std::vector<WeightedTheta> build_pool_samples(const std::vector<CandidateScore>& selected) const {
        std::vector<WeightedTheta> samples;
        double total_share = config_.pool_elite_weight_share + config_.pool_history_weight_share;
        double elite_share = history_pool_.empty() || total_share <= 0.0
            ? 1.0 : config_.pool_elite_weight_share / total_share;
        double pool_share = history_pool_.empty() || total_share <= 0.0
            ? 0.0 : config_.pool_history_weight_share / total_share;
        double elite_weight = selected.empty() ? 0.0 : elite_share / selected.size();
        for (const auto& elite : selected) samples.push_back({elite.theta, elite_weight});
        if (!history_pool_.empty()) {
            double pool_weight = pool_share / history_pool_.size();
            for (const auto& entry : history_pool_) samples.push_back({entry.theta, pool_weight});
        }
        return samples;
    }

    void update_robust_distribution(const std::vector<CandidateScore>& selected,
                                    double noise,
                                    int iteration) {
        std::vector<WeightedTheta> samples;
        double current_share = iteration + 1 <= config_.robust_pool_late_after_iteration
            ? config_.robust_pool_current_share_early
            : config_.robust_pool_current_share_late;
        double history_share = iteration + 1 <= config_.robust_pool_late_after_iteration
            ? config_.robust_pool_history_share_early
            : config_.robust_pool_history_share_late;
        double total_share = current_share + history_share;
        if (history_pool_.empty()) {
            current_share = 1.0;
            history_share = 0.0;
        } else if (total_share > 1e-12) {
            current_share /= total_share;
            history_share /= total_share;
        } else {
            current_share = 1.0;
            history_share = 0.0;
        }

        double current_weight = selected.empty() ? 0.0 : current_share / selected.size();
        for (const auto& elite : selected) samples.push_back({elite.theta, current_weight});
        if (!history_pool_.empty() && history_share > 0.0) {
            double history_weight = history_share / history_pool_.size();
            for (const auto& entry : history_pool_) samples.push_back({entry.theta, history_weight});
        }
        update_from_weighted_samples(samples, noise);
    }

    void update_from_weighted_samples(const std::vector<WeightedTheta>& samples, double noise) {
        if (samples.empty()) return;
        for (int d = 0; d < kParameterNum; ++d) {
            double mean = 0.0;
            double square = 0.0;
            for (const auto& sample : samples) {
                double value = sample.theta[d];
                mean += sample.weight * value;
                square += sample.weight * value * value;
            }
            double variance = std::max(0.0, square - mean * mean);
            means_[d] = mean;
            stds_[d] = std::sqrt(std::max(variance, 1e-8) + noise);
        }
    }

    void update_history_pool(const std::vector<CandidateScore>& selected) {
        for (const auto& elite : selected) {
            if (theta_in_history(elite.theta)) continue;
            double selection = history_selection_score(elite, selected);
            PoolEntry entry;
            entry.selection_score = selection;
            entry.train_score = elite.train_score;
            entry.validation_score = elite.validation_score;
            entry.validation_std = elite.validation_std;
            entry.robust_score = selection;
            entry.min_distance = diversity_distance(elite.theta);
            entry.source_iteration = elite.iteration;
            entry.source_rank = elite.train_rank;
            entry.theta = elite.theta;
            entry.behavior_signature = elite.behavior_signature;
            history_pool_.push_back(entry);
        }
        std::sort(history_pool_.begin(), history_pool_.end(), [](const PoolEntry& a, const PoolEntry& b) {
            return a.selection_score > b.selection_score;
        });
        if (static_cast<int>(history_pool_.size()) > config_.pool_size()) {
            history_pool_.resize(config_.pool_size());
        }
    }

    void update_robust_history_pool(const std::vector<CandidateScore>& selected) {
        for (const auto& elite : selected) {
            if (theta_in_history(elite.theta)) continue;
            PoolEntry entry;
            entry.selection_score = elite.robust_score;
            entry.train_score = elite.train_score;
            entry.validation_score = elite.validation_score;
            entry.validation_std = elite.validation_std;
            entry.robust_score = elite.robust_score;
            entry.min_distance = elite.behavior_distance;
            entry.source_iteration = elite.iteration;
            entry.source_rank = elite.train_rank;
            entry.theta = elite.theta;
            entry.behavior_signature = elite.behavior_signature.empty()
                ? behavior_signature(elite.theta)
                : elite.behavior_signature;
            history_pool_.push_back(entry);
        }
        std::sort(history_pool_.begin(), history_pool_.end(), [](const PoolEntry& a, const PoolEntry& b) {
            if (a.robust_score != b.robust_score) return a.robust_score > b.robust_score;
            return a.validation_score > b.validation_score;
        });
        if (static_cast<int>(history_pool_.size()) > config_.robust_history_pool_size()) {
            history_pool_.resize(config_.robust_history_pool_size());
        }
    }

    bool theta_in_history(const std::vector<double>& theta) const {
        for (const auto& entry : history_pool_) {
            bool same = true;
            for (int i = 0; i < kParameterNum; ++i) {
                if (std::abs(theta[i] - entry.theta[i]) > 1e-6) {
                    same = false;
                    break;
                }
            }
            if (same) return true;
        }
        return false;
    }

    double history_selection_score(const CandidateScore& candidate,
                                   const std::vector<CandidateScore>& selected) const {
        double base = selection_mode_ == SelectionMode::ValElite && std::isfinite(candidate.validation_score)
            ? candidate.validation_score : candidate.train_score;
        if (!uses_diversity(algorithm_)) return base;
        double min_score = std::numeric_limits<double>::infinity();
        double max_score = -std::numeric_limits<double>::infinity();
        double min_dist = std::numeric_limits<double>::infinity();
        double max_dist = -std::numeric_limits<double>::infinity();
        double candidate_dist = 0.0;
        for (const auto& elite : selected) {
            double score = selection_mode_ == SelectionMode::ValElite && std::isfinite(elite.validation_score)
                ? elite.validation_score : elite.train_score;
            double dist = diversity_distance(elite.theta);
            min_score = std::min(min_score, score);
            max_score = std::max(max_score, score);
            min_dist = std::min(min_dist, dist);
            max_dist = std::max(max_dist, dist);
            if (elite.population_index == candidate.population_index) candidate_dist = dist;
        }
        double norm_score = (base - min_score) / (max_score - min_score + 1e-8);
        double norm_dist = (candidate_dist - min_dist) / (max_dist - min_dist + 1e-8);
        return norm_score + config_.diversity_lambda * norm_dist;
    }

    double diversity_distance(const std::vector<double>& theta) const {
        if (history_pool_.empty()) return 1000.0;
        double best = std::numeric_limits<double>::infinity();
        for (const auto& entry : history_pool_) {
            double sum = 0.0;
            for (int i = 0; i < kParameterNum; ++i) {
                double d = theta[i] - entry.theta[i];
                sum += d * d;
            }
            best = std::min(best, std::sqrt(sum) / std::sqrt(static_cast<double>(kParameterNum)));
        }
        return best;
    }

    double validation_rank_score(const ScoredTheta& theta) const {
        if (is_robust() && std::isfinite(theta.robust_score)) return theta.robust_score;
        return theta.validation_score;
    }

    void offer_validation_pool(const ScoredTheta& candidate) {
        for (auto& existing : validation_pool_) {
            if (theta_close(existing.theta, candidate.theta)) {
                if (validation_rank_score(candidate) > validation_rank_score(existing)) existing = candidate;
                return;
            }
        }
        validation_pool_.push_back(candidate);
        std::sort(validation_pool_.begin(), validation_pool_.end(), [&](const ScoredTheta& a, const ScoredTheta& b) {
            double ar = validation_rank_score(a);
            double br = validation_rank_score(b);
            if (ar != br) return ar > br;
            return a.validation_score > b.validation_score;
        });
        int limit = is_robust() ? std::max(config_.global_top_n, config_.robust_final_top_n) : config_.global_top_n;
        if (static_cast<int>(validation_pool_.size()) > limit) {
            validation_pool_.resize(limit);
        }
    }

    std::vector<ScoredTheta> validation_pool_sorted() const {
        auto out = validation_pool_;
        std::sort(out.begin(), out.end(), [&](const ScoredTheta& a, const ScoredTheta& b) {
            double ar = validation_rank_score(a);
            double br = validation_rank_score(b);
            if (ar != br) return ar > br;
            return a.validation_score > b.validation_score;
        });
        return out;
    }

    const ScoredTheta* best_validation_theta() const {
        if (validation_pool_.empty()) return nullptr;
        return &*std::max_element(validation_pool_.begin(), validation_pool_.end(),
            [&](const ScoredTheta& a, const ScoredTheta& b) {
                return validation_rank_score(a) < validation_rank_score(b);
            });
    }

    double best_validation_score() const {
        const ScoredTheta* best = best_validation_theta();
        return best ? best->validation_score : 0.0;
    }

    void update_best_train(int iteration, int best_index, double score) {
        if (score > best_train_score_) {
            best_train_score_ = score;
            best_train_theta_ = thetas_[best_index];
            best_train_iteration_ = iteration;
            best_train_rank_ = 0;
            write_theta(best_train_theta_, result_dir_ / "best_train_theta.txt");
        }
    }

    void update_global_train_pool(int iteration) {
        for (int i = 0; i < config_.population; ++i) {
            global_train_pool_.push_back({iteration, i, scores_[i], thetas_[i]});
        }
        std::sort(global_train_pool_.begin(), global_train_pool_.end(), [](const TrainTheta& a, const TrainTheta& b) {
            return a.train_score > b.train_score;
        });
        if (static_cast<int>(global_train_pool_.size()) > config_.global_top_n) {
            global_train_pool_.resize(config_.global_top_n);
        }
    }

    void log_progress(int iteration, double best, double avg, double validation_best,
                      size_t elite_count, long long elapsed_ms) {
        std::ofstream f(result_dir_ / "progress.csv", std::ios::app);
        f << (iteration + 1) << ',' << fmt6(best) << ',' << fmt6(avg) << ','
          << fmt6(validation_best) << ',' << elite_count << ',' << elapsed_ms << "\n";
    }

    void log_population_scores(int iteration) {
        std::ofstream f(result_dir_ / "population_scores.csv", std::ios::app);
        f << (iteration + 1);
        for (double score : scores_) f << ',' << fmt6(score);
        f << "\n";
    }

    void log_candidate_validation(int iteration, const std::vector<CandidateScore>& candidates) {
        if (!is_robust()) return;
        std::ofstream f(result_dir_ / "candidate_validation.csv", std::ios::app);
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& c : candidates) {
            f << (iteration + 1) << ',' << rank++ << ',' << c.source << ','
              << (c.population_index + 1) << ',' << (c.train_rank + 1) << ','
              << fmt6(c.train_score) << ',' << fmt6(c.validation_score) << ','
              << fmt6(c.validation_std);
            if (is_robust_ablation_method(algorithm_)) {
                f << ',' << csv_escape(robust_score_formula(algorithm_))
                  << ',' << fmt6(c.validation_component)
                  << ',' << fmt6(c.std_penalty_component)
                  << ',' << fmt6(c.train_component_diagnostic)
                  << ',' << fmt6(c.behavior_component_diagnostic)
                  << ',' << fmt6(c.behavior_distance)
                  << ',' << fmt6(c.robust_score)
                  << ',' << (previous_val_candidates_enabled() ? "true" : "false")
                  << ',' << fmt6(c.robust_score);
            } else {
                f << ',' << fmt6(c.behavior_distance) << ','
                  << fmt6(c.robust_score);
            }
            for (double x : c.theta) f << ',' << x;
            f << "\n";
        }
    }

    void log_elites(int iteration, const std::vector<CandidateScore>& elites) {
        std::ofstream f(result_dir_ / "elites.csv", std::ios::app);
        for (size_t i = 0; i < elites.size(); ++i) {
            const auto& e = elites[i];
            double selection = is_robust() && std::isfinite(e.robust_score)
                ? e.robust_score
                : selection_mode_ == SelectionMode::ValElite && std::isfinite(e.validation_score)
                ? e.validation_score : e.train_score;
            f << (iteration + 1) << ',' << (i + 1) << ',' << (e.population_index + 1)
              << ',' << (e.train_rank + 1) << ',' << fmt6(e.train_score)
              << ',' << fmt6(e.validation_score) << ',' << fmt6(selection) << "\n";
        }
    }

    void log_elite_weights(int iteration, const std::vector<CandidateScore>& elites) {
        std::ofstream f(result_dir_ / "elite_weights.csv", std::ios::app);
        f << std::setprecision(10) << std::fixed;
        for (size_t i = 0; i < elites.size(); ++i) {
            const auto& e = elites[i];
            double selection = is_robust() && std::isfinite(e.robust_score)
                ? e.robust_score
                : selection_mode_ == SelectionMode::ValElite && std::isfinite(e.validation_score)
                ? e.validation_score : e.train_score;
            f << (iteration + 1) << ',' << (i + 1) << ',' << (e.population_index + 1)
              << ',' << (e.train_rank + 1) << ',' << fmt6(e.train_score)
              << ',' << fmt6(e.validation_score) << ',' << fmt6(selection)
              << ',' << effective_selection_label();
            for (double x : e.theta) f << ',' << x;
            f << "\n";
        }
    }

    void write_global_top_weights() {
        std::ofstream f(result_dir_ / "global_top_weights.csv", std::ios::trunc);
        f << theta_header("rank,score,source_iteration") << "\n";
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& e : global_train_pool_) {
            f << rank++ << ',' << fmt6(e.train_score) << ',' << (e.source_iteration + 1);
            for (double x : e.theta) f << ',' << x;
            f << "\n";
        }
    }

    void write_validation_pool() {
        std::ofstream f(result_dir_ / "validation_pool.csv", std::ios::trunc);
        f << theta_header(is_robust()
            ? "rank,source_iteration,source_rank,train_score,validation_score,validation_std,robust_score,behavior_distance"
            : "rank,source_iteration,source_rank,train_score,validation_score") << "\n";
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& e : validation_pool_sorted()) {
            f << rank++ << ',' << (e.source_iteration + 1) << ',' << (e.source_rank + 1)
              << ',' << fmt6(e.train_score) << ',' << fmt6(e.validation_score);
            if (is_robust()) {
                f << ',' << fmt6(e.validation_std) << ',' << fmt6(e.robust_score)
                  << ',' << fmt6(e.behavior_distance);
            }
            for (double x : e.theta) f << ',' << x;
            f << "\n";
        }
    }

    void write_history_pool() {
        if (!uses_pool(algorithm_)) return;
        std::ofstream f(result_dir_ / "pool_history.csv", std::ios::trunc);
        f << theta_header(is_robust()
            ? "rank,selection_score,train_score,validation_score,validation_std,robust_score,behavior_distance,source_iteration,source_rank"
            : "rank,selection_score,train_score,validation_score,min_distance") << "\n";
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& e : history_pool_) {
            f << rank++ << ',' << fmt6(e.selection_score) << ',' << fmt6(e.train_score)
              << ',' << fmt6(e.validation_score);
            if (is_robust()) {
                f << ',' << fmt6(e.validation_std) << ',' << fmt6(e.robust_score)
                  << ',' << fmt6(e.min_distance) << ',' << (e.source_iteration + 1)
                  << ',' << (e.source_rank + 1);
            } else {
                f << ',' << fmt6(e.min_distance);
            }
            for (double x : e.theta) f << ',' << x;
            f << "\n";
        }
    }

    void write_behavior_pool() {
        if (!is_robust()) return;
        std::ofstream f(result_dir_ / "behavior_pool.csv", std::ios::trunc);
        f << theta_header("rank,robust_score,validation_mean,validation_std,train_score,behavior_distance,source_iteration,source_rank,signature_hash") << "\n";
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& e : history_pool_) {
            f << rank++ << ',' << fmt6(e.robust_score) << ',' << fmt6(e.validation_score)
              << ',' << fmt6(e.validation_std) << ',' << fmt6(e.train_score)
              << ',' << fmt6(e.min_distance) << ',' << (e.source_iteration + 1)
              << ',' << (e.source_rank + 1) << ',' << signature_hash_value(e.behavior_signature);
            for (double x : e.theta) f << ',' << x;
            f << "\n";
        }
    }

    void write_theta(const std::vector<double>& theta, const fs::path& path) const {
        std::ofstream f(path, std::ios::trunc);
        f << std::setprecision(17);
        for (double x : theta) f << x << "\n";
    }

    void run_deep_validation_if_due(int iteration) {
        if (!is_robust()) return;
        int completed = iteration + 1;
        if (completed % config_.robust_deep_every != 0 && completed < config_.iterations) return;

        auto candidates = validation_pool_sorted();
        if (static_cast<int>(candidates.size()) > config_.robust_final_top_n) {
            candidates.resize(config_.robust_final_top_n);
        }
        if (candidates.empty()) return;

        double previous_best = best_deep_validation_;
        double iteration_best = -std::numeric_limits<double>::infinity();
        std::vector<ScoredTheta> refreshed;
        refreshed.reserve(candidates.size());
        std::ofstream f(result_dir_ / "deep_validation.csv", std::ios::app);
        f << std::setprecision(10) << std::fixed;
        int rank = 1;
        for (const auto& candidate : candidates) {
            EvalStats stats = evaluate_theta_stats_parallel(candidate.theta, robust_deep_seeds_, threads_);
            double distance = candidate.behavior_distance;
            std::vector<int> sig = candidate.behavior_signature;
            if (sig.empty()) sig = behavior_signature(candidate.theta);
            if (!std::isfinite(distance) || distance <= 0.0) {
                distance = behavior_distance_to_pool(sig, &candidate.theta);
            }
            double score = robust_score(stats.mean, stats.stddev, candidate.train_score, distance);
            bool improved = stats.mean > previous_best + config_.early_stop_min_delta;
            f << completed << ',' << rank++ << ',' << (candidate.source_iteration + 1)
              << ',' << (candidate.source_rank + 1) << ',' << fmt6(candidate.train_score)
              << ',' << fmt6(candidate.validation_score) << ',' << fmt6(candidate.validation_std)
              << ',' << fmt6(stats.mean) << ',' << fmt6(stats.stddev)
              << ',' << fmt6(score) << ',' << (improved ? "true" : "false");
            for (double x : candidate.theta) f << ',' << x;
            f << "\n";

            ScoredTheta updated = candidate;
            updated.validation_score = stats.mean;
            updated.validation_std = stats.stddev;
            updated.robust_score = score;
            updated.behavior_distance = distance;
            updated.behavior_signature = sig;
            refreshed.push_back(updated);
            iteration_best = std::max(iteration_best, stats.mean);
        }
        for (const auto& entry : refreshed) offer_validation_pool(entry);

        if (iteration_best > best_deep_validation_ + config_.early_stop_min_delta) {
            best_deep_validation_ = iteration_best;
            best_deep_iteration_ = completed;
            stale_deep_checks_ = 0;
        } else {
            ++stale_deep_checks_;
        }
        log_.info("[" + label() + "] deep validation iter=" + std::to_string(completed)
                  + " bestThisCheck=" + fmt6(iteration_best)
                  + " bestDeep=" + fmt6(best_deep_validation_)
                  + " staleChecks=" + std::to_string(stale_deep_checks_));
    }

    bool should_stop_early(int iteration) {
        if (!is_robust() || !config_.early_stop_enabled) return false;
        int completed = iteration + 1;
        if (completed < config_.early_stop_min_iterations) return false;
        if (stale_deep_checks_ >= config_.early_stop_patience_deep_checks) {
            early_stop_triggered_ = true;
            early_stop_iteration_ = completed;
            early_stop_reason_ = "deep_validation_patience";
            return true;
        }
        if (average_std() < config_.early_stop_min_avg_std && stale_deep_checks_ > 0) {
            early_stop_triggered_ = true;
            early_stop_iteration_ = completed;
            early_stop_reason_ = "avg_std_converged_without_deep_improvement";
            return true;
        }
        return false;
    }

    void write_early_stop_file() const {
        if (!is_robust()) return;
        std::ofstream f(result_dir_ / "early_stop.txt", std::ios::trunc);
        f << "status=" << (early_stop_triggered_ ? "triggered" : "not_triggered") << "\n";
        f << "reason=" << early_stop_reason_ << "\n";
        f << "stop_iteration=" << early_stop_iteration_ << "\n";
        f << "best_deep_iteration=" << best_deep_iteration_ << "\n";
        f << "best_deep_validation=" << fmt6(best_deep_validation_) << "\n";
        f << "stale_deep_checks=" << stale_deep_checks_ << "\n";
        f << "average_std=" << fmt6(average_std()) << "\n";
    }

    std::vector<SummaryRow> write_final_outputs() {
        std::vector<SummaryRow> rows;
        auto candidates = validation_pool_sorted();
        if (is_robust() && static_cast<int>(candidates.size()) > config_.robust_final_top_n) {
            candidates.resize(config_.robust_final_top_n);
        }
        fs::path final_test = result_dir_ / ("final_test_iter_" + padded(completed_iterations_, 3) + ".csv");
        log_.info("[" + label() + "] final test start candidates=" + std::to_string(candidates.size()));
        std::ofstream f(final_test, std::ios::trunc);
        if (is_robust()) {
            struct FinalCandidate {
                ScoredTheta theta;
                int selection_rank = 0;
                double test_score = 0.0;
            };
            std::vector<FinalCandidate> final_rows;
            int selection_rank = 1;
            for (const auto& c : candidates) {
                final_rows.push_back({c, selection_rank++, evaluate_test(c.theta)});
            }
            std::sort(final_rows.begin(), final_rows.end(), [](const FinalCandidate& a, const FinalCandidate& b) {
                return a.test_score > b.test_score;
            });
            f << "final_rank,selection_rank,source_iteration,source_rank,train_score,validation_score,validation_std,robust_score,behavior_distance,test_score\n";
            int final_rank = 1;
            for (const auto& row : final_rows) {
                const auto& c = row.theta;
                f << final_rank++ << ',' << row.selection_rank << ','
                  << (c.source_iteration + 1) << ',' << (c.source_rank + 1)
                  << ',' << fmt6(c.train_score) << ',' << fmt6(c.validation_score)
                  << ',' << fmt6(c.validation_std) << ',' << fmt6(c.robust_score)
                  << ',' << fmt6(c.behavior_distance) << ',' << fmt6(row.test_score) << "\n";
            }
            f.close();
            copy_latest(final_test, result_dir_ / "final_test.csv");

            if (!candidates.empty()) {
                const auto& selected = candidates[0];
                double selected_test = 0.0;
                for (const auto& row : final_rows) {
                    if (theta_close(row.theta.theta, selected.theta)) {
                        selected_test = row.test_score;
                        break;
                    }
                }
                rows.push_back(summary_row("validation_selected", selected.source_iteration + 1,
                                           selected.source_rank + 1, selected.train_score,
                                           selected.validation_score, selected_test,
                                           final_test.string()));
            }
            if (!final_rows.empty()) {
                const auto& best = final_rows[0].theta;
                rows.push_back(summary_row("best_final_test_selected", best.source_iteration + 1,
                                           best.source_rank + 1, best.train_score,
                                           best.validation_score, final_rows[0].test_score,
                                           final_test.string()));
            }
        } else {
            f << "rank,source_iteration,source_rank,train_score,validation_score,test_score\n";
            int rank = 1;
            std::vector<double> test_scores;
            for (const auto& c : candidates) {
                double test = evaluate_test(c.theta);
                test_scores.push_back(test);
                f << rank++ << ',' << (c.source_iteration + 1) << ',' << (c.source_rank + 1)
                  << ',' << fmt6(c.train_score) << ',' << fmt6(c.validation_score)
                  << ',' << fmt6(test) << "\n";
            }
            f.close();
            copy_latest(final_test, result_dir_ / "final_test.csv");

            if (!candidates.empty()) {
                const auto& selected = candidates[0];
                rows.push_back(summary_row("validation_selected", selected.source_iteration + 1,
                                           selected.source_rank + 1, selected.train_score,
                                           selected.validation_score, test_scores[0],
                                           final_test.string()));
            }
        }

        if (!best_train_theta_.empty()) {
            fs::path diagnostic = result_dir_ / ("best_train_final_test_iter_" + padded(completed_iterations_, 3) + ".csv");
            double validation = evaluate_theta_parallel(best_train_theta_, validation_seeds_, threads_);
            double test = evaluate_test(best_train_theta_);
            std::ofstream d(diagnostic, std::ios::trunc);
            d << "source_iteration,source_rank,train_score,validation_episodes,validation_mean,test_episodes,test_mean\n";
            d << (best_train_iteration_ + 1) << ',' << (best_train_rank_ + 1) << ','
              << fmt6(best_train_score_) << ',' << validation_episodes_ << ','
              << fmt6(validation) << ',' << config_.test_episodes << ',' << fmt6(test) << "\n";
            d.close();
            copy_latest(diagnostic, result_dir_ / "best_train_final_test.csv");
            rows.push_back(summary_row("best_train_diagnostic", best_train_iteration_ + 1,
                                       best_train_rank_ + 1, best_train_score_,
                                       validation, test, diagnostic.string()));
        }
        if (is_robust()) write_early_stop_file();
        return rows;
    }

    std::string padded(int value, int width) const {
        std::ostringstream os;
        os << std::setw(width) << std::setfill('0') << value;
        return os.str();
    }

    SummaryRow summary_row(const std::string& selected_theta, int source_iteration,
                           int source_rank, double train_score, double validation_mean,
                           double test_mean, const std::string& artifact) const {
        SummaryRow row;
        row.experiment = experiment_name_;
        row.method = method_name(algorithm_);
        row.noise = noise_status(with_noise_);
        row.ratio = ratio_label_;
        row.selection = effective_selection_label();
        row.selected_theta = selected_theta;
        row.run_id = "run_" + std::to_string(run_id_);
        row.run_seed = SeedManager::run_seed(config_, run_id_);
        row.source_iteration = source_iteration;
        row.source_rank = source_rank;
        row.train_score = train_score;
        row.validation_mean = validation_mean;
        row.test_mean = test_mean;
        row.train_episodes = config_.train_episodes;
        row.validation_episodes = is_robust() ? config_.robust_validation_episodes : validation_episodes_;
        row.test_episodes = config_.test_episodes;
        row.completed_iterations = completed_iterations_;
        row.artifact_path = artifact;
        return row;
    }

    double evaluate_test(const std::vector<double>& theta) const {
        std::vector<int64_t> seeds(config_.test_episodes);
        for (int i = 0; i < config_.test_episodes; ++i) seeds[i] = SeedManager::test_seed(config_, i);
        return evaluate_theta_parallel(theta, seeds, threads_);
    }

    double calculate_noise(int iteration) const {
        if (!with_noise_) return 0.0;
        return std::max(config_.initial_noise - iteration / 10.0, 0.0);
    }

    double average_std() const {
        return std::accumulate(stds_.begin(), stds_.end(), 0.0) / stds_.size();
    }

    void maybe_save_checkpoint() {
        if (!config_.checkpoint_enabled || completed_iterations_ <= 0) return;
        if (completed_iterations_ >= config_.iterations
            || completed_iterations_ % config_.checkpoint_every_iterations == 0) {
            save_checkpoint();
        }
    }

    std::string checkpoint_fingerprint() const {
        std::ostringstream os;
        os << "experiment=" << experiment_name_ << "\n";
        os << "method=" << method_name(algorithm_) << "\n";
        os << "noise.status=" << noise_status(with_noise_) << "\n";
        os << "selection.mode=" << effective_selection_label() << "\n";
        os << "iteration.stride=" << SeedManager::ITERATION_STRIDE << "\n";
        os << "ratio=" << ratio_label_ << "\n";
        os << "run.id=" << run_id_ << "\n";
        os << "episode.protocol=" << config_.episode_protocol << "\n";
        os << "base.seed=" << config_.base_seed << "\n";
        os << "run.stride=" << config_.run_stride << "\n";
        os << "run.seed=" << SeedManager::run_seed(config_, run_id_) << "\n";
        os << "theta.seed=" << SeedManager::theta_seed(config_, run_id_) << "\n";
        os << "validation.scope=" << validation_scope() << "\n";
        os << "train.offset=" << SeedManager::TRAIN_OFFSET << "\n";
        os << "validation.offset=" << SeedManager::VALIDATION_OFFSET << "\n";
        os << "test.offset=" << SeedManager::TEST_OFFSET << "\n";
        os << "theta.offset=" << SeedManager::THETA_OFFSET << "\n";
        os << "seed.formula.run=base.seed + (run.id - 1) * run.stride\n";
        os << "seed.formula.train=run.seed + train.offset + iteration * iteration.stride; JavaRandom.nextLong stream\n";
        os << "seed.formula.test=base.seed + test.offset + episode.index * iteration.stride\n";
        if (is_robust()) {
            os << "robust.validation.offset=" << SeedManager::ROBUST_VALIDATION_OFFSET << "\n";
            os << "robust.deep.offset=" << SeedManager::ROBUST_DEEP_OFFSET << "\n";
            os << "robust.behavior.offset=" << SeedManager::ROBUST_BEHAVIOR_OFFSET << "\n";
            os << "robust.candidate.offset=" << SeedManager::ROBUST_CANDIDATE_OFFSET << "\n";
            os << "robust.score.formula=" << robust_score_formula(algorithm_) << "\n";
            os << "robust.previous.val.enabled=" << previous_val_candidates_enabled() << "\n";
            os << "seed.formula.robust.validation=run.seed + robust.validation.offset; JavaRandom.nextLong stream\n";
            os << "seed.formula.robust.deep=run.seed + robust.deep.offset; JavaRandom.nextLong stream\n";
            os << "seed.formula.robust.behavior=run.seed + robust.behavior.offset\n";
            os << "seed.formula.robust.candidate=run.seed + robust.candidate.offset + iteration * iteration.stride\n";
        }
        os << "population=" << config_.population << "\n";
        os << "elite.ratio=" << config_.elite_ratio << "\n";
        os << "train.episodes=" << config_.train_episodes << "\n";
        os << "validation.episodes=" << validation_episodes_ << "\n";
        os << "validation.top.k=" << config_.validation_top_k << "\n";
        os << "global.top.n=" << config_.global_top_n << "\n";
        os << "test.episodes=" << config_.test_episodes << "\n";
        os << "initial.noise=" << config_.initial_noise << "\n";
        os << "proportional.max.weight.share=" << config_.proportional_max_weight_share << "\n";
        os << "pool.size.ratio=" << config_.pool_size_ratio << "\n";
        os << "pool.elite.weight.share=" << config_.pool_elite_weight_share << "\n";
        os << "pool.history.weight.share=" << config_.pool_history_weight_share << "\n";
        os << "diversity.lambda=" << config_.diversity_lambda << "\n";
        os << "robust.validation.candidate.train.top.k=" << config_.robust_candidate_train_top_k << "\n";
        os << "robust.validation.candidate.history.top.k=" << config_.robust_candidate_history_top_k << "\n";
        os << "robust.validation.candidate.previous.val.top.k=" << config_.robust_candidate_previous_val_top_k << "\n";
        os << "robust.validation.candidate.random.k=" << config_.robust_candidate_random_k << "\n";
        os << "robust.validation.episodes=" << config_.robust_validation_episodes << "\n";
        os << "robust.validation.std.penalty=" << config_.robust_validation_std_penalty << "\n";
        os << "robust.train.score.weight=" << config_.robust_train_score_weight << "\n";
        os << "robust.diversity.weight=" << config_.robust_diversity_weight << "\n";
        os << "robust.deep.every=" << config_.robust_deep_every << "\n";
        os << "robust.deep.episodes=" << config_.robust_deep_episodes << "\n";
        os << "behavior.probe.states=" << config_.behavior_probe_states << "\n";
        os << "robust.pool.size=" << config_.robust_pool_size << "\n";
        os << "robust.pool.current.share.early=" << config_.robust_pool_current_share_early << "\n";
        os << "robust.pool.history.share.early=" << config_.robust_pool_history_share_early << "\n";
        os << "robust.pool.current.share.late=" << config_.robust_pool_current_share_late << "\n";
        os << "robust.pool.history.share.late=" << config_.robust_pool_history_share_late << "\n";
        os << "robust.pool.late.after.iteration=" << config_.robust_pool_late_after_iteration << "\n";
        os << "robust.final.top.n=" << config_.robust_final_top_n << "\n";
        os << "early.stop.enabled=" << config_.early_stop_enabled << "\n";
        os << "early.stop.min.iterations=" << config_.early_stop_min_iterations << "\n";
        os << "early.stop.patience.deep.checks=" << config_.early_stop_patience_deep_checks << "\n";
        os << "early.stop.min.delta=" << config_.early_stop_min_delta << "\n";
        os << "early.stop.min.avg.std=" << config_.early_stop_min_avg_std << "\n";
        os << "row.num=" << config_.row_num << "\n";
        os << "parameter.num=" << kParameterNum << "\n";
        return os.str();
    }

    void save_checkpoint() {
        fs::path tmp = result_dir_ / (std::string(kCheckpointStateFile) + ".tmp");
        fs::path state = result_dir_ / kCheckpointStateFile;
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << std::setprecision(17);
            f << "version=" << kCheckpointVersion << "\n";
            f << "completedIterations=" << completed_iterations_ << "\n";
            f << "bestTrainScore=" << best_train_score_ << "\n";
            f << "bestTrainIteration=" << best_train_iteration_ << "\n";
            f << "bestTrainRank=" << best_train_rank_ << "\n";
            f << "thetaRandom=" << theta_rng_.serialize() << "\n";
            f << "means=" << join_doubles(means_) << "\n";
            f << "stds=" << join_doubles(stds_) << "\n";
            f << "bestTrainTheta=" << join_doubles(best_train_theta_) << "\n";
            f << "bestDeepValidation=" << best_deep_validation_ << "\n";
            f << "bestDeepIteration=" << best_deep_iteration_ << "\n";
            f << "staleDeepChecks=" << stale_deep_checks_ << "\n";
            f << "earlyStopTriggered=" << (early_stop_triggered_ ? 1 : 0) << "\n";
            f << "earlyStopIteration=" << early_stop_iteration_ << "\n";
            f << "earlyStopReason=" << early_stop_reason_ << "\n";
            f << "[validation_pool]\n";
            for (const auto& e : validation_pool_) {
                f << e.source_iteration << ',' << e.source_rank << ',' << e.train_score
                  << ',' << e.validation_score << ',' << e.validation_std
                  << ',' << e.robust_score << ',' << e.behavior_distance
                  << ',' << join_doubles(e.theta) << "\n";
            }
            f << "[global_train_pool]\n";
            for (const auto& e : global_train_pool_) {
                f << e.source_iteration << ',' << e.population_index << ',' << e.train_score
                  << ',' << join_doubles(e.theta) << "\n";
            }
            f << "[history_pool]\n";
            for (const auto& e : history_pool_) {
                f << e.selection_score << ',' << e.train_score << ',' << e.validation_score
                  << ',' << e.validation_std << ',' << e.robust_score
                  << ',' << e.min_distance << ',' << e.source_iteration
                  << ',' << e.source_rank << ',' << join_doubles(e.theta) << "\n";
            }
        }
        atomic_replace(tmp, state);

        std::string fp = checkpoint_fingerprint();
        {
            std::ofstream f(result_dir_ / kCheckpointFingerprintFile, std::ios::trunc);
            f << fp;
        }
        {
            std::ofstream f(result_dir_ / kCheckpointManifestFile, std::ios::trunc);
            f << "version=" << kCheckpointVersion << "\n";
            f << "completed.iterations=" << completed_iterations_ << "\n";
            f << "experiment=" << experiment_name_ << "\n";
            f << "method=" << method_name(algorithm_) << "\n";
            f << "noise.status=" << noise_status(with_noise_) << "\n";
            f << "selection.mode=" << effective_selection_label() << "\n";
            f << "ratio=" << ratio_label_ << "\n";
            f << "run.id=" << run_id_ << "\n";
            f << "fingerprint.hash=" << fnv1a64(fp) << "\n";
        }
        log_.info("[" + label() + "] checkpoint saved completedIterations=" + std::to_string(completed_iterations_));
    }

    void load_checkpoint() {
        fs::path state = result_dir_ / kCheckpointStateFile;
        fs::path fp_file = result_dir_ / kCheckpointFingerprintFile;
        if (!fs::exists(state)) throw std::runtime_error("checkpoint state not found: " + state.string());
        if (!fs::exists(fp_file)) throw std::runtime_error("checkpoint fingerprint not found: " + fp_file.string());
        {
            std::ifstream f(fp_file);
            std::stringstream buf;
            buf << f.rdbuf();
            std::string got = buf.str();
            std::string expected = checkpoint_fingerprint();
            if (got != expected) throw std::runtime_error("checkpoint fingerprint mismatch for " + result_dir_.string());
        }

        validation_pool_.clear();
        global_train_pool_.clear();
        history_pool_.clear();
        std::ifstream f(state);
        std::string section;
        std::string line;
        while (std::getline(f, line)) {
            line = trim(line);
            if (line.empty()) continue;
            if (line.front() == '[' && line.back() == ']') {
                section = line.substr(1, line.size() - 2);
                continue;
            }
            if (section.empty()) {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq);
                std::string val = line.substr(eq + 1);
                if (key == "version" && val != kCheckpointVersion) throw std::runtime_error("checkpoint version mismatch");
                else if (key == "completedIterations") completed_iterations_ = std::stoi(val);
                else if (key == "bestTrainScore") best_train_score_ = std::stod(val);
                else if (key == "bestTrainIteration") best_train_iteration_ = std::stoi(val);
                else if (key == "bestTrainRank") best_train_rank_ = std::stoi(val);
                else if (key == "thetaRandom") theta_rng_.deserialize(val);
                else if (key == "means") means_ = parse_doubles(val);
                else if (key == "stds") stds_ = parse_doubles(val);
                else if (key == "bestTrainTheta") best_train_theta_ = parse_doubles(val);
                else if (key == "bestDeepValidation") best_deep_validation_ = std::stod(val);
                else if (key == "bestDeepIteration") best_deep_iteration_ = std::stoi(val);
                else if (key == "staleDeepChecks") stale_deep_checks_ = std::stoi(val);
                else if (key == "earlyStopTriggered") early_stop_triggered_ = std::stoi(val) != 0;
                else if (key == "earlyStopIteration") early_stop_iteration_ = std::stoi(val);
                else if (key == "earlyStopReason") early_stop_reason_ = val;
            } else if (section == "validation_pool") {
                auto t = split_csv_simple(line);
                if (t.size() >= 7 + kParameterNum) {
                    std::vector<double> theta;
                    for (size_t i = 7; i < 7 + kParameterNum; ++i) theta.push_back(std::stod(t[i]));
                    ScoredTheta entry;
                    entry.source_iteration = std::stoi(t[0]);
                    entry.source_rank = std::stoi(t[1]);
                    entry.train_score = std::stod(t[2]);
                    entry.validation_score = std::stod(t[3]);
                    entry.validation_std = std::stod(t[4]);
                    entry.robust_score = std::stod(t[5]);
                    entry.behavior_distance = std::stod(t[6]);
                    entry.theta = theta;
                    if (is_robust()) entry.behavior_signature = behavior_signature(theta);
                    validation_pool_.push_back(entry);
                }
            } else if (section == "global_train_pool") {
                auto t = split_csv_simple(line);
                if (t.size() >= 4) {
                    std::vector<double> theta;
                    for (size_t i = 3; i < t.size(); ++i) theta.push_back(std::stod(t[i]));
                    global_train_pool_.push_back({std::stoi(t[0]), std::stoi(t[1]), std::stod(t[2]), theta});
                }
            } else if (section == "history_pool") {
                auto t = split_csv_simple(line);
                if (t.size() >= 8 + kParameterNum) {
                    std::vector<double> theta;
                    for (size_t i = 8; i < 8 + kParameterNum; ++i) theta.push_back(std::stod(t[i]));
                    PoolEntry entry;
                    entry.selection_score = std::stod(t[0]);
                    entry.train_score = std::stod(t[1]);
                    entry.validation_score = std::stod(t[2]);
                    entry.validation_std = std::stod(t[3]);
                    entry.robust_score = std::stod(t[4]);
                    entry.min_distance = std::stod(t[5]);
                    entry.source_iteration = std::stoi(t[6]);
                    entry.source_rank = std::stoi(t[7]);
                    entry.theta = theta;
                    if (is_robust()) entry.behavior_signature = behavior_signature(theta);
                    history_pool_.push_back(entry);
                }
            }
        }
        if (static_cast<int>(means_.size()) != kParameterNum || static_cast<int>(stds_.size()) != kParameterNum) {
            throw std::runtime_error("checkpoint parameter vector length mismatch");
        }
        if (completed_iterations_ > config_.iterations) {
            throw std::runtime_error("checkpoint completed more iterations than requested target");
        }
        write_global_top_weights();
        write_validation_pool();
        if (uses_pool(algorithm_)) write_history_pool();
        if (is_robust()) {
            write_behavior_pool();
            write_early_stop_file();
        }
        if (!best_train_theta_.empty()) write_theta(best_train_theta_, result_dir_ / "best_train_theta.txt");
        auto best_val = best_validation_theta();
        if (best_val) write_theta(best_val->theta, result_dir_ / "best_validation_theta.txt");
        log_.info("[" + label() + "] checkpoint loaded completedIterations=" + std::to_string(completed_iterations_));
    }

    const Config& config_;
    std::string experiment_name_;
    Algorithm algorithm_;
    SelectionMode selection_mode_;
    std::string ratio_label_;
    int validation_episodes_;
    int run_id_;
    bool with_noise_;
    fs::path result_dir_;
    Log& log_;
    int elite_num_;
    int threads_;
    std::vector<double> scores_;
    std::vector<std::vector<double>> thetas_;
    std::vector<double> means_;
    std::vector<double> stds_;
    JavaRandom theta_rng_;
    std::vector<int64_t> validation_seeds_;
    std::vector<int64_t> robust_validation_seeds_;
    std::vector<int64_t> robust_deep_seeds_;
    std::vector<State> behavior_probe_states_;
    std::vector<ScoredTheta> validation_pool_;
    std::vector<TrainTheta> global_train_pool_;
    std::vector<PoolEntry> history_pool_;
    double best_train_score_ = -std::numeric_limits<double>::infinity();
    std::vector<double> best_train_theta_;
    int best_train_iteration_ = -1;
    int best_train_rank_ = -1;
    double best_deep_validation_ = -std::numeric_limits<double>::infinity();
    int best_deep_iteration_ = -1;
    int stale_deep_checks_ = 0;
    bool early_stop_triggered_ = false;
    int early_stop_iteration_ = -1;
    std::string early_stop_reason_ = "not_triggered";
    int completed_iterations_ = 0;
};

void write_manifest(const fs::path& batch_dir, const std::string& name,
                    const Config& c, const std::map<std::string, std::string>& extra) {
    fs::path file = batch_dir / (c.resume_enabled
        ? "experiment_manifest_resume_" + timestamp("%Y%m%d_%H%M%S") + ".txt"
        : "experiment_manifest.txt");
    std::ofstream w(file, std::ios::trunc);
    w << "experiment.name=" << name << "\n";
    w << "created.at=" << timestamp("%Y-%m-%d %H:%M:%S") << "\n";
    w << "episode.protocol=" << c.episode_protocol << "\n";
    w << "base.seed=" << c.base_seed << "\n";
    w << "run.stride=" << c.run_stride << "\n";
    w << "runs=" << c.runs << "\n";
    w << "run.start=" << c.run_start << "\n";
    w << "run.end=" << c.run_end << "\n";
    w << "population=" << c.population << "\n";
    w << "elite.ratio=" << c.elite_ratio << "\n";
    w << "iterations=" << c.iterations << "\n";
    w << "checkpoint.enabled=" << c.checkpoint_enabled << "\n";
    w << "checkpoint.every.iterations=" << c.checkpoint_every_iterations << "\n";
    w << "resume.enabled=" << c.resume_enabled << "\n";
    w << "resume.batch.dir=" << c.resume_batch_dir << "\n";
    w << "train.episodes=" << c.train_episodes << "\n";
    w << "validation.episodes=" << c.validation_episodes << "\n";
    w << "validation.top.k=" << c.validation_top_k << "\n";
    w << "global.top.n=" << c.global_top_n << "\n";
    w << "test.episodes=" << c.test_episodes << "\n";
    w << "pool.size.ratio=" << c.pool_size_ratio << "\n";
    w << "pool.elite.weight.share=" << c.pool_elite_weight_share << "\n";
    w << "pool.history.weight.share=" << c.pool_history_weight_share << "\n";
    w << "diversity.lambda=" << c.diversity_lambda << "\n";
    w << "robust.validation.candidate.train.top.k=" << c.robust_candidate_train_top_k << "\n";
    w << "robust.validation.candidate.history.top.k=" << c.robust_candidate_history_top_k << "\n";
    w << "robust.validation.candidate.previous.val.top.k=" << c.robust_candidate_previous_val_top_k << "\n";
    w << "robust.validation.candidate.random.k=" << c.robust_candidate_random_k << "\n";
    w << "robust.validation.episodes=" << c.robust_validation_episodes << "\n";
    w << "robust.validation.std.penalty=" << c.robust_validation_std_penalty << "\n";
    w << "robust.train.score.weight=" << c.robust_train_score_weight << "\n";
    w << "robust.diversity.weight=" << c.robust_diversity_weight << "\n";
    w << "robust.deep.every=" << c.robust_deep_every << "\n";
    w << "robust.deep.episodes=" << c.robust_deep_episodes << "\n";
    w << "behavior.probe.states=" << c.behavior_probe_states << "\n";
    w << "robust.pool.size=" << c.robust_pool_size << "\n";
    w << "robust.pool.current.share.early=" << c.robust_pool_current_share_early << "\n";
    w << "robust.pool.history.share.early=" << c.robust_pool_history_share_early << "\n";
    w << "robust.pool.current.share.late=" << c.robust_pool_current_share_late << "\n";
    w << "robust.pool.history.share.late=" << c.robust_pool_history_share_late << "\n";
    w << "robust.pool.late.after.iteration=" << c.robust_pool_late_after_iteration << "\n";
    w << "robust.final.top.n=" << c.robust_final_top_n << "\n";
    w << "early.stop.enabled=" << c.early_stop_enabled << "\n";
    w << "early.stop.min.iterations=" << c.early_stop_min_iterations << "\n";
    w << "early.stop.patience.deep.checks=" << c.early_stop_patience_deep_checks << "\n";
    w << "early.stop.min.delta=" << c.early_stop_min_delta << "\n";
    w << "early.stop.min.avg.std=" << c.early_stop_min_avg_std << "\n";
    w << "seed.offset.train=" << SeedManager::TRAIN_OFFSET << "\n";
    w << "seed.offset.validation=" << SeedManager::VALIDATION_OFFSET << "\n";
    w << "seed.offset.test=" << SeedManager::TEST_OFFSET << "\n";
    w << "seed.offset.theta=" << SeedManager::THETA_OFFSET << "\n";
    w << "seed.offset.robust.validation=" << SeedManager::ROBUST_VALIDATION_OFFSET << "\n";
    w << "seed.offset.robust.deep=" << SeedManager::ROBUST_DEEP_OFFSET << "\n";
    w << "seed.offset.robust.behavior=" << SeedManager::ROBUST_BEHAVIOR_OFFSET << "\n";
    w << "seed.offset.robust.candidate=" << SeedManager::ROBUST_CANDIDATE_OFFSET << "\n";
    w << "seed.iteration.stride=" << SeedManager::ITERATION_STRIDE << "\n";
    w << "seed.formula.run=base.seed + (run.id - 1) * run.stride\n";
    w << "seed.formula.train=run.seed + seed.offset.train + iteration * seed.iteration.stride; JavaRandom.nextLong stream\n";
    w << "seed.formula.test=base.seed + seed.offset.test + episode.index * seed.iteration.stride\n";
    w << "seed.formula.robust.validation=run.seed + seed.offset.robust.validation; JavaRandom.nextLong stream\n";
    w << "seed.formula.robust.deep=run.seed + seed.offset.robust.deep; JavaRandom.nextLong stream\n";
    w << "seed.formula.robust.behavior=run.seed + seed.offset.robust.behavior\n";
    w << "seed.formula.robust.candidate=run.seed + seed.offset.robust.candidate + iteration * seed.iteration.stride\n";
    w << "proportional.max.weight.share=" << c.proportional_max_weight_share << "\n";
    w << "initial.noise=" << c.initial_noise << "\n";
    w << "thread.reserve=" << c.thread_reserve << "\n";
    w << "row.num=" << c.row_num << "\n";
    w << "benchmark.methods=" << c.benchmark_methods << "\n";
    w << "robust.ablation.methods=" << c.robust_ablation_methods << "\n";
    w << "benchmark.noise.levels=" << c.benchmark_noise_levels << "\n";
    w << "ratio.methods=" << c.ratio_methods << "\n";
    w << "ratio.selection.modes=" << c.ratio_selection_modes << "\n";
    for (const auto& kv : extra) w << kv.first << "=" << kv.second << "\n";
}

fs::path resolve_batch_dir(const Config& c, const std::string& prefix) {
    if (c.resume_enabled) {
        if (c.resume_batch_dir.empty()) throw std::runtime_error("resume.enabled=true requires resume.batch.dir");
        fs::path p(c.resume_batch_dir);
        if (!fs::is_directory(p)) throw std::runtime_error("resume.batch.dir is not a directory: " + p.string());
        return p;
    }
    fs::path p = fs::path(c.results_root) / (prefix + "_" + timestamp("%Y%m%d_%H%M%S"));
    fs::create_directories(p);
    return p;
}

void append_summary(const fs::path& file, const std::vector<SummaryRow>& rows) {
    bool header = !fs::exists(file);
    std::ofstream f(file, std::ios::app);
    if (header) f << SummaryRow::header() << "\n";
    for (const auto& row : rows) f << row.to_csv() << "\n";
}

void write_method_aggregate_summary(const fs::path& file, const std::vector<SummaryRow>& rows) {
    struct Stats {
        std::vector<double> values;
    };
    std::map<std::pair<std::string, std::string>, Stats> grouped;
    for (const auto& row : rows) {
        grouped[{row.method, row.selected_theta}].values.push_back(row.test_mean);
    }

    std::ofstream f(file, std::ios::trunc);
    f << "method,selected_theta,n,mean_test,std_test,min_test,max_test,ci95_half_width\n";
    for (const auto& kv : grouped) {
        const auto& method = kv.first.first;
        const auto& selected = kv.first.second;
        const auto& values = kv.second.values;
        if (values.empty()) continue;
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        double mean = sum / values.size();
        double sq = 0.0;
        for (double x : values) sq += (x - mean) * (x - mean);
        double stddev = values.size() > 1 ? std::sqrt(sq / (values.size() - 1)) : 0.0;
        auto mm = std::minmax_element(values.begin(), values.end());
        double ci95 = values.size() > 1 ? 1.96 * stddev / std::sqrt(static_cast<double>(values.size())) : 0.0;
        f << csv_escape(method) << ',' << csv_escape(selected) << ',' << values.size()
          << ',' << fmt6(mean) << ',' << fmt6(stddev)
          << ',' << fmt6(*mm.first) << ',' << fmt6(*mm.second)
          << ',' << fmt6(ci95) << "\n";
    }
}

std::string join_algorithms(const std::vector<Algorithm>& algorithms) {
    std::ostringstream os;
    for (size_t i = 0; i < algorithms.size(); ++i) {
        if (i) os << ',';
        os << method_name(algorithms[i]);
    }
    return os.str();
}

int run_benchmark(const Config& config, Log& log, const fs::path& batch_dir) {
    std::vector<Algorithm> algorithms = parse_algorithms(config.benchmark_methods, log, false);
    std::vector<bool> noises = parse_noise_levels(config.benchmark_noise_levels);
    std::map<std::string, std::string> extra = {
        {"experiment.type", "ce_full_method_benchmark"},
        {"methods", join_algorithms(algorithms)},
        {"noise.levels", config.benchmark_noise_levels},
        {"selection.mode", "train_elite; RobustPoolDiversityCE=robust_validation"},
        {"validation.seed.scope", "benchmark"},
        {"execution.order", "run -> noise -> method"},
    };
    write_manifest(batch_dir, "CE final full-method benchmark", config, extra);

    fs::path summary = batch_dir / "comparison_summary.csv";
    int planned = static_cast<int>(algorithms.size() * noises.size() * config.run_count());
    int done = 0;
    log.info("Final benchmark batch: " + batch_dir.string() + " plannedTrainers=" + std::to_string(planned));
    for (int run = config.run_start; run <= config.run_end; ++run) {
        for (bool noise : noises) {
            for (Algorithm algorithm : algorithms) {
                ++done;
                fs::path run_dir = batch_dir / method_name(algorithm) / noise_status(noise) / ("run_" + std::to_string(run));
                log.info("Starting trainer " + std::to_string(done) + "/" + std::to_string(planned)
                         + ": method=" + method_name(algorithm) + " noise=" + noise_status(noise)
                         + " run_" + std::to_string(run));
                Trainer trainer(config, "benchmark", algorithm, SelectionMode::TrainElite, "benchmark",
                                config.validation_episodes, run, noise, run_dir, log);
                append_summary(summary, trainer.train());
            }
        }
    }
    log.info("Benchmark batch completed: " + batch_dir.string());
    return 0;
}

int run_robust_ablation(const Config& config, Log& log, const fs::path& batch_dir) {
    std::vector<Algorithm> algorithms = parse_algorithms(config.robust_ablation_methods, log, false);
    for (Algorithm algorithm : algorithms) {
        if (!is_robust_ablation_method(algorithm)) {
            throw std::runtime_error("robust-ablation accepts only RobustValidation*DiversityCE methods: "
                                     + method_name(algorithm));
        }
    }
    std::vector<bool> noises = parse_noise_levels(config.benchmark_noise_levels);
    std::map<std::string, std::string> extra = {
        {"experiment.type", "robust_component_ablation"},
        {"reference.batch", "benchmark_batch_20260520_060507_all"},
        {"methods", join_algorithms(algorithms)},
        {"noise.levels", config.benchmark_noise_levels},
        {"selection.mode", "robust_component_ablation"},
        {"shared.history.pool", "enabled"},
        {"shared.history.pool.size", std::to_string(config.robust_history_pool_size())},
        {"shared.distribution.update", "current_elite_plus_history_pool"},
        {"shared.current.history.share.early", fmt6(config.robust_pool_current_share_early) + "/" + fmt6(config.robust_pool_history_share_early)},
        {"shared.current.history.share.late", fmt6(config.robust_pool_current_share_late) + "/" + fmt6(config.robust_pool_history_share_late)},
        {"score.formula.RobustValidationRerankDiversityCE", "validation_mean"},
        {"score.formula.RobustValidationStdDiversityCE", "validation_mean - std_penalty * validation_std"},
        {"score.formula.RobustValidationMemoryDiversityCE", "validation_mean - std_penalty * validation_std"},
        {"candidate.sources.common", "train_top,history_top,random"},
        {"candidate.previous_val_top.RobustValidationRerankDiversityCE", "0"},
        {"candidate.previous_val_top.RobustValidationStdDiversityCE", "0"},
        {"candidate.previous_val_top.RobustValidationMemoryDiversityCE", std::to_string(config.robust_candidate_previous_val_top_k)},
        {"train.behavior.components", "diagnostic_only"},
        {"execution.order", "run -> noise -> method"},
    };
    write_manifest(batch_dir, "Robust CE component ablation", config, extra);

    fs::path summary = batch_dir / "comparison_summary.csv";
    std::vector<SummaryRow> aggregate_rows;
    int planned = static_cast<int>(algorithms.size() * noises.size() * config.run_count());
    int done = 0;
    log.info("Robust component ablation batch: " + batch_dir.string()
             + " plannedTrainers=" + std::to_string(planned));
    for (int run = config.run_start; run <= config.run_end; ++run) {
        for (bool noise : noises) {
            for (Algorithm algorithm : algorithms) {
                ++done;
                fs::path run_dir = batch_dir / method_name(algorithm) / noise_status(noise) / ("run_" + std::to_string(run));
                log.info("Starting trainer " + std::to_string(done) + "/" + std::to_string(planned)
                         + ": method=" + method_name(algorithm)
                         + " noise=" + noise_status(noise)
                         + " run_" + std::to_string(run));
                Trainer trainer(config, "robust_ablation", algorithm, SelectionMode::TrainElite,
                                "robust_ablation", config.validation_episodes, run, noise, run_dir, log);
                auto rows = trainer.train();
                append_summary(summary, rows);
                aggregate_rows.insert(aggregate_rows.end(), rows.begin(), rows.end());
            }
        }
    }
    write_method_aggregate_summary(batch_dir / "method_aggregate_summary.csv", aggregate_rows);
    log.info("Robust component ablation batch completed: " + batch_dir.string());
    return 0;
}

int run_ratio(const Config& config, Log& log, const fs::path& batch_dir) {
    std::vector<Algorithm> algorithms = parse_algorithms(config.ratio_methods, log, true);
    std::vector<SelectionMode> selection_modes = parse_selection_modes(config.ratio_selection_modes);
    std::map<std::string, std::string> extra = {
        {"experiment.type", "pool_ratio_selection_ablation"},
        {"methods", join_algorithms(algorithms)},
        {"noise.levels", "noise_off"},
        {"selection.modes", config.ratio_selection_modes},
        {"validation.count.formula", "round(train.episodes * (1-a) / a)"},
        {"execution.order", "run -> ratio -> selection -> method"},
    };
    write_manifest(batch_dir, "CE final ratio and val-elite ablation", config, extra);

    fs::path summary = batch_dir / "comparison_summary.csv";
    int planned = static_cast<int>(algorithms.size() * selection_modes.size() * config.ratios.size() * config.run_count());
    int done = 0;
    log.info("Final ratio ablation batch: " + batch_dir.string() + " plannedTrainers=" + std::to_string(planned));
    for (int run = config.run_start; run <= config.run_end; ++run) {
        for (double ratio : config.ratios) {
            std::string label = ratio_label(ratio);
            int validation_episodes = std::max(1, static_cast<int>(std::llround(config.train_episodes * (1.0 - ratio) / ratio)));
            for (SelectionMode selection : selection_modes) {
                for (Algorithm algorithm : algorithms) {
                    ++done;
                    fs::path run_dir = batch_dir / method_name(algorithm) / label / selection_label(selection) / ("run_" + std::to_string(run));
                    log.info("Starting trainer " + std::to_string(done) + "/" + std::to_string(planned)
                             + ": method=" + method_name(algorithm)
                             + " ratio=" + label
                             + " selection=" + selection_label(selection)
                             + " run_" + std::to_string(run));
                    Trainer trainer(config, "ratio_elite", algorithm, selection, label,
                                    validation_episodes, run, false, run_dir, log);
                    append_summary(summary, trainer.train());
                }
            }
        }
    }
    log.info("Ratio ablation batch completed: " + batch_dir.string());
    return 0;
}

void usage() {
    std::cerr
        << "usage:\n"
        << "  tetris_core ce-final benchmark [--config PATH] [--key=value ...]\n"
        << "  tetris_core ce-final robust-ablation [--config PATH] [--key=value ...]\n"
        << "  tetris_core ce-final ratio [--config PATH] [--key=value ...]\n";
}

} // namespace

int run(int argc, char** argv) {
    if (argc < 1) {
        usage();
        return 2;
    }
    std::string mode = argv[0];
    std::string config_path = kDefaultConfigPath;
    std::map<std::string, std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                std::cerr << "missing value after --config\n";
                return 2;
            }
            config_path = argv[++i];
        } else if (starts_with(arg, "--config=")) {
            config_path = arg.substr(std::string("--config=").size());
        } else if (starts_with(arg, "--") && arg.find('=') != std::string::npos) {
            auto eq = arg.find('=');
            overrides[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
        } else {
            std::cerr << "unknown ce-final arg: " << arg << "\n";
            return 2;
        }
    }

    try {
        if (mode != "benchmark" && mode != "ratio" && mode != "robust-ablation") {
            usage();
            return 2;
        }
        Config config = load_config(config_path, overrides);
        set_row_num(config.row_num);
        init_action_table();

        std::string batch_prefix = mode == "ratio" ? "ratio_elite_batch"
            : mode == "robust-ablation" ? "robust_ablation_batch"
            : "benchmark_batch";
        fs::path batch_dir = resolve_batch_dir(config, batch_prefix);
        Log log;
        log.open(batch_dir / "run.log");
        log.info("Loaded final CE config: " + config_path);

        if (mode == "benchmark") return run_benchmark(config, log, batch_dir);
        if (mode == "robust-ablation") return run_robust_ablation(config, log, batch_dir);
        if (mode == "ratio") return run_ratio(config, log, batch_dir);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[ce-final] ERROR: " << e.what() << "\n";
        return 1;
    }
}

} // namespace tcore::ce_final
