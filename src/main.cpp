// tetris_core: command-line entry point for the linear afterstate Tetris core
// used in "Reducing Elite-Selection Bias in Cross-Entropy Training for Tetris".
//
// Subcommands:
//   bench greedy --row-num N --workers W --episodes E --theta FILE --out DIR
//        [--max-steps N] [--seed-offset N]
//       Runs the greedy one-ply afterstate policy under the nine signed CBMPI
//       features and writes per-episode and summary CSV/TSV files.
//
//   ce-final (benchmark|robust-ablation|ratio) [--config PATH] [--key=value ...]
//       Runs the robust-validation cross-entropy training / ablation driver.
//
//   help
//       Print usage.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "features.h"
#include "pieces.h"
#include "tetris_state.h"
#include "value_function.h"
#include "ce_final/ce_final.h"

using namespace tcore;

namespace {

struct Args {
    std::string mode = "help";
    int row_num   = 10;
    int workers   = 1;
    int episodes  = 100;
    std::string theta;
    std::string out;
    int max_steps = 0;
    int seed_offset = 0;
};

void usage() {
    std::cerr <<
        "usage:\n"
        "  tetris_core bench greedy --row-num N --workers W --episodes E \\\n"
        "      --theta FILE --out DIR [--max-steps N] [--seed-offset N]\n"
        "  tetris_core ce-final (benchmark|robust-ablation|ratio) [--config PATH] [--key=value ...]\n"
        "  tetris_core help\n";
}

struct EpStats {
    int lines = 0;
    int steps = 0;
    int decisions = 0;
};

EpStats play_episode(const ValueFunction& vf, uint64_t env_seed, int max_steps) {
    EpStats st;
    std::mt19937_64 env_rng(env_seed);
    State s = initial_state(static_cast<int>(env_rng() % 7));
    while (true) {
        uint64_t mask = action_list(s);
        if (mask == 0) break;
        int n = action_count(s.piece);
        double best_v = -1e300;
        int best_a = -1;
        for (int a = 0; a < n; ++a) {
            if (!action_valid(a, mask)) continue;
            double base[kFeatureDim];
            compute_features(s, a, base);
            double v = vf.value_from_features(base);
            if (v > best_v) { best_v = v; best_a = a; }
        }
        if (best_a < 0) break;
        s = after_state(s, best_a);
        st.lines += s.reward;
        s.score += s.reward;
        s.piece  = static_cast<int>(env_rng() % 7);
        st.steps++;
        st.decisions++;
        if (max_steps > 0 && st.steps >= max_steps) break;
        if (is_final(s)) break;
    }
    return st;
}

int cmd_bench(const Args& a) {
    set_row_num(a.row_num);
    init_action_table();

    ValueFunction vf;
    if (!vf.load(a.theta)) {
        std::cerr << "cannot load theta " << a.theta << "\n";
        return 2;
    }

    std::filesystem::create_directories(a.out);

    {
        std::ofstream f(a.out + "/config.txt");
        f << "mode=greedy\nrowNum=" << a.row_num
          << "\nworkers=" << a.workers << "\nepisodes=" << a.episodes
          << "\nseed_offset=" << a.seed_offset
          << "\ntheta=" << a.theta
          << "\nmax_steps=" << a.max_steps << "\n";
    }

    std::ofstream epCsv(a.out + "/episodes.csv");
    epCsv << "episode_id,worker,lines,steps,decisions,elapsed_ms\n";
    std::mutex csv_lock;

    std::atomic<int> next_ep(0);
    std::atomic<uint64_t> sum_lines(0), sum_steps(0), sum_decisions(0);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int w = 0; w < a.workers; ++w) {
        workers.emplace_back([&, w]() {
            while (true) {
                int ep = next_ep.fetch_add(1);
                if (ep >= a.episodes) break;
                uint64_t seed_ep = static_cast<uint64_t>(a.seed_offset + ep);
                uint64_t env_seed = 0xC0FFEE12345678ULL + seed_ep * 0x9E3779B97F4A7C15ULL;
                auto e0 = std::chrono::steady_clock::now();
                EpStats st = play_episode(vf, env_seed, a.max_steps);
                auto e1 = std::chrono::steady_clock::now();
                long ms = std::chrono::duration_cast<std::chrono::milliseconds>(e1 - e0).count();
                sum_lines     += st.lines;
                sum_steps     += st.steps;
                sum_decisions += st.decisions;
                {
                    std::lock_guard<std::mutex> g(csv_lock);
                    epCsv << ep << "," << w << "," << st.lines << ","
                          << st.steps << "," << st.decisions << "," << ms << "\n";
                }
            }
        });
    }
    for (auto& t : workers) t.join();
    auto t1 = std::chrono::steady_clock::now();
    epCsv.flush(); epCsv.close();

    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double secs = ms / 1000.0;
    double avg_lines = static_cast<double>(sum_lines) / a.episodes;

    {
        std::ofstream f(a.out + "/summary.tsv");
        f << "metric\tvalue\n";
        f << "mode\tgreedy\n";
        f << "rowNum\t" << a.row_num << "\n";
        f << "workers\t" << a.workers << "\n";
        f << "episodes\t" << a.episodes << "\n";
        f << "seed_offset\t" << a.seed_offset << "\n";
        f << "avg_lines\t" << avg_lines << "\n";
        f << "total_steps\t" << sum_steps.load() << "\n";
        f << "total_decisions\t" << sum_decisions.load() << "\n";
        f << "elapsed_ms\t" << ms << "\n";
    }

    std::cerr << "=== bench greedy done ===\n";
    std::cerr << "episodes=" << a.episodes << " workers=" << a.workers
              << " rowNum=" << a.row_num << "\n";
    std::cerr << "avg_lines=" << avg_lines << "  elapsed=" << secs << "s\n";
    return 0;
}

bool consume_flag(int& i, int argc, char** argv, const char* name, std::string& out) {
    if (std::strcmp(argv[i], name) != 0) return false;
    if (i + 1 >= argc) { std::cerr << "missing value after " << name << "\n"; std::exit(2); }
    out = argv[++i];
    return true;
}
bool consume_int(int& i, int argc, char** argv, const char* name, int& out) {
    std::string s;
    if (!consume_flag(i, argc, argv, name, s)) return false;
    out = std::atoi(s.c_str());
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    Args a;
    a.mode = argv[1];
    if (a.mode == "help" || a.mode == "-h" || a.mode == "--help") { usage(); return 0; }
    if (a.mode == "ce-final") {
        return tcore::ce_final::run(argc - 2, argv + 2);
    }

    int i = 2;
    if (a.mode == "bench") {
        if (i >= argc) { usage(); return 1; }
        std::string sub = argv[i++];
        if (sub != "greedy") {
            std::cerr << "only 'bench greedy' is supported in this release\n";
            return 2;
        }
        while (i < argc) {
            if (consume_int(i, argc, argv, "--row-num", a.row_num)) {}
            else if (consume_int(i, argc, argv, "--workers", a.workers)) {}
            else if (consume_int(i, argc, argv, "--episodes", a.episodes)) {}
            else if (consume_flag(i, argc, argv, "--theta", a.theta)) {}
            else if (consume_flag(i, argc, argv, "--out", a.out)) {}
            else if (consume_int(i, argc, argv, "--max-steps", a.max_steps)) {}
            else if (consume_int(i, argc, argv, "--seed-offset", a.seed_offset)) {}
            else { std::cerr << "unknown arg " << argv[i] << "\n"; return 2; }
            ++i;
        }
        return cmd_bench(a);
    }
    usage();
    return 1;
}
