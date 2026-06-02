// Smoke test for SplitMix64: compares the first N draws of nextInt(7) against
// the Java SplittableRandom reference sequence dumped by DumpSplit.java.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/rng.h"

int main(int argc, char** argv) {
    long long seed = argc >= 2 ? std::atoll(argv[1]) : 12345;
    int n          = argc >= 3 ? std::atoi(argv[2])  : 20;
    tcore::SplitMix64 r(static_cast<uint64_t>(seed));
    for (int i = 0; i < n; ++i) {
        if (i) std::printf(",");
        std::printf("%d", r.next_int(7));
    }
    std::printf("\n");
    return 0;
}
