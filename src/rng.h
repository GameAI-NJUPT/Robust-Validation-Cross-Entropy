// Deterministic RNG helpers used by the C++ Tetris engine.
//
// JavaRandom mirrors java.util.Random for the seeded constructor plus the
// nextInt(bound), nextLong(), nextDouble(), and nextGaussian() draws used by
// the Java TetrisCE/TetrisTeacher code.
//
// SplitMix64 is retained for older SplittableRandom smoke tests.
//
// References:
//   OpenJDK src/java.base/share/classes/java/util/Random.java
//   OpenJDK src/java.base/share/classes/java/util/SplittableRandom.java
//
// nextSeed:  seed += GOLDEN_GAMMA
// nextInt(bound): r = mix32(nextSeed()); rejection-sample if non-power-of-two
// mix32:      z = (z ^ (z >>> 33)) * 0x62a9d9ed799705f5L
//             return (int)(((z ^ (z >>> 28)) * 0xcb24d0a5c88c35b3L) >>> 32)

#pragma once

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tcore {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next_seed() { return state_ += kGamma; }

    int next_int(int bound) {
        // Non-power-of-two path with rejection sampling, identical to
        // SplittableRandom.nextInt(int).
        uint32_t m = static_cast<uint32_t>(bound) - 1u;
        if ((static_cast<uint32_t>(bound) & m) == 0) {
            // Power of two
            int32_t r = mix32(next_seed());
            return r & static_cast<int32_t>(m);
        }
        int32_t r = mix32(next_seed());
        int32_t u = static_cast<int32_t>(static_cast<uint32_t>(r) >> 1);
        while (true) {
            r = u % bound;
            // The Java check is: u + m - r < 0 (signed) which detects that
            // the unbiased upper bound was exceeded.
            if (static_cast<int32_t>(u + static_cast<int32_t>(m) - r) >= 0) break;
            u = static_cast<int32_t>(static_cast<uint32_t>(mix32(next_seed())) >> 1);
        }
        return r;
    }

private:
    static constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
    static int32_t mix32(uint64_t z) {
        z = (z ^ (z >> 33)) * 0x62A9D9ED799705F5ULL;
        return static_cast<int32_t>(
            ((z ^ (z >> 28)) * 0xCB24D0A5C88C35B3ULL) >> 32);
    }
    uint64_t state_;
};

class JavaRandom {
public:
    explicit JavaRandom(int64_t seed = 0) { set_seed(seed); }

    void set_seed(int64_t seed) {
        seed_ = (static_cast<uint64_t>(seed) ^ kMultiplier) & kMask;
        have_next_next_gaussian_ = false;
        next_next_gaussian_ = 0.0;
    }

    int32_t next_int(int32_t bound) {
        if (bound <= 0) throw std::invalid_argument("JavaRandom::next_int bound must be positive");
        int32_t m = bound - 1;
        if ((bound & m) == 0) {
            return static_cast<int32_t>((static_cast<int64_t>(bound) * next_bits(31)) >> 31);
        }
        int32_t u = static_cast<int32_t>(next_bits(31));
        int32_t r = u % bound;
        while (to_int32(static_cast<uint32_t>(u) - static_cast<uint32_t>(r) + static_cast<uint32_t>(m)) < 0) {
            u = static_cast<int32_t>(next_bits(31));
            r = u % bound;
        }
        return r;
    }

    int64_t next_long() {
        uint64_t high = static_cast<uint64_t>(next_bits(32)) << 32;
        int64_t low = static_cast<int64_t>(to_int32(next_bits(32)));
        return to_int64(high + static_cast<uint64_t>(low));
    }

    double next_double() {
        uint64_t a = static_cast<uint64_t>(next_bits(26));
        uint64_t b = static_cast<uint64_t>(next_bits(27));
        return static_cast<double>((a << 27) + b) / 9007199254740992.0;
    }

    double next_gaussian() {
        if (have_next_next_gaussian_) {
            have_next_next_gaussian_ = false;
            return next_next_gaussian_;
        }
        double v1, v2, s;
        do {
            v1 = 2.0 * next_double() - 1.0;
            v2 = 2.0 * next_double() - 1.0;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0 || s == 0.0);
        double multiplier = std::sqrt(-2.0 * std::log(s) / s);
        next_next_gaussian_ = v2 * multiplier;
        have_next_next_gaussian_ = true;
        return v1 * multiplier;
    }

    std::string serialize() const {
        std::ostringstream os;
        os << seed_ << " " << (have_next_next_gaussian_ ? 1 : 0)
           << " " << std::setprecision(17) << next_next_gaussian_;
        return os.str();
    }

    void deserialize(const std::string& s) {
        int have = 0;
        std::istringstream is(s);
        is >> seed_ >> have >> next_next_gaussian_;
        if (!is) throw std::runtime_error("invalid JavaRandom checkpoint state");
        seed_ &= kMask;
        have_next_next_gaussian_ = have != 0;
    }

private:
    static constexpr uint64_t kMultiplier = 0x5DEECE66DULL;
    static constexpr uint64_t kAddend = 0xBULL;
    static constexpr uint64_t kMask = (uint64_t{1} << 48) - 1;

    uint32_t next_bits(int bits) {
        seed_ = (seed_ * kMultiplier + kAddend) & kMask;
        return static_cast<uint32_t>(seed_ >> (48 - bits));
    }

    static int32_t to_int32(uint32_t x) {
        return x <= 0x7fffffffu
             ? static_cast<int32_t>(x)
             : static_cast<int32_t>(static_cast<int64_t>(x) - 4294967296LL);
    }

    static int64_t to_int64(uint64_t x) {
        return x <= 0x7fffffffffffffffULL
             ? static_cast<int64_t>(x)
             : std::numeric_limits<int64_t>::min()
                 + static_cast<int64_t>(x - 0x8000000000000000ULL);
    }

    uint64_t seed_ = 0;
    bool have_next_next_gaussian_ = false;
    double next_next_gaussian_ = 0.0;
};

} // namespace tcore
