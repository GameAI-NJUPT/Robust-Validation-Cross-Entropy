#include "features.h"

namespace tcore {

namespace {

int popcount32(uint32_t x) { return __builtin_popcount(x); }

double landing_height(const State& as, const Rotation& rot) {
    // Java: dropHeight + (block.height - 1) / 2.0  (integer math on numerator)
    return as.drop_height + (rot.height - 1) / 2.0;
}

int eroded_piece_cells(const State& as, const Rotation& rot, int /*shift*/) {
    if (as.reward == 0) return 0;
    int num = 0;
    uint32_t deleteLine = as.delete_mask >> as.drop_height;
    for (int j = 0; j < rot.width; ++j) {
        uint32_t t = deleteLine & rot.mask[j];
        num += popcount32(t);
    }
    return num * as.reward;
}

int row_transitions(const uint32_t* col) {
    const uint32_t guard =
        (g_rowNum == 20) ? 0x000fffffu :
        (g_rowNum == 10) ? 0x000003ffu : ((1u << g_rowNum) - 1u);
    int num = 0;
    num += popcount32(col[0] ^ guard);
    num += popcount32(col[9] ^ guard);
    for (int i = 0; i < 9; ++i) {
        num += popcount32(col[i] ^ col[i + 1]);
    }
    return num;
}

int column_transitions(const uint32_t* col) {
    int num = 0;
    for (int i = 0; i < 10; ++i) {
        // Java: col ^ ((col << 1) + 1)
        uint32_t y = col[i] ^ ((col[i] << 1) + 1u);
        num += popcount32(y);
    }
    return num;
}

int holes(const uint32_t* col) {
    int num = 0;
    for (int i = 0; i < 10; ++i) {
        if (col[i] > 1) {
            uint32_t y = col[i];
            y |= y >> 1;
            y |= y >> 2;
            y |= y >> 4;
            y |= y >> 8;
            y |= y >> 16;
            y ^= col[i];
            num += popcount32(y);
        }
    }
    return num;
}

int row_holes(const uint32_t* col) {
    uint32_t y2 = 0;
    for (int i = 0; i < 10; ++i) {
        if (col[i] > 1) {
            uint32_t y1 = col[i];
            y1 |= y1 >> 1;
            y1 |= y1 >> 2;
            y1 |= y1 >> 4;
            y1 |= y1 >> 8;
            y1 |= y1 >> 16;
            y1 ^= col[i];
            y2 |= y1;
        }
    }
    return popcount32(y2);
}

int cumulative_wells(const uint32_t* col) {
    int num = 0;
    uint32_t exState[12];
    const uint32_t guard =
        (g_rowNum == 20) ? 0x000fffffu :
        (g_rowNum == 10) ? 0x000003ffu : ((1u << g_rowNum) - 1u);
    exState[0] = guard;
    exState[11] = guard;
    for (int i = 0; i < 10; ++i) exState[i + 1] = col[i];
    for (int i = 1; i < 11; ++i) {
        int y = 0;
        for (int j = 0; j < g_rowNum; ++j) {
            if (((exState[i] >> j) & 1u) == 0) {
                ++y;
                if (((exState[i - 1] >> j) & 1u) == 1 && ((exState[i + 1] >> j) & 1u) == 1) {
                    num += y;
                }
            } else {
                y = 0;
            }
        }
    }
    return num;
}

int hole_depth(const uint32_t* col) {
    int num = 0;
    for (int i = 0; i < 10; ++i) {
        // Java:
        //   y = ~(col ^ (col+1));
        //   y = col & y;
        // The expression ~(col ^ (col+1)) returns the bits at and above the
        // lowest zero bit of col. Then "col & y" picks the populated cells
        // above the first hole.
        uint32_t y = ~(col[i] ^ (col[i] + 1u));
        y = col[i] & y;
        num += popcount32(y);
    }
    return num;
}

int pattern_diversity(const uint32_t* col) {
    auto height = [](uint32_t c) -> int {
        if (c == 0) return -1;
        // searchHeight inline
        uint32_t d = c; int r = 0;
        if (d & 0xffff0000u) { d >>= 16; r += 16; }
        if (d & 0x0000ff00u) { d >>= 8;  r += 8;  }
        if (d & 0x000000f0u) { d >>= 4;  r += 4;  }
        if (d & 0x0000000cu) { d >>= 2;  r += 2;  }
        if (d & 0x00000002u) {            r += 1; }
        return r;
    };
    int count[5] = {0, 0, 0, 0, 0};
    const int p[5] = {-2, -1, 0, 1, 2};
    for (int i = 0; i < 9; ++i) {
        int h1 = height(col[i]);
        int h2 = height(col[i + 1]);
        int h = h1 - h2;
        for (int j = 0; j < 5; ++j) if (h == p[j]) ++count[j];
    }
    int num = 0;
    for (int j = 0; j < 5; ++j) if (count[j] != 0) ++num;
    return num;
}

} // namespace

void compute_features(const State& s, int action, double* out, State* as_out) {
    const PieceAction& pa = g_actionTable[s.piece][action];
    State as = after_state(s, action);

    out[0] = row_holes(as.col);
    out[1] = column_transitions(as.col);
    out[2] = cumulative_wells(as.col);
    out[3] = landing_height(as, kPieces[s.piece].rotations[pa.rotation]);
    out[4] = row_transitions(as.col);
    out[5] = holes(as.col);
    out[6] = eroded_piece_cells(as, kPieces[s.piece].rotations[pa.rotation], pa.shift);
    out[7] = pattern_diversity(as.col);
    out[8] = hole_depth(as.col);

    if (as_out) *as_out = as;
}

} // namespace tcore
