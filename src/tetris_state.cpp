#include "tetris_state.h"

#include <cstring>

namespace tcore {

int g_rowNum = 10;
void set_row_num(int n) { g_rowNum = n; }

namespace {

// Java's TetrisEnvironment.searchHeight: returns the 1-indexed position of the
// most significant set bit. d=0 maps to 0 (special-cased by callers).
int search_height(uint32_t d) {
    int r = 0;
    if (d & 0xffff0000u) { d >>= 16; r += 16; }
    if (d & 0x0000ff00u) { d >>= 8;  r += 8;  }
    if (d & 0x000000f0u) { d >>= 4;  r += 4;  }
    if (d & 0x0000000cu) { d >>= 2;  r += 2;  }
    if (d & 0x00000002u) {            r += 1; }
    return r;
}

} // namespace

int search_highest_column(const uint32_t* cols) {
    int max = 0;
    for (int i = 0; i < kBoardCols; ++i) {
        int h = cols[i] != 0 ? search_height(cols[i]) : -1;
        if (h > max) max = h;
    }
    return max;
}

int get_highest(const uint32_t* cols, const PieceAction& a) {
    int max = 0;
    for (int i = 0; i < a.width; ++i) {
        uint32_t c = cols[a.shift + i];
        int h = c != 0 ? search_height(c) : -1;
        if (h > max) max = h;
    }
    return max;
}

bool valid(const uint32_t* cols) {
    const uint32_t guard = 1u << g_rowNum;
    for (int i = 0; i < kBoardCols; ++i) {
        if (cols[i] & guard) return false;
    }
    return true;
}

// Replicates the temporary afterstate construction inside Java's
// valid(int[], Block, shift) without touching the original state.
bool valid_drop(const State& s, const PieceAction& a) {
    uint32_t cols[kBoardCols];
    std::memcpy(cols, s.col, sizeof(cols));

    int h = get_highest(cols, a);
    int dropHigh = 0;
    uint32_t blockOver = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < a.width; ++j) {
            blockOver += (a.mask[j] << (h - i)) & cols[a.shift + j];
        }
        if (blockOver != 0 || (h - i) == -1) {
            dropHigh = h - i + 1;
            break;
        }
    }
    for (int j = 0; j < a.width; ++j) {
        cols[a.shift + j] |= (a.mask[j] << dropHigh);
    }
    return valid(cols);
}

uint64_t action_list(const State& s) {
    uint64_t actions = 0;
    const int n = g_actionCount[s.piece];
    for (int i = 0; i < n; ++i) {
        if (valid_drop(s, g_actionTable[s.piece][i])) {
            actions |= (uint64_t{1} << i);
        }
    }
    return actions;
}

State after_state(const State& s, int action) {
    State out = s;
    const PieceAction& a = g_actionTable[s.piece][action];

    int h = get_highest(out.col, a);
    uint32_t blockOver = 0;
    int dropHigh = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < a.width; ++j) {
            blockOver += (a.mask[j] << (h - i)) & out.col[a.shift + j];
        }
        if (blockOver != 0 || (h - i) == -1) {
            dropHigh = h - i + 1;
            break;
        }
    }
    out.drop_height = dropHigh;
    for (int j = 0; j < a.width; ++j) {
        out.col[a.shift + j] |= (a.mask[j] << dropHigh);
    }

    // Pre-collapse delete mask: bitwise AND of all 10 columns.
    uint32_t pre = out.col[0];
    for (int i = 1; i < kBoardCols; ++i) pre &= out.col[i];
    out.delete_mask = pre;

    int reward = 0;
    while (true) {
        uint32_t deleteLine = out.col[0];
        for (int i = 1; i < kBoardCols; ++i) deleteLine &= out.col[i];
        if (deleteLine == 0) break;
        // Java fills all bits at and below the highest set delete bit so the
        // collapse mask covers every row that ever gets shifted down.
        deleteLine |= (deleteLine >> 1);
        deleteLine |= (deleteLine >> 2);
        deleteLine |= (deleteLine >> 4);
        deleteLine |= (deleteLine >> 8);
        deleteLine |= (deleteLine >> 16);
        for (int i = 0; i < kBoardCols; ++i) {
            uint32_t above = (out.col[i] & ~deleteLine) >> 1;
            uint32_t below = (deleteLine >> 1) & out.col[i];
            out.col[i] = above | below;
        }
        ++reward;
    }
    out.reward = reward;
    return out;
}

State initial_state(int piece) {
    State s{};
    s.piece = piece;
    return s;
}

State successor_state(const State& s, int action, int next_piece) {
    State n = after_state(s, action);
    n.score += n.reward;
    n.piece  = next_piece;
    return n;
}

bool is_final(const State& s) {
    return action_list(s) == 0;
}

} // namespace tcore
