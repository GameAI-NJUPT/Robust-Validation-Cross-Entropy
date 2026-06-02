#include "pieces.h"

namespace tcore {

// Note on ordering. The original Java declares pieces in this exact order, so
// we keep it: O, I, S, Z, L, J, T. Rotation bitmasks come straight from
// Blocks.java; the comment after each entry is the original Java identifier.
const std::array<Piece, kPieceCount> kPieces = {{
    // 0: O
    Piece{1, 9, 2, {
        Rotation{2, 2, {3, 3, 0, 0}}, // O_block1 {3,3}
        Rotation{},
        Rotation{},
        Rotation{},
    }},
    // 1: I
    Piece{2, 17, 4, {
        Rotation{1, 4, {15, 0, 0, 0}}, // I_block1 {15}
        Rotation{4, 1, {1, 1, 1, 1}},  // I_block2 {1,1,1,1}
        Rotation{},
        Rotation{},
    }},
    // 2: S
    Piece{2, 17, 3, {
        Rotation{2, 3, {6, 3, 0, 0}},  // S_block1 {6,3}
        Rotation{3, 2, {1, 3, 2, 0}},  // S_block2 {1,3,2}
        Rotation{},
        Rotation{},
    }},
    // 3: Z (Java identifier OS)
    Piece{2, 17, 3, {
        Rotation{2, 3, {3, 6, 0, 0}},  // OS_block1 {3,6}
        Rotation{3, 2, {2, 3, 1, 0}},  // OS_block2 {2,3,1}
        Rotation{},
        Rotation{},
    }},
    // 4: L
    Piece{4, 34, 3, {
        Rotation{2, 3, {7, 1, 0, 0}},  // L_block1 {7,1}
        Rotation{3, 2, {3, 2, 2, 0}},  // L_block2 {3,2,2}
        Rotation{2, 3, {4, 7, 0, 0}},  // L_block3 {4,7}
        Rotation{3, 2, {1, 1, 3, 0}},  // L_block4 {1,1,3}
    }},
    // 5: J (Java identifier OL)
    Piece{4, 34, 3, {
        Rotation{2, 3, {1, 7, 0, 0}},  // OL_block1 {1,7}
        Rotation{3, 2, {3, 1, 1, 0}},  // OL_block2 {3,1,1}
        Rotation{2, 3, {7, 4, 0, 0}},  // OL_block3 {7,4}
        Rotation{3, 2, {2, 2, 3, 0}},  // OL_block4 {2,2,3}
    }},
    // 6: T
    Piece{4, 34, 3, {
        Rotation{3, 2, {1, 3, 1, 0}},  // T_block1 {1,3,1}
        Rotation{2, 3, {7, 2, 0, 0}},  // T_block2 {7,2}
        Rotation{3, 2, {2, 3, 2, 0}},  // T_block3 {2,3,2}
        Rotation{2, 3, {2, 7, 0, 0}},  // T_block4 {2,7}
    }},
}};

PieceAction g_actionTable[kPieceCount][kMaxActions];
int         g_actionCount[kPieceCount];

void init_action_table() {
    for (int p = 0; p < kPieceCount; ++p) {
        const Piece& piece = kPieces[p];
        int idx = 0;
        for (int r = 0; r < piece.rotationCount; ++r) {
            const Rotation& rot = piece.rotations[r];
            const int slots = 11 - rot.width;
            for (int sh = 0; sh < slots; ++sh) {
                PieceAction& a = g_actionTable[p][idx];
                a.rotation = static_cast<int8_t>(r);
                a.shift    = static_cast<int8_t>(sh);
                a.width    = static_cast<int8_t>(rot.width);
                a.height   = static_cast<int8_t>(rot.height);
                for (int c = 0; c < kMaxPieceWidth; ++c) a.mask[c] = rot.mask[c];
                ++idx;
            }
        }
        g_actionCount[p] = idx;
    }
}

} // namespace tcore
