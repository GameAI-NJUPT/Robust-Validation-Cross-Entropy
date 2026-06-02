// Tetris piece tables. Mirrors 20260414TetrisSuperAI/src/tetris/Blocks.java
// exactly: same piece order, same rotation order, same width/height, same
// column bitmasks, and the same action index decomposition into (rotation,
// shift).
//
// In the original Java code each block is a small int[] where entry i is the
// vertical bitmask for column i of the rotated piece. We reproduce those
// numbers verbatim. Action indexing follows Blocks.getAction(): rotations are
// enumerated in declaration order and within each rotation shift goes from 0
// to 10 - width inclusive (i.e. 11 - width slots).
//
// The fully expanded action table maps every legal (piece, action) pair to
// the rotation index, shift, and a copy of the rotated piece's column mask.
// This is precomputed once at startup so that the hot path (legal_actions,
// after_state, compute_features) never branches on piece id more than once.

#pragma once

#include <array>
#include <cstdint>

namespace tcore {

constexpr int kPieceCount     = 7;
constexpr int kBoardCols      = 10;
constexpr int kMaxActions     = 34;
constexpr int kMaxPieceWidth  = 4;

struct Rotation {
    int     width;
    int     height;
    uint32_t mask[kMaxPieceWidth]; // unused trailing entries are 0
};

struct Piece {
    int      rotationCount;
    int      actionSize;
    int      maxHeight;
    Rotation rotations[4];
};

struct PieceAction {
    int8_t rotation;       // rotation index used by this action
    int8_t shift;          // shift used by this action
    int8_t width;          // width of the rotated piece
    int8_t height;         // height of the rotated piece
    uint32_t mask[kMaxPieceWidth];
};

// All piece data, indexed by piece id (0..6).
extern const std::array<Piece, kPieceCount> kPieces;

// Decoded (rotation, shift, mask) table for every (piece, action) pair, plus
// the per-piece action counts. Filled once by init_action_table() and then
// read-only.
extern PieceAction         g_actionTable[kPieceCount][kMaxActions];
extern int                 g_actionCount[kPieceCount];

void init_action_table();

} // namespace tcore
