// Tetris environment that mirrors TetrisEnvironment.java verbatim.
//
// State layout matches the Java int[15]:
//   col[0..9]   column bitmasks (bit k corresponds to row k from the top of
//               the well; Java uses RowNum-1-j ordering when projecting to a
//               2D board, but the bit math here is identical to the Java code)
//   reward      lines cleared by the last drop (col[10] in the Java array)
//   score       cumulative lines cleared    (col[11])
//   piece       current piece id 0..6       (col[12])
//   dropHeight  vertical drop position of last placement (col[13])
//   deleteMask  pre-collapse delete mask    (col[14])
//
// The runtime board height is held in g_rowNum so the same binary can be used
// for RowNum=10 and RowNum=20 parity exports.

#pragma once

#include <cstdint>

#include "pieces.h"

namespace tcore {

extern int g_rowNum;
void set_row_num(int n);

struct State {
    uint32_t col[kBoardCols];
    int32_t  reward;
    int32_t  score;
    int32_t  piece;
    int32_t  drop_height;
    uint32_t delete_mask;
};

// Highest occupied row (1-indexed) across all columns, matching
// TetrisEnvironment.seachHighestColumn.
int search_highest_column(const uint32_t* cols);

// Maximum height across the columns the piece will occupy starting from
// shift. Matches TetrisEnvironment.getHighest with the same off-by-one
// behaviour ("max" treats empty columns as -1).
int get_highest(const uint32_t* cols, const PieceAction& a);

// Returns true if every column bit below row RowNum is clear, mirroring
// TetrisEnvironment.valid(int[]).
bool valid(const uint32_t* cols);

// Predicate used by getActionList: returns true if dropping (piece, action)
// onto the provided columns stays within the board. The original Java
// constructs the full afterstate first; we replicate that exactly so the
// legal action mask is identical.
bool valid_drop(const State& s, const PieceAction& a);

// Drop helper. Computes the resulting columns, reward, drop height and the
// pre-collapse delete mask. The function never inspects the board height
// (rowNum) for the drop itself - it only uses rowNum during the line clear
// loop, just like the Java code.
//
// Returns the post-collapse state. The caller decides whether the move was
// legal (call valid_drop first or check valid() on the result).
State after_state(const State& s, int action);

// Legal-action bitmask, same encoding as Java getActionList (bit i set means
// action i is legal). Up to 34 actions per piece.
uint64_t action_list(const State& s);

// Sample a fresh initial state. piece is supplied by the caller so the C++
// port stays deterministic when reproducing Java parity exports.
State initial_state(int piece);

// Advance: applies action, accumulates score, sets the next piece.
State successor_state(const State& s, int action, int next_piece);

bool is_final(const State& s);

inline int action_count(int piece) { return g_actionCount[piece]; }
inline bool action_valid(int action, uint64_t mask) {
    return (mask & (uint64_t{1} << action)) != 0;
}
inline int action_size(uint64_t mask) { return __builtin_popcountll(mask); }

} // namespace tcore
