// Nine-dimensional Dellacherie/CBMPI afterstate features.
//
// All routines are pure functions of an afterstate (post-drop, post-collapse)
// plus, for landing height and eroded piece cells, the rotation that was
// placed and the pre-collapse delete mask. The features are returned as raw
// (unsigned) values; the loaded weight vector carries the sign convention, so
// the linear value function is a single dot product theta^T phi.
//
// Feature order:
//   0 rowsWithHoles
//   1 columnTransitions
//   2 cumulativeWells
//   3 landingHeight
//   4 rowTransitions
//   5 holes
//   6 erodedPieceCells
//   7 patternDiversity
//   8 holeDepth

#pragma once

#include <cstdint>

#include "pieces.h"
#include "tetris_state.h"

namespace tcore {

constexpr int kFeatureDim = 9;

// Computes the raw 9-d feature vector for the action applied to s. Internally
// performs the same drop as the reference implementation (so we get rotation,
// dropHeight and deleteMask without an extra allocation) and writes the
// features to out[0..8]. as_out, when non-null, receives the resulting
// afterstate.
void compute_features(const State& s, int action, double* out,
                      State* as_out = nullptr);

} // namespace tcore
