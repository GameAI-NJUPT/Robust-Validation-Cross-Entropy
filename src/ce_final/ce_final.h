#pragma once

namespace tcore::ce_final {

// argv must start with the ce-final subcommand: benchmark, robust-ablation, or ratio.
int run(int argc, char** argv);

} // namespace tcore::ce_final
