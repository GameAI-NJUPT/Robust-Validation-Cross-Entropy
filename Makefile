# Tetris C++ core build (Linux g++ 13+, also MinGW-w64 g++ 14).
#
#   make           - build the tetris_core binary
#   make tests     - build unit tests
#   make clean
#
# Note: -iquote is used (not -I) so the project header features.h does not
# shadow the system <features.h> pulled in by libstdc++.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -DNDEBUG
LDFLAGS  ?=
LDLIBS   ?= -pthread

SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
BUILD_CE := $(BUILD_DIR)/ce_final

CORE_SRCS := \
    $(SRC_DIR)/pieces.cpp \
    $(SRC_DIR)/tetris_state.cpp \
    $(SRC_DIR)/features.cpp \
    $(SRC_DIR)/value_function.cpp \
    $(SRC_DIR)/ce_final/ce_final.cpp

CORE_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CORE_SRCS))

TARGET := $(BUILD_DIR)/tetris_core
TEST_TARGETS := $(BUILD_DIR)/test_rng

.PHONY: all tests clean
all: $(TARGET)

$(BUILD_DIR) $(BUILD_CE):
	@mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(BUILD_CE)
	$(CXX) $(CXXFLAGS) -iquote $(SRC_DIR) -c $< -o $@

$(TARGET): $(CORE_OBJS) $(BUILD_DIR)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -iquote $(SRC_DIR) -c $< -o $@

tests: $(TEST_TARGETS)

$(BUILD_DIR)/test_rng: $(TEST_DIR)/test_rng.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -iquote $(SRC_DIR) $< -o $@ $(LDLIBS)

clean:
	@rm -rf $(BUILD_DIR)
