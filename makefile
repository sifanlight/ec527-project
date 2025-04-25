# =========================
#  Simple Makefile
# =========================

# ---- user-configurable -----------------------------------
CXX       := g++
CXXFLAGS  := -std=c++11 -Wall -Wextra -O3 \
             -mavx2 -mfma -msse4.1
LDLIBS    := -lboost_system -lboost_thread -lm -pthread

TARGETS   := quant_matmul quant_matmul_thread
SRCDIR    := src
OBJDIR    := build
# ----------------------------------------------------------

# Source / object lists (add new *.c files here if needed)
SRCS      := main.c main_thread.c
OBJS      := $(patsubst %,$(OBJDIR)/%,$(SRCS:.c=.o))

.PHONY: all clean debug

# ----------------------------------------------------------
# Default target
all: $(TARGETS)

# Make sure the object directory exists
$(OBJDIR):
	@mkdir -p $@

# Generic rule: src/foo.c  →  build/foo.o
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ----------------------------------------------------------
# Executable links (use the same linker flags for both)

quant_matmul: $(OBJDIR)/main.o
	$(CXX) $^ $(LDLIBS) -o $@

quant_matmul_thread: $(OBJDIR)/main_thread.o
	$(CXX) $^ $(LDLIBS) -o $@

# ----------------------------------------------------------
# Convenience targets

debug: CXXFLAGS := -std=c++11 -Wall -Wextra -g -O0 -mavx2 -mfma -msse4.1
debug: clean all

clean:
	$(RM) -r $(OBJDIR) $(TARGETS)
