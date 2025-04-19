# =========================
#  Simple Makefile
# =========================

# ---- user‑configurable -----------------------------------
CC       := gcc
CFLAGS   := -std=c99 -Wall -Wextra -O3 -mavx2
LDFLAGS  := -lm
TARGETS  := quant_matmul quant_matmul_thread
SRCDIR   := src
OBJDIR   := build
# ----------------------------------------------------------

# source → object lists
SRCS     := main.c main_thread.c
OBJS     := $(patsubst %,$(OBJDIR)/%,$(SRCS:.c=.o))

.PHONY: all clean debug

# Build everything by default
all: $(TARGETS)

# Create the build directory automatically
$(OBJDIR):
	@mkdir -p $@

# Generic rule:  src/foo.c → build/foo.o
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- executable links ------------------------------------

# Single‑thread, vectorised version
quant_matmul: $(OBJDIR)/main.o
	$(CC) $^ $(LDFLAGS) -o $@

# Multi‑threaded version (adds pthreads)
quant_matmul_thread: $(OBJDIR)/main_thread.o
	$(CC) $^ $(LDFLAGS) -pthread -o $@

# ---- convenience targets --------------------------------

debug: CFLAGS := -std=c99 -Wall -Wextra -g -O0
debug: clean all

clean:
	$(RM) -r $(OBJDIR) $(TARGETS)
