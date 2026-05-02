all: main

CC = clang
override CFLAGS += -Wno-everything -pthread -lm -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512ifma -mbmi2 -march=native -mtune=native -O3 -funroll-loops -fno-plt -fomit-frame-pointer

SRCS = $(shell find . -name '.ccls-cache' -type d -prune -o -type f -name '*.c' -print)
HEADERS = $(shell find . -name '.ccls-cache' -type d -prune -o -type f -name '*.h' -print)

# Nix wrapper around clang strips -march=native unless this var is unset.
main: $(SRCS) $(HEADERS)
	unset NIX_ENFORCE_NO_NATIVE NIX_ENFORCE_NO_NATIVE_clang_wrapper && \
	$(CC) $(CFLAGS) $(SRCS) -o "$@"

main-debug: $(SRCS) $(HEADERS)
	unset NIX_ENFORCE_NO_NATIVE NIX_ENFORCE_NO_NATIVE_clang_wrapper && \
	$(CC) $(CFLAGS) -O0 $(SRCS) -o "$@"

clean:
	rm -f main main-debug
