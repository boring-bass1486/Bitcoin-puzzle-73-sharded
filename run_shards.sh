#!/usr/bin/env bash
# Launch N shards. A compiled C helper redraws exactly N+1 lines in-place.
#
# Usage: ./run_shards.sh [N]   (default N=4)

N=${1:-4}

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ] || [ "$N" -gt 256 ]; then
    echo "error: N must be an integer in 1..256 (got '$N')" >&2
    exit 2
fi

if [ ! -x ./main ]; then
    echo "error: ./main not found or not executable. Run 'make' first." >&2
    exit 1
fi

STATUS_DIR=$(mktemp -d)

cleanup() {
    pkill -TERM -x main 2>/dev/null
    wait 2>/dev/null
    rm -rf "$STATUS_DIR"
    echo "All shards stopped. Checkpoints saved."
    exit 0
}
trap cleanup INT TERM

export PUZZLE_COMPACT=1

# ── Compile a tiny display helper (reads status files, rewrites N+1 lines) ──
clang -O2 -x c -o "$STATUS_DIR/display" - << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>

/* read the last non-empty line from a file into buf (max len-1 chars) */
static int read_last_line(const char *path, char *buf, int len)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    buf[0] = '\0';
    char tmp[1024];
    while (fgets(tmp, sizeof tmp, f)) {
        int l = (int)strlen(tmp);
        while (l > 0 && (tmp[l-1]=='\n' || tmp[l-1]=='\r')) tmp[--l]='\0';
        if (l > 0) { strncpy(buf, tmp, len-1); buf[len-1] = '\0'; }
    }
    fclose(f);
    return buf[0] != '\0';
}

/* parse the float immediately before the first occurrence of 'tag' in line */
static double parse_before(const char *line, const char *tag)
{
    const char *p = strstr(line, tag);
    if (!p || p == line) return 0.0;
    p--;
    while (p > line && *p == ' ') p--;
    while (p > line && *p != ' ') p--;
    return atof(p);
}

/* parse "att:NNN" → unsigned long long */
static unsigned long long parse_att(const char *line)
{
    const char *p = strstr(line, "att:");
    if (!p) return 0ULL;
    return strtoull(p + 4, NULL, 10);
}

/* parse "el:NNN.N" → double */
static double parse_el(const char *line)
{
    const char *p = strstr(line, "el:");
    if (!p) return 0.0;
    return atof(p + 3);
}

/* format attempts as X.XXG / X.XXM / NNNK */
static void fmt_att(unsigned long long att, char *buf, int len)
{
    if      (att >= 1000000000ULL) snprintf(buf, len, "%.2fG", att / 1e9);
    else if (att >= 1000000ULL)    snprintf(buf, len, "%.2fM", att / 1e6);
    else                           snprintf(buf, len, "%lluK",  att / 1000ULL);
}

/* format elapsed seconds as Xd Xh Xm Xs */
static void fmt_el(double sec, char *buf, int len)
{
    int s = (int)sec;
    int d = s / 86400; s %= 86400;
    int h = s / 3600;  s %= 3600;
    int m = s / 60;    s %= 60;
    if      (d) snprintf(buf, len, "%dd %dh %dm", d, h, m);
    else if (h) snprintf(buf, len, "%dh %dm %ds", h, m, s);
    else if (m) snprintf(buf, len, "%dm %ds",         m, s);
    else        snprintf(buf, len, "%ds",                 s);
}

int main(int argc, char **argv)
{
    if (argc < 3) return 1;
    const char *dir = argv[1];
    int n = atoi(argv[2]);

    int npids = argc - 3;
    pid_t *pids = NULL;
    if (npids > 0) {
        pids = malloc(npids * sizeof(pid_t));
        for (int i = 0; i < npids; i++) pids[i] = (pid_t)atoi(argv[3+i]);
    }

    char path[512], line[1024], frame[16384];
    struct timespec ts = {0, 500000000L};

    while (1) {
        nanosleep(&ts, NULL);

        int pos = 0;
        pos += snprintf(frame+pos, sizeof frame-pos, "\033[2J\033[H");

        double total_speed = 0.0;
        unsigned long long total_att = 0ULL;
        double max_el = 0.0;
        int reporting = 0;

        for (int i = 1; i <= n; i++) {
            snprintf(path, sizeof path, "%s/shard_%d.txt", dir, i);
            if (read_last_line(path, line, sizeof line)) {
                double spd = parse_before(line, "Mk/s");
                unsigned long long att = parse_att(line);
                double el = parse_el(line);

                total_speed += spd;
                total_att   += att;
                if (el > max_el) max_el = el;
                reporting++;

                /* display the line up to (not including) the att: field */
                char *att_tag = strstr(line, "  att:");
                if (att_tag) *att_tag = '\0';
                pos += snprintf(frame+pos, sizeof frame-pos, "%s\n", line);
            } else {
                pos += snprintf(frame+pos, sizeof frame-pos,
                                "[%d/%d] starting...\n", i, n);
            }
        }

        char att_str[32], el_str[32];
        fmt_att(total_att, att_str, sizeof att_str);
        fmt_el(max_el,     el_str,  sizeof el_str);

        pos += snprintf(frame+pos, sizeof frame-pos,
                        "[fleet] %.1f Mk/s  |  %s keys  |  %s elapsed  |  %d/%d shards\n",
                        total_speed, att_str, el_str, reporting, n);

        fwrite(frame, 1, pos, stdout);
        fflush(stdout);

        if (pids) {
            int alive = 0;
            for (int i = 0; i < npids; i++)
                if (kill(pids[i], 0) == 0 || errno == EPERM) alive++;
            if (!alive) break;
        }
    }
    free(pids);
    return 0;
}
CEOF

if [ $? -ne 0 ]; then
    echo "error: failed to compile display helper" >&2
    cleanup
fi

# ── Start each shard (stdout → status file, stderr discarded) ──
declare -a PIDS
for i in $(seq 1 "$N"); do
    ./main --shard "$i/$N" > "$STATUS_DIR/shard_$i.txt" 2>/dev/null &
    PIDS+=($!)
done

# ── Run display helper (blocks until all shards exit) ──
"$STATUS_DIR/display" "$STATUS_DIR" "$N" "${PIDS[@]}"

wait
rm -rf "$STATUS_DIR"
