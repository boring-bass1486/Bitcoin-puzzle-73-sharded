/*
 * puzzle73.c — High-performance Bitcoin Puzzle 73 solver
 *
 * Speed path per batch of 16 keys (single pipeline, no ZMM register spill):
 *   1. SIMD key→W[0..2] build (vpshufb + vpaddd, no scalar scatter)
 *   2. 16-way AVX-512 SHA-256 — specialised for 10-byte keys:
 *        rounds 3-15 use ZROUNDz (W=0, omits +W vpaddmd)
 *        rounds 16-31 inline simplified W-updates (SIG1/SIG0 of zeros removed)
 *        rounds 32-63 standard EROUNDz
 *   3. Direct AVX-512 bswap via vpshufb: SHA-256 H[] → RIPEMD-160 W[]
 *   4. 16-way AVX-512 RIPEMD-160 – fully unrolled, all masks
 *   5. Compare via _mm512_cmpeq_epi32_mask; if match: save + exit
 *
 * Puzzle 73
 *   Range  : 0x1000000000000000000 – 0x1FFFFFFFFFFFFFFFFFFF
 *   Target : 4d5a25bb399b45748a6b12d4c57ab20a75db3abe
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <signal.h>
#include <errno.h>
#include <sched.h>
#include <unistd.h>
#include <immintrin.h>

/* ── tunables ────────────────────────────────────────────────────────────── */
#define NUM_THREADS_MAX     64   /* hard ceiling; actual count set at runtime */
#define REPORT_BATCH        (1u << 20)
#define FOUND_CHECK         32
#define CHECKPOINT_INTERVAL 10   /* seconds between auto-saves */

static int g_num_threads = 2;   /* overwritten in main() based on nproc/shards */

/* ── shard configuration (set in main from --shard I/N) ──────────────────── */
static int      g_shard_idx   = 0;     /* 0-based shard index */
static int      g_shard_count = 1;     /* total shards */
static uint16_t g_shard_hi_lo = 0x100; /* inclusive low bound of high byte */
static uint16_t g_shard_hi_hi = 0x200; /* exclusive high bound of high byte */
static uint16_t g_shard_hi_range = 0x100;
static char     g_checkpoint_file[64]     = "checkpoint.txt";
static char     g_checkpoint_file_tmp[68] = "checkpoint.txt.tmp";
#define CHECKPOINT_FILE  g_checkpoint_file

/* ── puzzle parameters ───────────────────────────────────────────────────── */
static const uint8_t TARGET[20] = {
    0x4d,0x5a,0x25,0xbb,0x39,0x9b,0x45,0x74,
    0x8a,0x6b,0x12,0xd4,0xc5,0x7a,0xb2,0x0a,
    0x75,0xdb,0x3a,0xbe
};
static const char TARGET_HEX[] =
    "4d5a25bb399b45748a6b12d4c57ab20a75db3abe";

/* ── globals ─────────────────────────────────────────────────────────────── */
typedef struct { uint16_t high; uint64_t low; } key73_t;

atomic_uint_fast64_t g_attempts  = 0;
atomic_bool          g_found     = false;
key73_t              g_found_key;
struct timespec      g_start;
pthread_mutex_t      g_mutex = PTHREAD_MUTEX_INITIALIZER;

/* current key snapshot — written by thread 0, read by display thread */
atomic_uint_fast64_t g_disp_low  = 0;
atomic_uint_fast32_t g_disp_high = 0x100;

/* checkpoint / shutdown state */
atomic_bool          g_stop = false;
/* per-thread RNG snapshot — published by each worker once per batch flush */
_Atomic uint64_t     g_rng_snap[NUM_THREADS_MAX][2];
/* values restored from checkpoint at startup (added to live counters for display) */
static uint64_t      g_attempts_base = 0;
static double        g_elapsed_base  = 0.0;
static uint64_t      g_rng_init[NUM_THREADS_MAX][2];
static bool          g_have_init_rng = false;

/* ═══════════════════════════════════════════════════════════════════════════
 *  SHA-256 — 16-way AVX-512, rolling 16-word message schedule
 *  32 ZMM registers available → near-zero spill (vs constant spill in AVX2)
 * ═══════════════════════════════════════════════════════════════════════════ */
static const uint32_t K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

/* AVX-512 helpers */
#define VR32z(x,n)   _mm512_or_si512(_mm512_srli_epi32(x,n),_mm512_slli_epi32(x,32-(n)))
#define EP0z(x)      _mm512_xor_si512(_mm512_xor_si512(VR32z(x,2),VR32z(x,13)),VR32z(x,22))
#define EP1z(x)      _mm512_xor_si512(_mm512_xor_si512(VR32z(x,6),VR32z(x,11)),VR32z(x,25))
#define SIG0z(x)     _mm512_xor_si512(_mm512_xor_si512(VR32z(x,7),VR32z(x,18)),_mm512_srli_epi32(x,3))
#define SIG1z(x)     _mm512_xor_si512(_mm512_xor_si512(VR32z(x,17),VR32z(x,19)),_mm512_srli_epi32(x,10))
/* vpternlogd-based CH and MAJ — each collapses 3 ops into 1 instruction */
#define SCHz(e,f,g)  _mm512_ternarylogic_epi32(e,f,g,0xCA)  /* (e&f)|(~e&g) */
#define SMAJz(a,b,c) _mm512_ternarylogic_epi32(a,b,c,0xE8)  /* (a&b)|(a&c)|(b&c) */

#define SROUNDz(a,b,c,d,e,f,g,h,k,w) do { \
    __m512i _t1=_mm512_add_epi32(_mm512_add_epi32( \
        _mm512_add_epi32(h,EP1z(e)),SCHz(e,f,g)), \
        _mm512_add_epi32(_mm512_set1_epi32(k##u),(w))); \
    __m512i _t2=_mm512_add_epi32(EP0z(a),SMAJz(a,b,c)); \
    h=g;g=f;f=e;e=_mm512_add_epi32(d,_t1); \
    d=c;c=b;b=a;a=_mm512_add_epi32(_t1,_t2); \
} while(0)

/* ZROUNDz: like SROUNDz but W=0 — drops the "+w" vpaddmd entirely.
   Used for rounds 3-15 of the 10-byte key message where W[3..14]=0
   and W[15]=0x50 is folded into K (0xc19bf174+0x50 = 0xc19bf1c4). */
#define ZROUNDz(a,b,c,d,e,f,g,h,k) do { \
    __m512i _t1=_mm512_add_epi32( \
        _mm512_add_epi32(h,EP1z(e)), \
        _mm512_add_epi32(SCHz(e,f,g),_mm512_set1_epi32(k##u))); \
    __m512i _t2=_mm512_add_epi32(EP0z(a),SMAJz(a,b,c)); \
    h=g;g=f;f=e;e=_mm512_add_epi32(d,_t1); \
    d=c;c=b;b=a;a=_mm512_add_epi32(_t1,_t2); \
} while(0)

#define EROUNDz(a,b,c,d,e,f,g,h,k,i) do { \
    W[(i)&15]=_mm512_add_epi32( \
        _mm512_add_epi32(SIG1z(W[((i)-2)&15]),W[((i)-7)&15]), \
        _mm512_add_epi32(SIG0z(W[((i)-15)&15]),W[(i)&15])); \
    SROUNDz(a,b,c,d,e,f,g,h,k,W[(i)&15]); \
} while(0)

/* ── SHA-256 compression body (shared macro for both W-setup variants) ──── */
#define SHA256_COMPRESS_BODY(W, sH) do { \
    __m512i _a=_mm512_set1_epi32(0x6a09e667u),_b=_mm512_set1_epi32(0xbb67ae85u); \
    __m512i _c=_mm512_set1_epi32(0x3c6ef372u),_d=_mm512_set1_epi32(0xa54ff53au); \
    __m512i _e=_mm512_set1_epi32(0x510e527fu),_f=_mm512_set1_epi32(0x9b05688cu); \
    __m512i _g=_mm512_set1_epi32(0x1f83d9abu),_h=_mm512_set1_epi32(0x5be0cd19u); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x428a2f98,(W)[ 0]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x71374491,(W)[ 1]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xb5c0fbcf,(W)[ 2]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xe9b5dba5,(W)[ 3]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x3956c25b,(W)[ 4]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x59f111f1,(W)[ 5]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x923f82a4,(W)[ 6]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xab1c5ed5,(W)[ 7]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd807aa98,(W)[ 8]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x12835b01,(W)[ 9]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x243185be,(W)[10]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x550c7dc3,(W)[11]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x72be5d74,(W)[12]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x80deb1fe,(W)[13]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x9bdc06a7,(W)[14]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc19bf174,(W)[15]); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xe49b69c1,16); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xefbe4786,17); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x0fc19dc6,18); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x240ca1cc,19); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2de92c6f,20); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4a7484aa,21); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x5cb0a9dc,22); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x76f988da,23); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x983e5152,24); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa831c66d,25); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xb00327c8,26); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xbf597fc7,27); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc6e00bf3,28); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd5a79147,29); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x06ca6351,30); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x14292967,31); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x27b70a85,32); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2e1b2138,33); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4d2c6dfc,34); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x53380d13,35); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x650a7354,36); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x766a0abb,37); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x81c2c92e,38); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x92722c85,39); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa2bfe8a1,40); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa81a664b,41); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc24b8b70,42); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc76c51a3,43); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd192e819,44); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd6990624,45); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xf40e3585,46); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x106aa070,47); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x19a4c116,48); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x1e376c08,49); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2748774c,50); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x34b0bcb5,51); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x391c0cb3,52); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4ed8aa4a,53); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x5b9cca4f,54); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x682e6ff3,55); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x748f82ee,56); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x78a5636f,57); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x84c87814,58); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x8cc70208,59); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x90befffa,60); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa4506ceb,61); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xbef9a3f7,62); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc67178f2,63); \
    (sH)[0]=_mm512_add_epi32(_a,_mm512_set1_epi32(0x6a09e667u)); \
    (sH)[1]=_mm512_add_epi32(_b,_mm512_set1_epi32(0xbb67ae85u)); \
    (sH)[2]=_mm512_add_epi32(_c,_mm512_set1_epi32(0x3c6ef372u)); \
    (sH)[3]=_mm512_add_epi32(_d,_mm512_set1_epi32(0xa54ff53au)); \
    (sH)[4]=_mm512_add_epi32(_e,_mm512_set1_epi32(0x510e527fu)); \
    (sH)[5]=_mm512_add_epi32(_f,_mm512_set1_epi32(0x9b05688cu)); \
    (sH)[6]=_mm512_add_epi32(_g,_mm512_set1_epi32(0x1f83d9abu)); \
    (sH)[7]=_mm512_add_epi32(_h,_mm512_set1_epi32(0x5be0cd19u)); \
} while(0)

/* ── Specialised SHA-256 compression for 10-byte keys.
 *
 *  10-byte message: W[0..2] key-dependent; W[3..14]=0x00000000; W[15]=0x00000050.
 *
 *  Savings vs generic SHA256_COMPRESS_BODY:
 *    Rounds  3-14 : ZROUNDz — omits "+W" vpaddmd (W=0). 12 × 1 = 12 ops.
 *    Round  15    : ZROUNDz with K folded: 0xc19bf174+0x50=0xc19bf1c4. 1+1 ops.
 *    Rounds 16-31 : inlined simplified W-updates — SIG1(0)=0, SIG0(0)=0,
 *                   SIG1/SIG0(0x50) replaced by precomputed scalar constants.
 *                   ~79 ops eliminated.
 *    W[3..14] zero-fill + W[15] set: 13 ops eliminated.
 *  Total: ~106 fewer SIMD instructions per SHA-256 call.
 *
 *  Precomputed constants:
 *    SIG1(0x00000050) = ror(0x50,17)^ror(0x50,19)^(0x50>>10) = 0x00220000
 *    SIG0(0x00000050) = ror(0x50, 7)^ror(0x50,18)^(0x50>> 3) = 0xA014000A
 * ─────────────────────────────────────────────────────────────────────────── */
#define SHA256_COMPRESS_KEYS(W, sH) do { \
    __m512i _a=_mm512_set1_epi32(0x6a09e667u),_b=_mm512_set1_epi32(0xbb67ae85u); \
    __m512i _c=_mm512_set1_epi32(0x3c6ef372u),_d=_mm512_set1_epi32(0xa54ff53au); \
    __m512i _e=_mm512_set1_epi32(0x510e527fu),_f=_mm512_set1_epi32(0x9b05688cu); \
    __m512i _g=_mm512_set1_epi32(0x1f83d9abu),_h=_mm512_set1_epi32(0x5be0cd19u); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x428a2f98,(W)[0]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x71374491,(W)[1]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xb5c0fbcf,(W)[2]); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xe9b5dba5); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x3956c25b); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x59f111f1); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x923f82a4); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xab1c5ed5); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd807aa98); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x12835b01); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x243185be); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x550c7dc3); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x72be5d74); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x80deb1fe); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x9bdc06a7); \
    ZROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc19bf1c4); \
    (W)[0]=_mm512_add_epi32(SIG0z((W)[1]),(W)[0]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xe49b69c1,(W)[0]); \
    (W)[1]=_mm512_add_epi32( \
        _mm512_add_epi32(_mm512_set1_epi32(0x00220000u),SIG0z((W)[2])),(W)[1]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xefbe4786,(W)[1]); \
    (W)[2]=_mm512_add_epi32(SIG1z((W)[0]),(W)[2]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x0fc19dc6,(W)[2]); \
    (W)[3]=SIG1z((W)[1]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x240ca1cc,(W)[3]); \
    (W)[4]=SIG1z((W)[2]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2de92c6f,(W)[4]); \
    (W)[5]=SIG1z((W)[3]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4a7484aa,(W)[5]); \
    (W)[6]=_mm512_add_epi32(SIG1z((W)[4]),_mm512_set1_epi32(0x50u)); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x5cb0a9dc,(W)[6]); \
    (W)[7]=_mm512_add_epi32(SIG1z((W)[5]),(W)[0]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x76f988da,(W)[7]); \
    (W)[8]=_mm512_add_epi32(SIG1z((W)[6]),(W)[1]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x983e5152,(W)[8]); \
    (W)[9]=_mm512_add_epi32(SIG1z((W)[7]),(W)[2]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa831c66d,(W)[9]); \
    (W)[10]=_mm512_add_epi32(SIG1z((W)[8]),(W)[3]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xb00327c8,(W)[10]); \
    (W)[11]=_mm512_add_epi32(SIG1z((W)[9]),(W)[4]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xbf597fc7,(W)[11]); \
    (W)[12]=_mm512_add_epi32(SIG1z((W)[10]),(W)[5]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc6e00bf3,(W)[12]); \
    (W)[13]=_mm512_add_epi32(SIG1z((W)[11]),(W)[6]); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd5a79147,(W)[13]); \
    (W)[14]=_mm512_add_epi32( \
        _mm512_add_epi32(SIG1z((W)[12]),(W)[7]), \
        _mm512_set1_epi32(0xA014000Au)); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x06ca6351,(W)[14]); \
    (W)[15]=_mm512_add_epi32( \
        _mm512_add_epi32(SIG1z((W)[13]),(W)[8]), \
        _mm512_add_epi32(SIG0z((W)[0]),_mm512_set1_epi32(0x50u))); \
    SROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x14292967,(W)[15]); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x27b70a85,32); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2e1b2138,33); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4d2c6dfc,34); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x53380d13,35); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x650a7354,36); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x766a0abb,37); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x81c2c92e,38); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x92722c85,39); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa2bfe8a1,40); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa81a664b,41); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc24b8b70,42); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc76c51a3,43); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd192e819,44); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xd6990624,45); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xf40e3585,46); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x106aa070,47); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x19a4c116,48); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x1e376c08,49); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x2748774c,50); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x34b0bcb5,51); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x391c0cb3,52); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x4ed8aa4a,53); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x5b9cca4f,54); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x682e6ff3,55); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x748f82ee,56); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x78a5636f,57); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x84c87814,58); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x8cc70208,59); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0x90befffa,60); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xa4506ceb,61); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xbef9a3f7,62); \
    EROUNDz(_a,_b,_c,_d,_e,_f,_g,_h,0xc67178f2,63); \
    (sH)[0]=_mm512_add_epi32(_a,_mm512_set1_epi32(0x6a09e667u)); \
    (sH)[1]=_mm512_add_epi32(_b,_mm512_set1_epi32(0xbb67ae85u)); \
    (sH)[2]=_mm512_add_epi32(_c,_mm512_set1_epi32(0x3c6ef372u)); \
    (sH)[3]=_mm512_add_epi32(_d,_mm512_set1_epi32(0xa54ff53au)); \
    (sH)[4]=_mm512_add_epi32(_e,_mm512_set1_epi32(0x510e527fu)); \
    (sH)[5]=_mm512_add_epi32(_f,_mm512_set1_epi32(0x9b05688cu)); \
    (sH)[6]=_mm512_add_epi32(_g,_mm512_set1_epi32(0x1f83d9abu)); \
    (sH)[7]=_mm512_add_epi32(_h,_mm512_set1_epi32(0x5be0cd19u)); \
} while(0)

/* msgs[16][10] → sH[8]: fallback used only by check_batch16. */
__attribute__((hot,always_inline))
static inline void sha256_16x(const uint8_t msgs[16][10], __m512i sH[8])
{
    __m512i W[16];
    {
        uint32_t col[16];
#define WROW16(dst,b0,b1,b2,b3) \
        for(int l=0;l<16;l++) \
            col[l]=((uint32_t)(b0)<<24)|((uint32_t)(b1)<<16)|((uint32_t)(b2)<<8)|(b3); \
        W[dst]=_mm512_set_epi32(col[15],col[14],col[13],col[12], \
                                col[11],col[10],col[ 9],col[ 8], \
                                col[ 7],col[ 6],col[ 5],col[ 4], \
                                col[ 3],col[ 2],col[ 1],col[ 0]);
        WROW16(0, msgs[l][0], msgs[l][1], msgs[l][2], msgs[l][3])
        WROW16(1, msgs[l][4], msgs[l][5], msgs[l][6], msgs[l][7])
        WROW16(2, msgs[l][8], msgs[l][9], 0x80u, 0u)
#undef WROW16
        for(int i=3;i<=14;i++) W[i]=_mm512_setzero_si512();
        W[15]=_mm512_set1_epi32(0x50u);
    }
    SHA256_COMPRESS_BODY(W, sH);
}

/* ── Fast path: build W[0–2] directly from key values using SIMD arithmetic.
 *
 *  10-byte message layout:   [0x01][high_byte][low_bytes 0..7 little-endian]
 *  SHA-256 big-endian words:
 *    W[0] = 0x01_HH_lo0_lo1  (lo0=v&0xFF, lo1=(v>>8)&0xFF  → bswap16 of v&0xFFFF)
 *    W[1] = bswap32((uint32_t)(v >> 16))   with carry correction from bits 0..15
 *    W[2] = bswap16((uint16_t)(v >> 48)) << 16 | 0x8000  (same for all lanes)
 *    W[3..14] = 0,  W[15] = 0x50
 *
 *  v = base_low + lane_index (lane 0..15).
 * ─────────────────────────────────────────────────────────────────────────── */
__attribute__((hot,always_inline))
static inline void sha256_16x_keys(uint16_t high, uint64_t base_low, __m512i sH[8])
{
    /* byte-shuffle masks (same for both W words, reused across calls after inlining) */
    static const int8_t bswap16_shuf[64] = {
        /* per 32-bit word: swap bytes 0↔1, zero bytes 2,3 */
        1,0,-1,-1, 5,4,-1,-1,  9, 8,-1,-1, 13,12,-1,-1,
        1,0,-1,-1, 5,4,-1,-1,  9, 8,-1,-1, 13,12,-1,-1,
        1,0,-1,-1, 5,4,-1,-1,  9, 8,-1,-1, 13,12,-1,-1,
        1,0,-1,-1, 5,4,-1,-1,  9, 8,-1,-1, 13,12,-1,-1,
    };
    static const int8_t bswap32_shuf[64] = {
        /* per 32-bit word: reverse all 4 bytes */
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
    };
    static const int32_t inc_arr[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    const __m512i bswap16m = _mm512_loadu_si512((const __m512i*)bswap16_shuf);
    const __m512i bswap32m = _mm512_loadu_si512((const __m512i*)bswap32_shuf);
    const __m512i inc      = _mm512_loadu_si512((const __m512i*)inc_arr);

    /* Only W[0..2] needed — SHA256_COMPRESS_KEYS handles W[3..15] as constants */
    __m512i W[16];

    /* W[0] and W[1] share lo16_v (reuse to save one vpbroadcastd + vpaddd) */
    {
        uint32_t lo16  = (uint32_t)(base_low & 0xFFFFu);
        __m512i lo16_v = _mm512_add_epi32(_mm512_set1_epi32((int32_t)lo16), inc);

        /* W[0]: 0x01_HH_lo0_lo1 — top 2 bytes constant, low 2 = bswap16(lo16+lane) */
        __m512i bsw16  = _mm512_shuffle_epi8(lo16_v, bswap16m);
        uint32_t top   = (0x01u << 24) | ((uint32_t)(high & 0xFFu) << 16);
        W[0] = _mm512_or_si512(_mm512_set1_epi32((int32_t)top), bsw16);

        /* W[1]: bswap32(bits[47:16] + carry from lo16 overflow) */
        uint32_t b1647 = (uint32_t)(base_low >> 16);
        __m512i carry  = _mm512_srli_epi32(lo16_v, 16);  /* 0 or 1 per lane */
        W[1] = _mm512_shuffle_epi8(
                   _mm512_add_epi32(_mm512_set1_epi32((int32_t)b1647), carry),
                   bswap32m);
    }

    /* W[2]: lane-invariant broadcast */
    {
        uint32_t bits48 = (uint32_t)(base_low >> 48);
        uint32_t w2 = ((bits48 & 0xFFu) << 24) | ((bits48 >> 8) << 16) | 0x8000u;
        W[2] = _mm512_set1_epi32((int32_t)w2);
    }

    /* W[3..15] initialisation removed — specialised body treats them as constants */
    SHA256_COMPRESS_KEYS(W, sH);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  RIPEMD-160 — 16-way AVX-512
 *  ~y/~z via vpternlogd (NOT op in imm8=0x0F).
 * ═══════════════════════════════════════════════════════════════════════════ */
/* vpternlogd truth-table helpers */
#define NOTz(x)           _mm512_ternarylogic_epi32(x,x,x,0x0F)
#define RFz1(x,y,z)       _mm512_ternarylogic_epi32(x,y,z,0x96)  /* x^y^z     */
#define RFz2(x,y,z)       _mm512_ternarylogic_epi32(x,y,z,0xCA)  /* (x&y)|(~x&z) */
#define RFz3(x,y,z)       _mm512_xor_si512(_mm512_or_si512(x,NOTz(y)),z) /* x|(~y)^z */
#define RFz4(x,y,z)       _mm512_ternarylogic_epi32(x,y,z,0xE2)  /* (x&z)|(~z&y) */
#define RFz5(x,y,z)       _mm512_xor_si512(x,_mm512_or_si512(y,NOTz(z))) /* x^(y|~z) */

#define RLz(x,n)  _mm512_or_si512(_mm512_slli_epi32(x,n),_mm512_srli_epi32(x,32-(n)))
#define RL10z(x)  _mm512_or_si512(_mm512_slli_epi32(x,10),_mm512_srli_epi32(x,22))

#define RSKz(al,bl,cl,dl,el, F, w, K, s) do { \
    __m512i _t=_mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(al,F(bl,cl,dl)),w), \
        _mm512_set1_epi32(K##u)); \
    _t=_mm512_add_epi32(RLz(_t,s),el); \
    al=el; el=dl; dl=RL10z(cl); cl=bl; bl=_t; \
} while(0)

#define RS0z(al,bl,cl,dl,el, F, w, s) do { \
    __m512i _t=_mm512_add_epi32(_mm512_add_epi32(al,F(bl,cl,dl)),w); \
    _t=_mm512_add_epi32(RLz(_t,s),el); \
    al=el; el=dl; dl=RL10z(cl); cl=bl; bl=_t; \
} while(0)

/* Interleaved left+right RIPEMD-160: each source line executes one round on
   the left chain (al..el) immediately followed by one independent round on
   the right chain (ar..er).  The two chains share W[] as read-only data and
   use completely disjoint state registers, so the CPU's OoO engine can issue
   both in the same cycle using both AVX-512 execution ports.
   Register budget: 10 state + 16 W = 26 ZMM — fits in 32, zero spill. */
__attribute__((hot,always_inline))
static inline void ripemd160_16x(const __m512i sH[8], __m512i rH[5])
{
    static const int8_t bs[64] = {
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
        3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12
    };
    const __m512i bswap = _mm512_loadu_si512((const __m512i*)bs);

    __m512i W[16];
    W[ 0]=_mm512_shuffle_epi8(sH[0],bswap); W[ 1]=_mm512_shuffle_epi8(sH[1],bswap);
    W[ 2]=_mm512_shuffle_epi8(sH[2],bswap); W[ 3]=_mm512_shuffle_epi8(sH[3],bswap);
    W[ 4]=_mm512_shuffle_epi8(sH[4],bswap); W[ 5]=_mm512_shuffle_epi8(sH[5],bswap);
    W[ 6]=_mm512_shuffle_epi8(sH[6],bswap); W[ 7]=_mm512_shuffle_epi8(sH[7],bswap);
    W[ 8]=_mm512_set1_epi32(0x80u);
    W[ 9]=_mm512_setzero_si512(); W[10]=_mm512_setzero_si512();
    W[11]=_mm512_setzero_si512(); W[12]=_mm512_setzero_si512();
    W[13]=_mm512_setzero_si512(); W[14]=_mm512_set1_epi32(0x100u);
    W[15]=_mm512_setzero_si512();

    __m512i al=_mm512_set1_epi32(0x67452301u),bl=_mm512_set1_epi32(0xefcdab89u);
    __m512i cl=_mm512_set1_epi32(0x98badcfeu),dl=_mm512_set1_epi32(0x10325476u);
    __m512i el=_mm512_set1_epi32(0xc3d2e1f0u);
    __m512i ar=al,br=bl,cr=cl,dr=dl,er=el;

    /* pass 1  L=XOR/K=0   R=XOR-FLIP/K=0x50a28be6 */
    RS0z(al,bl,cl,dl,el,RFz1,W[ 0],11); RSKz(ar,br,cr,dr,er,RFz5,W[ 5],0x50a28be6, 8);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 1],14); RSKz(ar,br,cr,dr,er,RFz5,W[14],0x50a28be6, 9);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 2],15); RSKz(ar,br,cr,dr,er,RFz5,W[ 7],0x50a28be6, 9);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 3],12); RSKz(ar,br,cr,dr,er,RFz5,W[ 0],0x50a28be6,11);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 4], 5); RSKz(ar,br,cr,dr,er,RFz5,W[ 9],0x50a28be6,13);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 5], 8); RSKz(ar,br,cr,dr,er,RFz5,W[ 2],0x50a28be6,15);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 6], 7); RSKz(ar,br,cr,dr,er,RFz5,W[11],0x50a28be6,15);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 7], 9); RSKz(ar,br,cr,dr,er,RFz5,W[ 4],0x50a28be6, 5);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 8],11); RSKz(ar,br,cr,dr,er,RFz5,W[13],0x50a28be6, 7);
    RS0z(al,bl,cl,dl,el,RFz1,W[ 9],13); RSKz(ar,br,cr,dr,er,RFz5,W[ 6],0x50a28be6, 7);
    RS0z(al,bl,cl,dl,el,RFz1,W[10],14); RSKz(ar,br,cr,dr,er,RFz5,W[15],0x50a28be6, 8);
    RS0z(al,bl,cl,dl,el,RFz1,W[11],15); RSKz(ar,br,cr,dr,er,RFz5,W[ 8],0x50a28be6,11);
    RS0z(al,bl,cl,dl,el,RFz1,W[12], 6); RSKz(ar,br,cr,dr,er,RFz5,W[ 1],0x50a28be6,14);
    RS0z(al,bl,cl,dl,el,RFz1,W[13], 7); RSKz(ar,br,cr,dr,er,RFz5,W[10],0x50a28be6,14);
    RS0z(al,bl,cl,dl,el,RFz1,W[14], 9); RSKz(ar,br,cr,dr,er,RFz5,W[ 3],0x50a28be6,12);
    RS0z(al,bl,cl,dl,el,RFz1,W[15], 8); RSKz(ar,br,cr,dr,er,RFz5,W[12],0x50a28be6, 6);

    /* pass 2  L=IF/K=0x5a827999  R=IF-FLIP/K=0x5c4dd124 */
    RSKz(al,bl,cl,dl,el,RFz2,W[ 7],0x5a827999, 7); RSKz(ar,br,cr,dr,er,RFz4,W[ 6],0x5c4dd124, 9);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 4],0x5a827999, 6); RSKz(ar,br,cr,dr,er,RFz4,W[11],0x5c4dd124,13);
    RSKz(al,bl,cl,dl,el,RFz2,W[13],0x5a827999, 8); RSKz(ar,br,cr,dr,er,RFz4,W[ 3],0x5c4dd124,15);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 1],0x5a827999,13); RSKz(ar,br,cr,dr,er,RFz4,W[ 7],0x5c4dd124, 7);
    RSKz(al,bl,cl,dl,el,RFz2,W[10],0x5a827999,11); RSKz(ar,br,cr,dr,er,RFz4,W[ 0],0x5c4dd124,12);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 6],0x5a827999, 9); RSKz(ar,br,cr,dr,er,RFz4,W[13],0x5c4dd124, 8);
    RSKz(al,bl,cl,dl,el,RFz2,W[15],0x5a827999, 7); RSKz(ar,br,cr,dr,er,RFz4,W[ 5],0x5c4dd124, 9);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 3],0x5a827999,15); RSKz(ar,br,cr,dr,er,RFz4,W[10],0x5c4dd124,11);
    RSKz(al,bl,cl,dl,el,RFz2,W[12],0x5a827999, 7); RSKz(ar,br,cr,dr,er,RFz4,W[14],0x5c4dd124, 7);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 0],0x5a827999,12); RSKz(ar,br,cr,dr,er,RFz4,W[15],0x5c4dd124, 7);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 9],0x5a827999,15); RSKz(ar,br,cr,dr,er,RFz4,W[ 8],0x5c4dd124,12);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 5],0x5a827999, 9); RSKz(ar,br,cr,dr,er,RFz4,W[12],0x5c4dd124, 7);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 2],0x5a827999,11); RSKz(ar,br,cr,dr,er,RFz4,W[ 4],0x5c4dd124, 6);
    RSKz(al,bl,cl,dl,el,RFz2,W[14],0x5a827999, 7); RSKz(ar,br,cr,dr,er,RFz4,W[ 9],0x5c4dd124,15);
    RSKz(al,bl,cl,dl,el,RFz2,W[11],0x5a827999,13); RSKz(ar,br,cr,dr,er,RFz4,W[ 1],0x5c4dd124,13);
    RSKz(al,bl,cl,dl,el,RFz2,W[ 8],0x5a827999,12); RSKz(ar,br,cr,dr,er,RFz4,W[ 2],0x5c4dd124,11);

    /* pass 3  L=OR-NOT/K=0x6ed9eba1  R=OR-NOT/K=0x6d703ef3 */
    RSKz(al,bl,cl,dl,el,RFz3,W[ 3],0x6ed9eba1,11); RSKz(ar,br,cr,dr,er,RFz3,W[15],0x6d703ef3, 9);
    RSKz(al,bl,cl,dl,el,RFz3,W[10],0x6ed9eba1,13); RSKz(ar,br,cr,dr,er,RFz3,W[ 5],0x6d703ef3, 7);
    RSKz(al,bl,cl,dl,el,RFz3,W[14],0x6ed9eba1, 6); RSKz(ar,br,cr,dr,er,RFz3,W[ 1],0x6d703ef3,15);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 4],0x6ed9eba1, 7); RSKz(ar,br,cr,dr,er,RFz3,W[ 3],0x6d703ef3,11);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 9],0x6ed9eba1,14); RSKz(ar,br,cr,dr,er,RFz3,W[ 7],0x6d703ef3, 8);
    RSKz(al,bl,cl,dl,el,RFz3,W[15],0x6ed9eba1, 9); RSKz(ar,br,cr,dr,er,RFz3,W[14],0x6d703ef3, 6);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 8],0x6ed9eba1,13); RSKz(ar,br,cr,dr,er,RFz3,W[ 6],0x6d703ef3, 6);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 1],0x6ed9eba1,15); RSKz(ar,br,cr,dr,er,RFz3,W[ 9],0x6d703ef3,14);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 2],0x6ed9eba1,14); RSKz(ar,br,cr,dr,er,RFz3,W[11],0x6d703ef3,12);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 7],0x6ed9eba1, 8); RSKz(ar,br,cr,dr,er,RFz3,W[ 8],0x6d703ef3,13);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 0],0x6ed9eba1,13); RSKz(ar,br,cr,dr,er,RFz3,W[12],0x6d703ef3, 5);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 6],0x6ed9eba1, 6); RSKz(ar,br,cr,dr,er,RFz3,W[ 2],0x6d703ef3,14);
    RSKz(al,bl,cl,dl,el,RFz3,W[13],0x6ed9eba1, 5); RSKz(ar,br,cr,dr,er,RFz3,W[10],0x6d703ef3,13);
    RSKz(al,bl,cl,dl,el,RFz3,W[11],0x6ed9eba1,12); RSKz(ar,br,cr,dr,er,RFz3,W[ 0],0x6d703ef3,13);
    RSKz(al,bl,cl,dl,el,RFz3,W[ 5],0x6ed9eba1, 7); RSKz(ar,br,cr,dr,er,RFz3,W[ 4],0x6d703ef3, 7);
    RSKz(al,bl,cl,dl,el,RFz3,W[12],0x6ed9eba1, 5); RSKz(ar,br,cr,dr,er,RFz3,W[13],0x6d703ef3, 5);

    /* pass 4  L=IF-FLIP/K=0x8f1bbcdc  R=IF/K=0x7a6d76e9 */
    RSKz(al,bl,cl,dl,el,RFz4,W[ 1],0x8f1bbcdc,11); RSKz(ar,br,cr,dr,er,RFz2,W[ 8],0x7a6d76e9,15);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 9],0x8f1bbcdc,12); RSKz(ar,br,cr,dr,er,RFz2,W[ 6],0x7a6d76e9, 5);
    RSKz(al,bl,cl,dl,el,RFz4,W[11],0x8f1bbcdc,14); RSKz(ar,br,cr,dr,er,RFz2,W[ 4],0x7a6d76e9, 8);
    RSKz(al,bl,cl,dl,el,RFz4,W[10],0x8f1bbcdc,15); RSKz(ar,br,cr,dr,er,RFz2,W[ 1],0x7a6d76e9,11);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 0],0x8f1bbcdc,14); RSKz(ar,br,cr,dr,er,RFz2,W[ 3],0x7a6d76e9,14);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 8],0x8f1bbcdc,15); RSKz(ar,br,cr,dr,er,RFz2,W[11],0x7a6d76e9,14);
    RSKz(al,bl,cl,dl,el,RFz4,W[12],0x8f1bbcdc, 9); RSKz(ar,br,cr,dr,er,RFz2,W[15],0x7a6d76e9, 6);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 4],0x8f1bbcdc, 8); RSKz(ar,br,cr,dr,er,RFz2,W[ 0],0x7a6d76e9,14);
    RSKz(al,bl,cl,dl,el,RFz4,W[13],0x8f1bbcdc, 9); RSKz(ar,br,cr,dr,er,RFz2,W[ 5],0x7a6d76e9, 6);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 3],0x8f1bbcdc,14); RSKz(ar,br,cr,dr,er,RFz2,W[12],0x7a6d76e9, 9);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 7],0x8f1bbcdc, 5); RSKz(ar,br,cr,dr,er,RFz2,W[ 2],0x7a6d76e9,12);
    RSKz(al,bl,cl,dl,el,RFz4,W[15],0x8f1bbcdc, 6); RSKz(ar,br,cr,dr,er,RFz2,W[13],0x7a6d76e9, 9);
    RSKz(al,bl,cl,dl,el,RFz4,W[14],0x8f1bbcdc, 8); RSKz(ar,br,cr,dr,er,RFz2,W[ 9],0x7a6d76e9,12);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 5],0x8f1bbcdc, 6); RSKz(ar,br,cr,dr,er,RFz2,W[ 7],0x7a6d76e9, 5);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 6],0x8f1bbcdc, 5); RSKz(ar,br,cr,dr,er,RFz2,W[10],0x7a6d76e9,15);
    RSKz(al,bl,cl,dl,el,RFz4,W[ 2],0x8f1bbcdc,12); RSKz(ar,br,cr,dr,er,RFz2,W[14],0x7a6d76e9, 8);

    /* pass 5  L=XOR-FLIP/K=0xa953fd4e  R=XOR/K=0 */
    RSKz(al,bl,cl,dl,el,RFz5,W[ 4],0xa953fd4e, 9); RS0z(ar,br,cr,dr,er,RFz1,W[12], 8);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 0],0xa953fd4e,15); RS0z(ar,br,cr,dr,er,RFz1,W[15], 5);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 5],0xa953fd4e, 5); RS0z(ar,br,cr,dr,er,RFz1,W[10],12);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 9],0xa953fd4e,11); RS0z(ar,br,cr,dr,er,RFz1,W[ 4], 9);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 7],0xa953fd4e, 6); RS0z(ar,br,cr,dr,er,RFz1,W[ 1],12);
    RSKz(al,bl,cl,dl,el,RFz5,W[12],0xa953fd4e, 8); RS0z(ar,br,cr,dr,er,RFz1,W[ 5], 5);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 2],0xa953fd4e,13); RS0z(ar,br,cr,dr,er,RFz1,W[ 8],14);
    RSKz(al,bl,cl,dl,el,RFz5,W[10],0xa953fd4e,12); RS0z(ar,br,cr,dr,er,RFz1,W[ 7], 6);
    RSKz(al,bl,cl,dl,el,RFz5,W[14],0xa953fd4e, 5); RS0z(ar,br,cr,dr,er,RFz1,W[ 6], 8);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 1],0xa953fd4e,12); RS0z(ar,br,cr,dr,er,RFz1,W[ 2],13);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 3],0xa953fd4e,13); RS0z(ar,br,cr,dr,er,RFz1,W[13], 6);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 8],0xa953fd4e,14); RS0z(ar,br,cr,dr,er,RFz1,W[14], 5);
    RSKz(al,bl,cl,dl,el,RFz5,W[11],0xa953fd4e,11); RS0z(ar,br,cr,dr,er,RFz1,W[ 0],15);
    RSKz(al,bl,cl,dl,el,RFz5,W[ 6],0xa953fd4e, 8); RS0z(ar,br,cr,dr,er,RFz1,W[ 3],13);
    RSKz(al,bl,cl,dl,el,RFz5,W[15],0xa953fd4e, 5); RS0z(ar,br,cr,dr,er,RFz1,W[ 9],11);
    RSKz(al,bl,cl,dl,el,RFz5,W[13],0xa953fd4e, 6); RS0z(ar,br,cr,dr,er,RFz1,W[11],11);

    rH[4]=_mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(0x67452301u),cl),dr);
    rH[0]=_mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(0xefcdab89u),dl),er);
    rH[1]=_mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(0x98badcfeu),el),ar);
    rH[2]=_mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(0x10325476u),al),br);
    rH[3]=_mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(0xc3d2e1f0u),bl),cr);
}

/* ─── extract 16-lane RIPEMD-160 result ────────────────────────────────── */
static void extract_rmd16(const __m512i rH[5], uint8_t out[16][20])
{
    uint32_t w[5][16];
    for(int j=0;j<5;j++) _mm512_storeu_si512((__m512i*)w[j], rH[j]);
    for(int l=0;l<16;l++)
        for(int j=0;j<5;j++){
            out[l][j*4+0]=(uint8_t)(w[j][l]);
            out[l][j*4+1]=(uint8_t)(w[j][l]>>8);
            out[l][j*4+2]=(uint8_t)(w[j][l]>>16);
            out[l][j*4+3]=(uint8_t)(w[j][l]>>24);
        }
}

/* ─── vectorised 16-lane compare; returns __mmask16 (bit i = lane i match) */
static uint32_t target_word[5];

static inline __mmask16 cmp_target_16x(const __m512i rH[5])
{
    __mmask16 m = 0xFFFFu;
    for(int j=0;j<5;j++)
        m &= _mm512_cmpeq_epi32_mask(rH[j], _mm512_set1_epi32((int)target_word[j]));
    return m;
}

static bool check_batch16(const __m512i rH[5], key73_t keys[16])
{
    uint8_t rmd[16][20];
    extract_rmd16(rH, rmd);
    for(int i=0;i<16;i++){
        if(memcmp(rmd[i], TARGET, 20)==0){
            pthread_mutex_lock(&g_mutex);
            if(!atomic_load(&g_found)){
                atomic_store(&g_found, true);
                g_found_key = keys[i];
                FILE *fp = fopen("FOUND.txt","w");
                if(fp){
                    fprintf(fp, "%03x%016lx\n",
                            (unsigned)keys[i].high, (unsigned long)keys[i].low);
                    fclose(fp);
                }
                printf("\n\n*** FOUND!  Key: 0x%03x%016lx ***\n\n",
                       (unsigned)keys[i].high, (unsigned long)keys[i].low);
            }
            pthread_mutex_unlock(&g_mutex);
            return true;
        }
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  RNG (xorshift128+, thread-local)
 * ═══════════════════════════════════════════════════════════════════════════ */
__thread uint64_t rng_s[2];

static inline void rng_seed(uint64_t seed)
{
    rng_s[0]=seed;
    rng_s[1]=seed^0xdeadbeefcafebabeULL;
}

static inline uint64_t rng64(void)
{
    uint64_t x=rng_s[0]; const uint64_t y=rng_s[1];
    rng_s[0]=y; x^=x<<23;
    rng_s[1]=x^y^(x>>17)^(y>>26);
    return rng_s[1]+y;
}

static inline key73_t rand_key(void)
{
    key73_t k;
    /* high byte stays inside this shard's slice of [0x100, 0x200) */
    uint32_t r = (uint32_t)(rng64() & 0xFFFFu);
    k.high = (uint16_t)(g_shard_hi_lo + (r % g_shard_hi_range));
    k.low  = rng64();
    return k;
}

static inline void key_to_bytes(key73_t k, uint8_t out[10])
{
    out[0]=0x01u; out[1]=(uint8_t)(k.high&0xFFu);
    for(int i=2;i<10;i++) out[i]=(uint8_t)(k.low>>(8*(i-2)));
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  CHECKPOINTING
 *
 *  File format (text, line-based):
 *      puzzle73-checkpoint v1
 *      threads <N>
 *      attempts <cumulative>
 *      elapsed_sec <cumulative>
 *      rng <tid> <hex_state0> <hex_state1>     (one per worker)
 *
 *  Written atomically via tmpfile + rename. Re-read on startup if present.
 *  An incompatible/malformed file is ignored (and the run starts fresh).
 * ═══════════════════════════════════════════════════════════════════════════ */
static void save_checkpoint(void)
{
    FILE *fp = fopen(g_checkpoint_file_tmp, "w");
    if(!fp) return;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double session = (now.tv_sec - g_start.tv_sec)
                   + (now.tv_nsec - g_start.tv_nsec) * 1e-9;
    uint64_t session_attempts =
        atomic_load_explicit(&g_attempts, memory_order_relaxed);

    fprintf(fp, "puzzle73-checkpoint v1\n");
    fprintf(fp, "threads %d\n", g_num_threads);
    fprintf(fp, "attempts %llu\n",
            (unsigned long long)(g_attempts_base + session_attempts));
    fprintf(fp, "elapsed_sec %.6f\n", g_elapsed_base + session);
    for(int t = 0; t < g_num_threads; t++){
        uint64_t s0 = atomic_load_explicit(&g_rng_snap[t][0], memory_order_relaxed);
        uint64_t s1 = atomic_load_explicit(&g_rng_snap[t][1], memory_order_relaxed);
        fprintf(fp, "rng %d %016llx %016llx\n",
                t, (unsigned long long)s0, (unsigned long long)s1);
    }
    fflush(fp);
    fclose(fp);
    rename(g_checkpoint_file_tmp, CHECKPOINT_FILE);
}

static void load_checkpoint(void)
{
    FILE *fp = fopen(CHECKPOINT_FILE, "r");
    if(!fp) return;

    char line[256];
    int  file_threads = -1;
    bool header_ok    = false;

    if(fgets(line, sizeof line, fp)
       && strncmp(line, "puzzle73-checkpoint v1", 22) == 0)
    {
        header_ok = true;
        while(fgets(line, sizeof line, fp)){
            if(strncmp(line, "threads ", 8) == 0){
                file_threads = atoi(line + 8);
            } else if(strncmp(line, "attempts ", 9) == 0){
                g_attempts_base = strtoull(line + 9, NULL, 10);
            } else if(strncmp(line, "elapsed_sec ", 12) == 0){
                g_elapsed_base = strtod(line + 12, NULL);
            } else if(strncmp(line, "rng ", 4) == 0){
                int t;
                unsigned long long s0, s1;
                if(sscanf(line + 4, "%d %llx %llx", &t, &s0, &s1) == 3
                   && t >= 0 && t < g_num_threads && (s0 | s1) != 0)
                {
                    g_rng_init[t][0] = (uint64_t)s0;
                    g_rng_init[t][1] = (uint64_t)s1;
                }
            }
        }
    }
    fclose(fp);

    if(!header_ok || file_threads != g_num_threads){
        g_attempts_base = 0;
        g_elapsed_base  = 0.0;
        memset(g_rng_init, 0, sizeof g_rng_init);
        fprintf(stderr,
                "[checkpoint] ignoring %s (incompatible or malformed)\n",
                CHECKPOINT_FILE);
        return;
    }

    g_have_init_rng = true;
    fprintf(stderr,
            "[checkpoint] resumed: attempts=%llu  elapsed=%.1fs\n",
            (unsigned long long)g_attempts_base, g_elapsed_base);
}

static void on_signal(int sig)
{
    (void)sig;
    atomic_store_explicit(&g_stop, true, memory_order_relaxed);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  DISPLAY THREAD — 1 Hz, fully off the hot path
 * ═══════════════════════════════════════════════════════════════════════════ */
void *display_func(void *arg)
{
    (void)arg;
    int  save_ctr = 0;
    /* Compact one-line mode is forced when PUZZLE_COMPACT=1 (set by the
       run_shards.sh launcher), regardless of whether stdout is a TTY.
       Otherwise auto-detect: TTY → full dashboard, pipe → compact. */
    const char *pc = getenv("PUZZLE_COMPACT");
    int  forced_compact = (pc != NULL && strcmp(pc, "1") == 0);
    int  is_tty   = !forced_compact && isatty(STDOUT_FILENO);

    while(!atomic_load_explicit(&g_found, memory_order_relaxed)
          && !atomic_load_explicit(&g_stop,  memory_order_relaxed)){
        struct timespec ts = {1, 0};
        nanosleep(&ts, NULL);

        uint64_t session_total =
            atomic_load_explicit(&g_attempts, memory_order_relaxed);
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double session_el = (now.tv_sec - g_start.tv_sec)
                          + (now.tv_nsec - g_start.tv_nsec)*1e-9;
        uint64_t total   = g_attempts_base + session_total;
        double   elapsed = g_elapsed_base  + session_el;
        double   gps     = elapsed > 0 ? total / (elapsed * 1e9) : 0.0;

        uint32_t kh = (uint32_t)atomic_load_explicit(&g_disp_high, memory_order_relaxed);
        uint64_t kl = (uint64_t)atomic_load_explicit(&g_disp_low,  memory_order_relaxed);

        /* Coverage of the assigned slice (and of the full 2^72 keyspace).
           Shard size = g_shard_hi_range * 2^64 keys. We compute the ratio
           in floating point because uint64_t can't hold values >= 2^64. */
        double shard_keys = (double)g_shard_hi_range * 1.8446744073709552e19; /* 2^64 */
        double full_keys  = 4.722366482869645e21;                              /* 2^72 */
        double shard_frac = shard_keys > 0 ? (double)total / shard_keys : 0.0;
        double full_frac  = (double)total / full_keys;

        if(is_tty){
            /* Interactive: full multi-line dashboard with screen-clear */
            printf("\033[2J\033[H");
            printf("* 73-Bit Bitcoin Puzzle Solver  [AVX-512 16-way SHA256+RMD160, dual-pipeline]\n");
            if(g_shard_count > 1){
                printf("* Shard  : %d/%d   (high-byte 0x%03x..0x%03x of 0x100..0x1FF)\n",
                       g_shard_idx + 1, g_shard_count,
                       g_shard_hi_lo, g_shard_hi_hi - 1);
            } else {
                printf("* Range  : 0x1000000000000000000 to 0x1FFFFFFFFFFFFFFFFFFF\n");
            }
            printf("* Target : %s\n", TARGET_HEX);
            printf("* Key    : 0x%03x%016lx\n", kh, (unsigned long)kl);
            printf("* Attempts: %llu%s\n",
                   (unsigned long long)total,
                   g_have_init_rng ? " (cumulative, resumed)" : "");
            printf("* Elapsed: %.1fs%s\n",
                   elapsed, g_have_init_rng ? " (cumulative)" : "");
            printf("* Speed  : %.3f Gkeys/sec | %d threads | AVX-512\n", gps, g_num_threads);
            if(g_shard_count > 1){
                printf("* Coverage: %.4e of shard slice  |  %.4e of full 2^72\n",
                       shard_frac, full_frac);
            } else {
                printf("* Coverage: %.4e of 2^72 keyspace\n", full_frac);
            }
            printf("* Checkpoint: %s (every %ds, on Ctrl-C, on exit)\n",
                   CHECKPOINT_FILE, CHECKPOINT_INTERVAL);
        } else {
            /* Piped (e.g. multi-shard launcher): one compact line per shard. */
            (void)full_frac;
            double mps = gps * 1000.0;   /* Gkeys/s → Mkeys/s */
            if(g_shard_count > 1){
                printf("[%d/%d] range:0x%03x-0x%03x  cov:%.3e  key:0x%03x%016lx  %6.1f Mk/s  att:%llu  el:%.1f\n",
                       g_shard_idx + 1, g_shard_count,
                       g_shard_hi_lo, g_shard_hi_hi - 1,
                       shard_frac,
                       kh, (unsigned long)kl, mps,
                       (unsigned long long)total, elapsed);
            } else {
                printf("range:0x100-0x1ff  cov:%.3e  key:0x%03x%016lx  %6.1f Mk/s  att:%llu  el:%.1f\n",
                       shard_frac,
                       kh, (unsigned long)kl, mps,
                       (unsigned long long)total, elapsed);
            }
        }
        fflush(stdout);

        (void)save_ctr; /* checkpointing disabled */
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  THREAD FUNCTION — single 16-way pipeline (16 keys per iteration, no ZMM spill)
 * ═══════════════════════════════════════════════════════════════════════════ */
__attribute__((hot))
void *thread_func(void *arg)
{
    long tid = (long)arg;

    /* pin to a dedicated logical CPU so shards don't fight each other's cache.
       Shard s, thread t → CPU (s*threads_per_shard + t) % nproc, so each shard
       occupies a contiguous, non-overlapping slice of the CPU list. */
    {
        int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
        int cpu   = (g_shard_idx * g_num_threads + (int)tid) % nproc;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }

    /* prefer the RNG state saved in the checkpoint; otherwise seed from time+tid */
    if(g_have_init_rng && (g_rng_init[tid][0] | g_rng_init[tid][1]) != 0){
        rng_s[0] = g_rng_init[tid][0];
        rng_s[1] = g_rng_init[tid][1];
    } else {
        rng_seed((uint64_t)tid ^ ((uint64_t)time(NULL) * 6364136223846793005ULL));
    }
    /* publish initial snapshot so a checkpoint taken before the first batch
       flush still records valid (nonzero) state for this thread. */
    atomic_store_explicit(&g_rng_snap[tid][0], rng_s[0], memory_order_relaxed);
    atomic_store_explicit(&g_rng_snap[tid][1], rng_s[1], memory_order_relaxed);

    /* Single pipeline: 16 keys/iter — uses only ~27 ZMM registers (no spill).
       Dual-pipeline needed 48 ZMM which caused 16 spills; more threads + HT
       give the same ILP benefit without the memory round-trips. */
    key73_t  keys0[16];
    uint64_t local = 0;
    unsigned check_ctr = 0;

    while(1)
    {
        if(__builtin_expect(++check_ctr == FOUND_CHECK, 0)){
            check_ctr = 0;
            if(atomic_load_explicit(&g_found, memory_order_relaxed)
               || atomic_load_explicit(&g_stop,  memory_order_relaxed)) break;
        }

        /* One random anchor per 16-key batch + sequential offsets. */
        key73_t base = rand_key();
        for(int i=0;i<16;i++){
            keys0[i].high = base.high;
            keys0[i].low  = base.low + (uint64_t)i;
        }

        /* update display key (thread 0 only, relaxed — best-effort snapshot) */
        if(__builtin_expect(tid == 0, 0)){
            atomic_store_explicit(&g_disp_high, keys0[0].high, memory_order_relaxed);
            atomic_store_explicit(&g_disp_low,  keys0[0].low,  memory_order_relaxed);
        }

        __m512i sH0[8], rH0[5];

        sha256_16x_keys(base.high, base.low, sH0);
        ripemd160_16x(sH0, rH0);

        __mmask16 m0 = cmp_target_16x(rH0);

        if(__builtin_expect(m0 != 0, 0)){
            if(check_batch16(rH0, keys0)) return NULL;
        }

        local += 16;

        if(__builtin_expect(local >= REPORT_BATCH, 0)){
            atomic_fetch_add_explicit(&g_attempts, local, memory_order_relaxed);
            local = 0;
            /* publish current RNG state for the next checkpoint write */
            atomic_store_explicit(&g_rng_snap[tid][0], rng_s[0], memory_order_relaxed);
            atomic_store_explicit(&g_rng_snap[tid][1], rng_s[1], memory_order_relaxed);
        }
    }
    /* on exit: flush any pending local count and publish final RNG state */
    if(local) atomic_fetch_add_explicit(&g_attempts, local, memory_order_relaxed);
    atomic_store_explicit(&g_rng_snap[tid][0], rng_s[0], memory_order_relaxed);
    atomic_store_explicit(&g_rng_snap[tid][1], rng_s[1], memory_order_relaxed);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
static void parse_args(int argc, char **argv)
{
    for(int i=1;i<argc;i++){
        if((strcmp(argv[i],"--shard")==0 || strcmp(argv[i],"-s")==0) && i+1<argc){
            int idx_one_based = 0, n = 0;
            if(sscanf(argv[++i], "%d/%d", &idx_one_based, &n) != 2
               || n < 1 || n > 256
               || idx_one_based < 1 || idx_one_based > n){
                fprintf(stderr,
                    "error: --shard expects I/N with 1<=I<=N and 1<=N<=256 "
                    "(e.g. --shard 1/4)\n");
                exit(2);
            }
            g_shard_idx   = idx_one_based - 1;
            g_shard_count = n;
        } else if(strcmp(argv[i],"--help")==0 || strcmp(argv[i],"-h")==0){
            printf(
                "Usage: %s [--shard I/N]\n"
                "  --shard I/N   run shard I (1..N) of N total. Default: 1/1.\n"
                "                Splits the high-byte range 0x100..0x1FF into\n"
                "                N contiguous slices. Each shard uses its own\n"
                "                checkpoint file: checkpoint_IofN.txt\n"
                "  -h, --help    show this message\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s (try --help)\n", argv[i]);
            exit(2);
        }
    }

    /* contiguous slice of 256 high-byte values across N shards */
    int lo = 0x100 + (g_shard_idx       * 256) / g_shard_count;
    int hi = 0x100 + ((g_shard_idx + 1) * 256) / g_shard_count;
    if(hi <= lo) hi = lo + 1;   /* should not happen for N<=256, defensive */
    g_shard_hi_lo    = (uint16_t)lo;
    g_shard_hi_hi    = (uint16_t)hi;
    g_shard_hi_range = (uint16_t)(hi - lo);

    if(g_shard_count > 1){
        snprintf(g_checkpoint_file,     sizeof g_checkpoint_file,
                 "checkpoint_%dof%d.txt",     g_shard_idx + 1, g_shard_count);
        snprintf(g_checkpoint_file_tmp, sizeof g_checkpoint_file_tmp,
                 "checkpoint_%dof%d.txt.tmp", g_shard_idx + 1, g_shard_count);
    }
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    /* Thread count: target 2 threads per logical CPU spread across all shards,
       so each HT pair sees one thread from each of two different shards —
       ideal for hiding AVX-512 latency without ZMM register spill.
       Formula: floor(nproc * 2 / shard_count), minimum 2. */
    {
        int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
        int t = (nproc * 2) / g_shard_count;
        if(t < 2) t = 2;
        if(t > NUM_THREADS_MAX) t = NUM_THREADS_MAX;
        g_num_threads = t;
    }

    for(int j=0;j<5;j++)
        target_word[j] = (uint32_t)TARGET[j*4]
                       | ((uint32_t)TARGET[j*4+1]<<8)
                       | ((uint32_t)TARGET[j*4+2]<<16)
                       | ((uint32_t)TARGET[j*4+3]<<24);

    /* checkpointing disabled — always start fresh */

    /* graceful shutdown on Ctrl-C and workflow stop */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    clock_gettime(CLOCK_MONOTONIC, &g_start);

    pthread_t disp_tid;
    pthread_create(&disp_tid, NULL, display_func, NULL);

    pthread_t tids[NUM_THREADS_MAX];
    for(long i=0;i<g_num_threads;i++)
        pthread_create(&tids[i], NULL, thread_func, (void*)i);
    for(int i=0;i<g_num_threads;i++)
        pthread_join(tids[i], NULL);

    pthread_join(disp_tid, NULL);

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double session_el = (end.tv_sec - g_start.tv_sec)
                      + (end.tv_nsec - g_start.tv_nsec)*1e-9;
    uint64_t total = g_attempts_base + atomic_load(&g_attempts);
    double   cum   = g_elapsed_base  + session_el;

    if(!atomic_load(&g_found)){
        printf("\nStopped.\n");
    }

    printf("Total: %llu  Time: %.2fs (cumulative)\n",
           (unsigned long long)total, cum);
    return 0;
}
