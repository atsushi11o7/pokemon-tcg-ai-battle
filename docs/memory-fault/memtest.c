/* メモリの書き込み/読み戻しを検証し、不良領域の有無を確認する。
 *
 * 学習中に SIGSEGV / corrupted size vs. prev_size / 異常に巨大な整数 が頻発した際、
 * 原因の切り分けに使ったもの。確保量が閾値を超えると必ず壊れる、という形で
 * 不良領域を検出する。
 *
 *   gcc -O2 -pthread -o memtest memtest.c
 *   ./memtest [スレッド数] [1スレッドあたりMB]      既定: 1 7200
 *
 * 「不一致 0 件」なら、その確保量では不良領域に触れていない。
 * BIOS で XMP/EXPO を切り替えた前後で同じ条件を流せば、設定起因かを判別できる。
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int threads = 1;
static int mb_per_thread = 7200;
static const int passes = 6;

static volatile long errors = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* xorshift64。同じseedから同じ列を再生成できるので、書いた値と読んだ値を
   バッファを2つ持たずに比較できる。 */
static inline uint64_t next(uint64_t v) {
    v ^= v << 13;
    v ^= v >> 7;
    v ^= v << 17;
    return v;
}

static void *worker(void *arg) {
    long id = (long)arg;
    size_t n = (size_t)mb_per_thread * 1024 * 1024 / sizeof(uint64_t);
    uint64_t *buf = malloc(n * sizeof(uint64_t));
    if (!buf) {
        fprintf(stderr, "  thread %ld: %dMB の確保に失敗\n", id, mb_per_thread);
        return NULL;
    }
    for (int pass = 0; pass < passes; pass++) {
        uint64_t seed = 0x9E3779B97F4A7C15ULL * (pass + 1) + id;
        uint64_t v = seed;
        for (size_t i = 0; i < n; i++) buf[i] = (v = next(v));
        v = seed;
        for (size_t i = 0; i < n; i++) {
            v = next(v);
            if (buf[i] == v) continue;
            pthread_mutex_lock(&lock);
            if (++errors <= 5)
                fprintf(stderr, "  不一致 thread=%ld pass=%d offset=%zu 期待=%llx 実際=%llx\n",
                        id, pass, i, (unsigned long long)v, (unsigned long long)buf[i]);
            pthread_mutex_unlock(&lock);
        }
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) mb_per_thread = atoi(argv[2]);
    pthread_t *t = malloc(sizeof(pthread_t) * threads);
    if (!t) return 2;
    printf("%dスレッド x %dMB x %dパス (合計 %.1f GB を読み書き)\n", threads, mb_per_thread,
           passes, (double)threads * mb_per_thread * passes / 1024.0);
    fflush(stdout);
    time_t start = time(NULL);
    for (long i = 0; i < threads; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    printf("完了: %ld秒  不一致 %ld 件\n", (long)(time(NULL) - start), errors);
    free(t);
    return errors ? 1 : 0;
}
