/* 不良メモリ領域の物理アドレスを特定する。
 *
 * `/proc/self/pagemap` から物理ページ番号を読むため、**ホスト側でrootとして**
 * 実行すること。コンテナ内や一般ユーザでは、ページ番号が0にマスクされて
 * 物理アドレスが得られない。
 *
 *   gcc -O2 -o phys phys.c
 *   sudo ./phys [MB]        既定: 7200
 *
 * 出力された物理アドレスは、そのページをカーネルに使わせないために使う。
 *
 *   再起動なしで試す(再起動すると元に戻る):
 *     echo 0x<物理アドレス> | sudo tee /sys/devices/system/memory/soft_offline_page
 *
 *   恒久化する(/etc/default/grub の GRUB_CMDLINE_LINUX_DEFAULT に追記 -> update-grub):
 *     memmap=<サイズ>K$0x<開始アドレス>
 *
 * memmap= の書式を誤ると起動しなくなり得るので、遠隔で再投入する前に必ず確認すること。
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PAGEMAP_PRESENT (1ULL << 63)
#define PAGEMAP_PFN_MASK ((1ULL << 55) - 1)

/* 仮想アドレスに対応する物理アドレス。読めない場合は0。 */
static uint64_t physical_address(int pagemap_fd, void *addr) {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t entry = 0;
    uint64_t virtual_page = (uint64_t)addr / page_size;
    if (pread(pagemap_fd, &entry, sizeof(entry), virtual_page * sizeof(entry)) != sizeof(entry))
        return 0;
    if (!(entry & PAGEMAP_PRESENT)) return 0;
    return (entry & PAGEMAP_PFN_MASK) * page_size + (uint64_t)addr % page_size;
}

static inline uint64_t next(uint64_t v) {
    v ^= v << 13;
    v ^= v >> 7;
    v ^= v << 17;
    return v;
}

int main(int argc, char **argv) {
    int mb = argc > 1 ? atoi(argv[1]) : 7200;
    size_t n = (size_t)mb * 1024 * 1024 / sizeof(uint64_t);
    uint64_t *buf = malloc(n * sizeof(uint64_t));
    if (!buf) {
        printf("%dMB の確保に失敗\n", mb);
        return 2;
    }
    int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pagemap_fd < 0) {
        printf("/proc/self/pagemap を開けません。rootで実行してください\n");
        return 3;
    }

    const uint64_t seed = 0x9E3779B97F4A7C15ULL;
    uint64_t v = seed;
    for (size_t i = 0; i < n; i++) buf[i] = (v = next(v));

    /* 物理アドレスは連続しないので、ページ単位で最小・最大を集計する。 */
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t lowest = 0, highest = 0;
    long count = 0, shown = 0;
    v = seed;
    for (size_t i = 0; i < n; i++) {
        v = next(v);
        if (buf[i] == v) continue;
        uint64_t physical = physical_address(pagemap_fd, &buf[i]);
        if (!physical) {
            printf("物理アドレスを読めません(権限不足)。ホスト側でrootとして実行してください\n");
            return 3;
        }
        if (!count || physical < lowest) lowest = physical;
        if (physical > highest) highest = physical;
        count++;
        if (shown++ < 5)
            printf("  仮想 %p -> 物理 0x%llx\n", (void *)&buf[i], (unsigned long long)physical);
    }
    close(pagemap_fd);

    if (!count) {
        printf("%dMB: 不一致なし(この確保量では不良領域に触れていない)\n", mb);
        free(buf);
        return 0;
    }

    uint64_t first_page = lowest & ~(uint64_t)(page_size - 1);
    uint64_t last_page = highest & ~(uint64_t)(page_size - 1);
    uint64_t span_kb = (last_page - first_page + page_size) / 1024;
    printf("\n不一致 %ld 件\n", count);
    printf("物理アドレス範囲: 0x%llx 〜 0x%llx\n", (unsigned long long)lowest,
           (unsigned long long)highest);
    printf("該当ページ: 0x%llx から %llu KB (%llu ページ)\n", (unsigned long long)first_page,
           (unsigned long long)span_kb, (unsigned long long)(span_kb * 1024 / page_size));
    printf("\n再起動せずに無効化する:\n");
    for (uint64_t p = first_page; p <= last_page; p += page_size)
        printf("  echo 0x%llx | sudo tee /sys/devices/system/memory/soft_offline_page\n",
               (unsigned long long)p);
    printf("\n恒久化する(GRUB_CMDLINE_LINUX_DEFAULT に追記):\n");
    printf("  memmap=%lluK$0x%llx\n", (unsigned long long)span_kb,
           (unsigned long long)first_page);
    free(buf);
    return 1;
}
