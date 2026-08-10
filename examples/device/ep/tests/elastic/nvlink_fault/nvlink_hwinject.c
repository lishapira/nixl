/*
 * nvlink_hwinject — the NVL5 *hardware* NVLink error-injection path,
 * NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT (0x20803081).
 *
 * This is a different mechanism from nvlink_swinject (INJECT_SW_ERROR,
 * 0x20803089). The SW path raises a RAS event, and for fatal severities the
 * driver defers the actual link teardown via "Drain and Reset" until the GPU
 * goes idle -- which never happens while a workload is running. The HW path is
 * supposed to act on the link directly, including FORCE_LINK_DOWN, so it is the
 * candidate for taking a link down *during* live traffic.
 *
 * Usage:
 *   nvlink_hwinject <gpu_minor>                     capability probe, linkMask=0, no link touched
 *   nvlink_hwinject <gpu_minor> down <link>         FORCE_LINK_DOWN on one link
 *   nvlink_hwinject <gpu_minor> stomp <link> <n>    inject n stomped packets
 *   nvlink_hwinject <gpu_minor> poison <link> <n>   inject n poisoned packets
 *
 * ABI from open-gpu-kernel-modules tag 570.158.01.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>
#include <sys/ioctl.h>

#define NV_IOCTL_MAGIC           0x46
#define NV_ESC_RM_CONTROL        0x2a
#define NV_ESC_RM_ALLOC          0x2b
#define NV_ESC_CHECK_VERSION_STR 0xd2

#define NV01_ROOT        0x00000000
#define NV01_DEVICE_0    0x00000080
#define NV20_SUBDEVICE_0 0x00002080

#define CMD_NVLINK_GET_NVLINK_CAPS   0x20803001
#define CMD_NVLINK_SET_HW_ERR_INJECT 0x20803081
#define CMD_NVLINK_GET_HW_ERR_INJECT 0x20803082

#define MAX_ARR 64

/* NV2080_CTRL_NVLINK_HW_ERROR_INJECT_ERR_TYPE */
#define ERR_TYPE_TX_ERR       1
#define ERR_TYPE_PKT_ERR      2
#define ERR_TYPE_AUTH_TAG_ERR 3
#define ERR_TYPE_LINK_ERR     4

/* PKT_ERR errSettings: 15:0 count, bit16 STOMP, bit17 POISON, bit18 clear */
#define PKT_ERR_STOMP  (1u << 16)
#define PKT_ERR_POISON (1u << 17)
/* LINK_ERR errSettings: bit0 force link down */
#define LINK_ERR_FORCE_DOWN 1u

typedef struct { uint32_t cmd, reply; char versionString[64]; } rm_api_version_t;

typedef struct {
    uint32_t hRoot, hObjectParent, hObjectNew;
    int32_t  hClass;
    uint64_t pAllocParms __attribute__((aligned(8)));
    uint32_t paramsSize;
    int32_t  status;
} NVOS21;

typedef struct {
    uint32_t hClient, hObject;
    int32_t  cmd;
    uint32_t flags;
    uint64_t params __attribute__((aligned(8)));
    uint32_t paramsSize;
    int32_t  status;
} NVOS54;

typedef struct {
    uint32_t deviceId, hClientShare, hTargetClient, hTargetDevice;
    int32_t  flags;
    uint64_t vaSpaceSize __attribute__((aligned(8)));
    uint64_t vaStartInternal, vaLimitInternal;
    int32_t  vaMode;
} NV0080_ALLOC;

typedef struct { uint32_t subDeviceId; } NV2080_ALLOC;

typedef struct {
    uint32_t capsTbl;
    uint8_t  lowestNvlinkVersion, highestNvlinkVersion;
    uint8_t  lowestNciVersion, highestNciVersion;
    uint32_t discoveredLinkMask, enabledLinkMask;
    uint64_t discoveredLinks __attribute__((aligned(8)));
    uint64_t enabledLinks;
} NVLINK_CAPS_PARAMS;

/* NV2080_CTRL_NVLINK_HW_ERROR_INJECT_CFG: enum + NvU64 aligned 8 = 16 bytes */
typedef struct {
    uint32_t errType;
    uint32_t _pad;
    uint64_t errSettings;
} HW_ERR_INJECT_CFG;

typedef struct {
    uint64_t linkMask __attribute__((aligned(8)));
    HW_ERR_INJECT_CFG errCfg[MAX_ARR];
} SET_HW_ERR_INJECT_PARAMS;

static int ctlfd;
static uint32_t hClient, hSubDevice;

static const char *nvstatus(int s)
{
    switch (s) {
    case 0x0000: return "NV_OK";
    case 0x001b: return "NV_ERR_INSUFFICIENT_PERMISSIONS";
    case 0x001f: return "NV_ERR_INVALID_ARGUMENT";
    case 0x003a: return "NV_ERR_INVALID_PARAM_STRUCT";
    case 0x003b: return "NV_ERR_INVALID_PARAMETER";
    case 0x0040: return "NV_ERR_INVALID_STATE";
    case 0x0056: return "NV_ERR_NOT_SUPPORTED";
    case 0xffff: return "NV_ERR_GENERIC";
    }
    return "(unlisted)";
}

static int rmcontrol(uint32_t cmd, void *p, size_t sz, int *status)
{
    NVOS54 c;
    memset(&c, 0, sizeof(c));
    c.hClient = hClient;
    c.hObject = hSubDevice;
    c.cmd = (int32_t)cmd;
    c.params = (uint64_t)(uintptr_t)p;
    c.paramsSize = (uint32_t)sz;
    int r = ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54), &c);
    *status = c.status;
    return r;
}

static void kmsg(const char *fmt, ...)
{
    int fd = open("/dev/kmsg", O_WRONLY);
    if (fd < 0) return;
    va_list ap;
    va_start(ap, fmt);
    vdprintf(fd, fmt, ap);
    va_end(ap);
    close(fd);
}

int main(int argc, char **argv)
{
    int minor = (argc > 1) ? atoi(argv[1]) : 2;
    const char *mode = (argc > 2) ? argv[2] : "probe";
    int link = (argc > 3) ? atoi(argv[3]) : -1;
    unsigned count = (argc > 4) ? (unsigned)atoi(argv[4]) : 16;

    char version[64] = {0};
    FILE *f = fopen("/proc/driver/nvidia/version", "r");
    char line[512];
    if (f && fgets(line, sizeof(line), f)) {
        char *p = strstr(line, "x86_64");
        if (p) sscanf(p + 6, "%63s", version);
    }
    if (f) fclose(f);

    ctlfd = open("/dev/nvidiactl", O_RDWR);
    if (ctlfd < 0) { perror("open /dev/nvidiactl"); return 1; }
    char devpath[64];
    snprintf(devpath, sizeof(devpath), "/dev/nvidia%d", minor);
    int gpufd = open(devpath, O_RDWR);

    rm_api_version_t ver;
    memset(&ver, 0, sizeof(ver));
    snprintf(ver.versionString, sizeof(ver.versionString), "%s", version);
    if (ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_CHECK_VERSION_STR, rm_api_version_t), &ver)) {
        fprintf(stderr, "version handshake rejected\n"); return 1;
    }

    NVOS21 a;
    memset(&a, 0, sizeof(a));
    a.hClass = NV01_ROOT;
    if (ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21), &a) || a.status) {
        fprintf(stderr, "client alloc failed 0x%x\n", a.status); return 1;
    }
    hClient = a.hObjectNew;

    NV0080_ALLOC dev;
    memset(&dev, 0, sizeof(dev));
    dev.deviceId = (uint32_t)minor;
    dev.hClientShare = hClient;
    memset(&a, 0, sizeof(a));
    a.hRoot = a.hObjectParent = hClient;
    a.hObjectNew = 0xbeef0080;
    a.hClass = NV01_DEVICE_0;
    a.pAllocParms = (uint64_t)(uintptr_t)&dev;
    a.paramsSize = sizeof(dev);
    if (ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21), &a) || a.status) {
        fprintf(stderr, "device alloc failed 0x%x\n", a.status); return 1;
    }
    uint32_t hDevice = a.hObjectNew;

    NV2080_ALLOC sub;
    memset(&sub, 0, sizeof(sub));
    memset(&a, 0, sizeof(a));
    a.hRoot = hClient;
    a.hObjectParent = hDevice;
    a.hObjectNew = 0xbeef2080;
    a.hClass = NV20_SUBDEVICE_0;
    a.pAllocParms = (uint64_t)(uintptr_t)&sub;
    a.paramsSize = sizeof(sub);
    if (ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21), &a) || a.status) {
        fprintf(stderr, "subdevice alloc failed 0x%x\n", a.status); return 1;
    }
    hSubDevice = a.hObjectNew;

    int st;
    NVLINK_CAPS_PARAMS caps;
    memset(&caps, 0, sizeof(caps));
    rmcontrol(CMD_NVLINK_GET_NVLINK_CAPS, &caps, sizeof(caps), &st);
    printf("GPU minor %d: NVLink v%u, enabled links 0x%llx\n\n",
           minor, caps.highestNvlinkVersion, (unsigned long long)caps.enabledLinks);

    SET_HW_ERR_INJECT_PARAMS sp;
    memset(&sp, 0, sizeof(sp));

    if (strcmp(mode, "probe") == 0) {
        /* linkMask 0 targets no link at all: pure capability probe. */
        sp.linkMask = 0;
        int r = rmcontrol(CMD_NVLINK_SET_HW_ERR_INJECT, &sp, sizeof(sp), &st);
        printf("capability probe (linkMask=0, no link touched):\n");
        printf("  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s (size=%zu)\n",
               r, st, nvstatus(st), sizeof(sp));
        if (st == 0x56 || st == 0xffff)
            printf("\n  => NOT usable on this driver.\n");
        else
            printf("\n  => reachable; the HW path may work here\n");
        return (st == 0x56 || st == 0xffff) ? 2 : 0;
    }

    if (link < 0 || !((caps.enabledLinks >> link) & 1)) {
        fprintf(stderr, "link %d is not enabled on this GPU\n", link);
        return 1;
    }

    sp.linkMask = 1ull << link;
    if (strcmp(mode, "down") == 0) {
        sp.errCfg[link].errType = ERR_TYPE_LINK_ERR;
        sp.errCfg[link].errSettings = LINK_ERR_FORCE_DOWN;
        printf("*** FORCE_LINK_DOWN on GPU %d link %d ***\n", minor, link);
    } else if (strcmp(mode, "stomp") == 0) {
        sp.errCfg[link].errType = ERR_TYPE_PKT_ERR;
        sp.errCfg[link].errSettings = PKT_ERR_STOMP | (count & 0xffff);
        printf("*** STOMP %u packets on GPU %d link %d ***\n", count, minor, link);
    } else if (strcmp(mode, "poison") == 0) {
        sp.errCfg[link].errType = ERR_TYPE_PKT_ERR;
        sp.errCfg[link].errSettings = PKT_ERR_POISON | (count & 0xffff);
        printf("*** POISON %u packets on GPU %d link %d ***\n", count, minor, link);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 1;
    }

    kmsg("NVLINK_HWINJECT_MARK begin gpu=%d link=%d mode=%s\n", minor, link, mode);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = rmcontrol(CMD_NVLINK_SET_HW_ERR_INJECT, &sp, sizeof(sp), &st);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    kmsg("NVLINK_HWINJECT_MARK end status=0x%x\n", st);

    printf("  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s\n", r, st, nvstatus(st));
    printf("  ioctl took %.3f ms\n",
           ((t1.tv_sec + t1.tv_nsec / 1e9) - (t0.tv_sec + t0.tv_nsec / 1e9)) * 1e3);

    if (gpufd >= 0) close(gpufd);
    close(ctlfd);
    return (st == 0) ? 0 : 3;
}
