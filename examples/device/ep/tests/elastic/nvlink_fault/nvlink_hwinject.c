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
 *   nvlink_hwinject <gpu>                     capability probe, linkMask=0, no link touched
 *   nvlink_hwinject <gpu> down <link>         FORCE_LINK_DOWN on one link
 *   nvlink_hwinject <gpu> stomp <link> <n>    inject n stomped packets
 *   nvlink_hwinject <gpu> poison <link> <n>   inject n poisoned packets
 *
 * <gpu> is a GPU UUID as printed by `nvidia-smi -L`, or a device minor number.
 * Device minors need not follow nvidia-smi indices, so prefer the UUID.
 *
 * ABI from open-gpu-kernel-modules tags 570.158.01 and 580.173.02. r580 replaced the
 * NvU64 link masks of GET_NVLINK_CAPS and SET_HW_ERROR_INJECT with
 * NV2080_CTRL_NVLINK_LINK_MASK (unchanged through 615.71.09); the layout is picked at
 * runtime.
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
#include <strings.h>
#include <dirent.h>
#include <sys/ioctl.h>

#define NV_IOCTL_MAGIC           0x46
#define NV_ESC_RM_CONTROL        0x2a
#define NV_ESC_RM_ALLOC          0x2b
#define NV_ESC_CHECK_VERSION_STR 0xd2

#define NV01_ROOT        0x00000000
#define NV01_DEVICE_0    0x00000080
#define NV20_SUBDEVICE_0 0x00002080

#define CMD_GPU_GET_GID_INFO         0x2080014a
#define CMD_NVLINK_GET_NVLINK_CAPS   0x20803001
#define CMD_NVLINK_SET_HW_ERR_INJECT 0x20803081
#define CMD_NVLINK_GET_HW_ERR_INJECT 0x20803082

#define MAX_ARR     64
#define MAX_DEVICES 32

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

/* NV2080_CTRL_GPU_GET_GID_INFO_PARAMS; flags 0 selects the ASCII "GPU-..." form */
typedef struct {
    uint32_t index, flags, length;
    uint8_t  data[256];
} GID_INFO_PARAMS;

/* NV2080_CTRL_NVLINK_LINK_MASK (r580+) */
typedef struct {
    uint8_t  lenMasks;
    uint64_t masks[1] __attribute__((aligned(8)));
} LINK_MASK;

typedef struct {
    uint32_t capsTbl;
    uint8_t  lowestNvlinkVersion, highestNvlinkVersion;
    uint8_t  lowestNciVersion, highestNciVersion;
    uint32_t discoveredLinkMask, enabledLinkMask;
    uint64_t discoveredLinks __attribute__((aligned(8)));
    uint64_t enabledLinks;
} NVLINK_CAPS_PARAMS_R570;

typedef struct {
    uint32_t  capsTbl;
    uint8_t   lowestNvlinkVersion, highestNvlinkVersion;
    uint8_t   lowestNciVersion, highestNciVersion;
    uint32_t  discoveredLinkMask, enabledLinkMask;
    LINK_MASK discoveredLinks, enabledLinks;
} NVLINK_CAPS_PARAMS_R580;

/* NV2080_CTRL_NVLINK_HW_ERROR_INJECT_CFG: enum + NvU64 aligned 8 = 16 bytes */
typedef struct {
    uint32_t errType;
    uint32_t _pad;
    uint64_t errSettings;
} HW_ERR_INJECT_CFG;

typedef struct {
    uint64_t linkMask __attribute__((aligned(8)));
    HW_ERR_INJECT_CFG errCfg[MAX_ARR];
} SET_HW_ERR_INJECT_PARAMS_R570;

typedef struct {
    uint64_t  linkMask __attribute__((aligned(8)));
    LINK_MASK links;
    HW_ERR_INJECT_CFG errCfg[MAX_ARR];
} SET_HW_ERR_INJECT_PARAMS_R580;

_Static_assert(sizeof(NVLINK_CAPS_PARAMS_R570) == 32, "r570 GET_NVLINK_CAPS layout");
_Static_assert(sizeof(NVLINK_CAPS_PARAMS_R580) == 48, "r580 GET_NVLINK_CAPS layout");
_Static_assert(sizeof(SET_HW_ERR_INJECT_PARAMS_R570) == 1032, "r570 SET_HW_ERROR_INJECT layout");
_Static_assert(sizeof(SET_HW_ERR_INJECT_PARAMS_R580) == 1048, "r580 SET_HW_ERROR_INJECT layout");

static int ctlfd;
static uint32_t hClient, hSubDevice;
static int r580;

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

static int rmalloc(uint32_t hParent, uint32_t hNew, uint32_t hClass, void *p, size_t sz)
{
    NVOS21 a;
    memset(&a, 0, sizeof(a));
    a.hRoot = hClient;
    a.hObjectParent = hParent;
    a.hObjectNew = hNew;
    a.hClass = (int32_t)hClass;
    a.pAllocParms = (uint64_t)(uintptr_t)p;
    a.paramsSize = (uint32_t)sz;
    if (ioctl(ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21), &a))
        return 0xffff;
    return a.status;
}

/* Binds hSubDevice to RM device instance `device`. RM allocates a GPU only for a
 * process that holds its /dev/nvidiaN open, and the instance need not equal the minor. */
static int alloc_subdevice(uint32_t device)
{
    NV0080_ALLOC dev;
    memset(&dev, 0, sizeof(dev));
    dev.deviceId = device;
    dev.hClientShare = hClient;
    int st = rmalloc(hClient, 0xbeef0080 + device, NV01_DEVICE_0, &dev, sizeof(dev));
    if (st)
        return st;

    NV2080_ALLOC sub;
    memset(&sub, 0, sizeof(sub));
    st = rmalloc(0xbeef0080 + device, 0xbeef2080 + device, NV20_SUBDEVICE_0, &sub, sizeof(sub));
    if (st == 0)
        hSubDevice = 0xbeef2080 + device;
    return st;
}

static int subdevice_uuid(char *uuid, size_t size)
{
    GID_INFO_PARAMS g;
    int st;
    memset(&g, 0, sizeof(g));
    if (rmcontrol(CMD_GPU_GET_GID_INFO, &g, sizeof(g), &st))
        return 0xffff;
    if (st == 0)
        snprintf(uuid, size, "%.*s", (int)(g.length < sizeof(g.data) ? g.length : sizeof(g.data)),
                 (const char *)g.data);
    return st;
}

static int uuid_to_minor(const char *uuid)
{
    DIR *d = opendir("/proc/driver/nvidia/gpus");
    struct dirent *e;
    int minor = -1;
    while (d && minor < 0 && (e = readdir(d))) {
        char path[512], line[256], id[96] = "";
        int m = -1;
        if (e->d_name[0] == '.')
            continue;
        snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", e->d_name);
        FILE *f = fopen(path, "r");
        while (f && fgets(line, sizeof(line), f)) {
            sscanf(line, "GPU UUID: %95s", id);
            sscanf(line, "Device Minor: %d", &m);
        }
        if (f)
            fclose(f);
        if (strcasecmp(id, uuid) == 0)
            minor = m;
    }
    if (d)
        closedir(d);
    return minor;
}

/* RM rejects a params struct of the wrong size with NV_ERR_INVALID_ARGUMENT before the
 * control runs, so this read-only query picks the layout for both controls. */
static int get_nvlink_caps(unsigned *version, uint64_t *enabled)
{
    int st;
    NVLINK_CAPS_PARAMS_R580 c580;
    memset(&c580, 0, sizeof(c580));
    if (rmcontrol(CMD_NVLINK_GET_NVLINK_CAPS, &c580, sizeof(c580), &st))
        return 0xffff;
    if (st == 0) {
        r580 = 1;
        *version = c580.highestNvlinkVersion;
        *enabled = c580.enabledLinks.masks[0];
        return 0;
    }
    if (st != 0x1f && st != 0x3a)
        return st;

    NVLINK_CAPS_PARAMS_R570 c570;
    memset(&c570, 0, sizeof(c570));
    if (rmcontrol(CMD_NVLINK_GET_NVLINK_CAPS, &c570, sizeof(c570), &st))
        return 0xffff;
    *version = c570.highestNvlinkVersion;
    *enabled = c570.enabledLinks;
    return st;
}

static size_t set_hw_err_inject_size(void)
{
    return r580 ? sizeof(SET_HW_ERR_INJECT_PARAMS_R580) : sizeof(SET_HW_ERR_INJECT_PARAMS_R570);
}

static int set_hw_err_inject(uint64_t linkMask, const HW_ERR_INJECT_CFG *errCfg, int *status)
{
    static SET_HW_ERR_INJECT_PARAMS_R580 p580;
    static SET_HW_ERR_INJECT_PARAMS_R570 p570;

    if (r580) {
        /* GSP-RM may read either mask field; RM's own helpers OR them together. */
        p580.linkMask = linkMask;
        p580.links.lenMasks = 1;
        p580.links.masks[0] = linkMask;
        memcpy(p580.errCfg, errCfg, sizeof(p580.errCfg));
        return rmcontrol(CMD_NVLINK_SET_HW_ERR_INJECT, &p580, sizeof(p580), status);
    }
    p570.linkMask = linkMask;
    memcpy(p570.errCfg, errCfg, sizeof(p570.errCfg));
    return rmcontrol(CMD_NVLINK_SET_HW_ERR_INJECT, &p570, sizeof(p570), status);
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

static double seconds(struct timespec t)
{
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void print_wall_clock(struct timespec started, struct timespec begin, struct timespec end)
{
    printf("  wall clock: injector start %.6f, ioctl begin %.6f, ioctl end %.6f\n",
           seconds(started), seconds(begin), seconds(end));
}

int main(int argc, char **argv)
{
    struct timespec started;
    clock_gettime(CLOCK_REALTIME, &started);
    const char *gpu = (argc > 1) ? argv[1] : "2";
    const char *mode = (argc > 2) ? argv[2] : "probe";
    int link = (argc > 3) ? atoi(argv[3]) : -1;
    unsigned count = (argc > 4) ? (unsigned)atoi(argv[4]) : 16;
    int by_uuid = strncasecmp(gpu, "GPU-", 4) == 0;
    int minor = by_uuid ? uuid_to_minor(gpu) : atoi(gpu);
    if (minor < 0) { fprintf(stderr, "%s not found in /proc/driver/nvidia/gpus\n", gpu); return 1; }

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
    if (gpufd < 0) { perror(devpath); return 1; }

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

    /* Only the open node's GPU can allocate; the UUID check guards the match. */
    int device = -1, st = 0;
    char uuid[96] = "";
    for (int i = 0; i < MAX_DEVICES && device < 0; i++) {
        if (!by_uuid && i != minor)
            continue;
        st = alloc_subdevice((uint32_t)i);
        if (st == 0)
            st = subdevice_uuid(uuid, sizeof(uuid));
        if (st == 0 && (!by_uuid || strcasecmp(uuid, gpu) == 0))
            device = i;
    }
    if (device < 0) {
        fprintf(stderr, "no RM device matches %s (minor %d): status=0x%04x %s\n",
                gpu, minor, st, nvstatus(st));
        return 1;
    }

    unsigned nvlink_version = 0;
    uint64_t enabled_links = 0;
    st = get_nvlink_caps(&nvlink_version, &enabled_links);
    if (st) {
        fprintf(stderr, "GET_NVLINK_CAPS failed: status=0x%04x %s\n", st, nvstatus(st));
        return 1;
    }
    printf("GPU %s (minor %d, RM device %d): NVLink v%u, enabled links 0x%llx (%s params layout)\n\n",
           uuid, minor, device, nvlink_version, (unsigned long long)enabled_links,
           r580 ? "r580+" : "r570");

    HW_ERR_INJECT_CFG errCfg[MAX_ARR];
    memset(errCfg, 0, sizeof(errCfg));

    if (strcmp(mode, "probe") == 0) {
        /* linkMask 0 targets no link at all: pure capability probe. */
        struct timespec wall0, wall1;
        clock_gettime(CLOCK_REALTIME, &wall0);
        int r = set_hw_err_inject(0, errCfg, &st);
        clock_gettime(CLOCK_REALTIME, &wall1);
        printf("capability probe (linkMask=0, no link touched):\n");
        printf("  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s (size=%zu)\n",
               r, st, nvstatus(st), set_hw_err_inject_size());
        print_wall_clock(started, wall0, wall1);
        if (r || st == 0x56 || st == 0xffff) {
            printf("\n  => NOT usable on this driver.\n");
            return 2;
        }
        if (st == 0x1b) {
            printf("\n  => params accepted, but the control is privileged: rerun as root.\n");
            return 4;
        }
        if (st != 0 && st != 0x3b) {
            printf("\n  => unexpected status; the HW path is unconfirmed.\n");
            return 5;
        }
        printf("\n  => reachable; the HW path may work here\n");
        return 0;
    }

    if (link < 0 || link >= MAX_ARR || !((enabled_links >> link) & 1)) {
        fprintf(stderr, "link %d is not enabled on this GPU\n", link);
        return 1;
    }

    if (strcmp(mode, "down") == 0) {
        errCfg[link].errType = ERR_TYPE_LINK_ERR;
        errCfg[link].errSettings = LINK_ERR_FORCE_DOWN;
        printf("*** FORCE_LINK_DOWN on GPU %s link %d ***\n", uuid, link);
    } else if (strcmp(mode, "stomp") == 0) {
        errCfg[link].errType = ERR_TYPE_PKT_ERR;
        errCfg[link].errSettings = PKT_ERR_STOMP | (count & 0xffff);
        printf("*** STOMP %u packets on GPU %s link %d ***\n", count, uuid, link);
    } else if (strcmp(mode, "poison") == 0) {
        errCfg[link].errType = ERR_TYPE_PKT_ERR;
        errCfg[link].errSettings = PKT_ERR_POISON | (count & 0xffff);
        printf("*** POISON %u packets on GPU %s link %d ***\n", count, uuid, link);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 1;
    }

    kmsg("NVLINK_HWINJECT_MARK begin gpu=%s link=%d mode=%s\n", uuid, link, mode);
    struct timespec t0, t1, wall0, wall1;
    clock_gettime(CLOCK_REALTIME, &wall0);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = set_hw_err_inject(1ull << link, errCfg, &st);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    clock_gettime(CLOCK_REALTIME, &wall1);
    kmsg("NVLINK_HWINJECT_MARK end status=0x%x\n", st);

    printf("  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s\n", r, st, nvstatus(st));
    printf("  ioctl took %.3f ms\n", (seconds(t1) - seconds(t0)) * 1e3);
    print_wall_clock(started, wall0, wall1);

    if (gpufd >= 0) close(gpufd);
    close(ctlfd);
    return (st == 0) ? 0 : 3;
}
