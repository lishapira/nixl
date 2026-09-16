// SPDX-License-Identifier: Apache-2.0
/*
 * R580 NVLink hardware injector for GB300.
 *
 * Usage:
 *   nvlink_hwinject <gpu_minor>                  harmless linkMask=0 probe
 *   nvlink_hwinject <gpu_minor> down <link>      destructive FORCE_LINK_DOWN
 *
 * The RM ABI was transcribed from NVIDIA's open kernel modules. Revalidate
 * these layouts when changing driver branches.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define NV_IOCTL_MAGIC           0x46
#define NV_ESC_RM_CONTROL        0x2a
#define NV_ESC_RM_ALLOC          0x2b
#define NV_ESC_CHECK_VERSION_STR 0xd2

#define NV01_ROOT        0x00000000
#define NV01_DEVICE_0    0x00000080
#define NV20_SUBDEVICE_0 0x00002080

#define CMD_NVLINK_SET_HW_ERR_INJECT 0x20803081
#define MAX_ARR 64
#define MAX_LINKS 18

#define ERR_TYPE_LINK_ERR 4
#define LINK_ERR_FORCE_DOWN 1u

typedef struct {
    uint32_t cmd;
    uint32_t reply;
    char versionString[64];
} rm_api_version_t;

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectNew;
    int32_t hClass;
    uint64_t pAllocParms __attribute__((aligned(8)));
    uint32_t paramsSize;
    int32_t status;
} NVOS21;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    int32_t cmd;
    uint32_t flags;
    uint64_t params __attribute__((aligned(8)));
    uint32_t paramsSize;
    int32_t status;
} NVOS54;

typedef struct {
    uint32_t deviceId;
    uint32_t hClientShare;
    uint32_t hTargetClient;
    uint32_t hTargetDevice;
    int32_t flags;
    uint64_t vaSpaceSize __attribute__((aligned(8)));
    uint64_t vaStartInternal;
    uint64_t vaLimitInternal;
    int32_t vaMode;
} NV0080_ALLOC;

typedef struct {
    uint32_t subDeviceId;
} NV2080_ALLOC;

typedef struct {
    uint32_t errType;
    uint32_t _pad;
    uint64_t errSettings;
} HW_ERR_INJECT_CFG;

/* R580 NV2080_CTRL_NVLINK_LINK_MASK: 1 byte + padding + one NvU64. */
typedef struct {
    uint8_t lenMasks;
    uint64_t masks[1] __attribute__((aligned(8)));
} NVLINK_LINK_MASK;

/* R580 adds links after the legacy linkMask: 8 + 16 + 64 * 16 = 1048. */
typedef struct {
    uint64_t linkMask __attribute__((aligned(8)));
    NVLINK_LINK_MASK links __attribute__((aligned(8)));
    HW_ERR_INJECT_CFG errCfg[MAX_ARR];
} SET_HW_ERR_INJECT_PARAMS;

_Static_assert(sizeof(HW_ERR_INJECT_CFG) == 16,
               "unexpected HW_ERR_INJECT_CFG ABI size");
_Static_assert(sizeof(NVLINK_LINK_MASK) == 16,
               "unexpected NVLINK_LINK_MASK ABI size");
_Static_assert(sizeof(SET_HW_ERR_INJECT_PARAMS) == 1048,
               "unexpected R580 SET_HW_ERROR_INJECT ABI size");

static int ctlfd;
static uint32_t hClient;
static uint32_t hSubDevice;

static const char *nvstatus(int status)
{
    switch (status) {
    case 0x0000:
        return "NV_OK";
    case 0x001b:
        return "NV_ERR_INSUFFICIENT_PERMISSIONS";
    case 0x001f:
        return "NV_ERR_INVALID_ARGUMENT";
    case 0x003a:
        return "NV_ERR_INVALID_PARAM_STRUCT";
    case 0x003b:
        return "NV_ERR_INVALID_PARAMETER";
    case 0x0040:
        return "NV_ERR_INVALID_STATE";
    case 0x0056:
        return "NV_ERR_NOT_SUPPORTED";
    case 0xffff:
        return "NV_ERR_GENERIC";
    default:
        return "(unlisted)";
    }
}

static int rmcontrol(uint32_t cmd, void *params, size_t size, int *status)
{
    NVOS54 control;

    memset(&control, 0, sizeof(control));
    control.hClient = hClient;
    control.hObject = hSubDevice;
    control.cmd = (int32_t)cmd;
    control.params = (uint64_t)(uintptr_t)params;
    control.paramsSize = (uint32_t)size;

    int result = ioctl(
        ctlfd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54), &control
    );
    *status = control.status;
    return result;
}

static void kmsg(const char *format, ...)
{
    int fd = open("/dev/kmsg", O_WRONLY);
    va_list args;

    if (fd < 0)
        return;
    va_start(args, format);
    vdprintf(fd, format, args);
    va_end(args);
    close(fd);
}

static int get_driver_version(char *version, size_t size)
{
    FILE *file = fopen("/proc/driver/nvidia/version", "r");
    char line[512];
    char *token;
    unsigned int major;
    unsigned int minor;
    unsigned int patch;

    if (file == NULL || fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL)
            fclose(file);
        return -1;
    }
    fclose(file);

    token = strtok(line, " \t\r\n");
    while (token != NULL) {
        if (sscanf(token, "%u.%u.%u", &major, &minor, &patch) == 3) {
            snprintf(version, size, "%s", token);
            return 0;
        }
        token = strtok(NULL, " \t\r\n");
    }
    return -1;
}

int main(int argc, char **argv)
{
    int minor = argc > 1 ? atoi(argv[1]) : 2;
    const char *mode = argc > 2 ? argv[2] : "probe";
    int link = argc > 3 ? atoi(argv[3]) : -1;
    char version[64] = {0};
    char devpath[64];
    int gpufd;
    int status;

    if (get_driver_version(version, sizeof(version)) != 0) {
        fprintf(stderr, "could not parse NVIDIA driver version\n");
        return 1;
    }
    fprintf(stderr, "driver version string: '%s'\n", version);

    ctlfd = open("/dev/nvidiactl", O_RDWR);
    if (ctlfd < 0) {
        perror("open /dev/nvidiactl");
        return 1;
    }
    snprintf(devpath, sizeof(devpath), "/dev/nvidia%d", minor);
    gpufd = open(devpath, O_RDWR);
    if (gpufd < 0) {
        perror(devpath);
        close(ctlfd);
        return 1;
    }

    rm_api_version_t api_version;
    memset(&api_version, 0, sizeof(api_version));
    snprintf(
        api_version.versionString,
        sizeof(api_version.versionString),
        "%s",
        version
    );
    if (ioctl(
            ctlfd,
            _IOWR(NV_IOCTL_MAGIC, NV_ESC_CHECK_VERSION_STR, rm_api_version_t),
            &api_version
        ) != 0) {
        fprintf(stderr, "version handshake rejected\n");
        return 1;
    }

    NVOS21 allocation;
    memset(&allocation, 0, sizeof(allocation));
    allocation.hClass = NV01_ROOT;
    if (ioctl(
            ctlfd,
            _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21),
            &allocation
        ) != 0 ||
        allocation.status != 0) {
        fprintf(stderr, "client alloc failed 0x%x\n", allocation.status);
        return 1;
    }
    hClient = allocation.hObjectNew;

    NV0080_ALLOC device;
    memset(&device, 0, sizeof(device));
    device.deviceId = (uint32_t)minor;
    device.hClientShare = hClient;
    memset(&allocation, 0, sizeof(allocation));
    allocation.hRoot = hClient;
    allocation.hObjectParent = hClient;
    allocation.hObjectNew = 0xbeef0080;
    allocation.hClass = NV01_DEVICE_0;
    allocation.pAllocParms = (uint64_t)(uintptr_t)&device;
    allocation.paramsSize = sizeof(device);
    if (ioctl(
            ctlfd,
            _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21),
            &allocation
        ) != 0 ||
        allocation.status != 0) {
        fprintf(stderr, "device alloc failed 0x%x\n", allocation.status);
        return 1;
    }
    uint32_t hDevice = allocation.hObjectNew;

    NV2080_ALLOC subdevice;
    memset(&subdevice, 0, sizeof(subdevice));
    memset(&allocation, 0, sizeof(allocation));
    allocation.hRoot = hClient;
    allocation.hObjectParent = hDevice;
    allocation.hObjectNew = 0xbeef2080;
    allocation.hClass = NV20_SUBDEVICE_0;
    allocation.pAllocParms = (uint64_t)(uintptr_t)&subdevice;
    allocation.paramsSize = sizeof(subdevice);
    if (ioctl(
            ctlfd,
            _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21),
            &allocation
        ) != 0 ||
        allocation.status != 0) {
        fprintf(stderr, "subdevice alloc failed 0x%x\n", allocation.status);
        return 1;
    }
    hSubDevice = allocation.hObjectNew;

    printf("GPU minor %d\n\n", minor);
    SET_HW_ERR_INJECT_PARAMS params;
    memset(&params, 0, sizeof(params));

    if (strcmp(mode, "probe") == 0) {
        int result = rmcontrol(
            CMD_NVLINK_SET_HW_ERR_INJECT, &params, sizeof(params), &status
        );
        printf("capability probe (linkMask=0, no link touched):\n");
        printf(
            "  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s (size=%zu)\n",
            result,
            status,
            nvstatus(status),
            sizeof(params)
        );
        if (result != 0) {
            perror("NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT");
            return 2;
        }
        if (status == 0x1b) {
            printf("  => ABI accepted, but host root is required.\n");
            return 2;
        }
        if (status == 0x56 || status == 0xffff || status == 0x3a) {
            printf("  => control is not usable with this driver/ABI.\n");
            return 2;
        }
        printf("  => control and ABI are reachable; no link was touched.\n");
        return status == 0 ? 0 : 2;
    }

    if (strcmp(mode, "down") != 0) {
        fprintf(stderr, "only probe and down modes are supported\n");
        return 1;
    }
    if (link < 0 || link >= MAX_LINKS) {
        fprintf(stderr, "link %d out of range [0,%d)\n", link, MAX_LINKS);
        return 1;
    }

    params.linkMask = 1ull << link;
    params.links.lenMasks = 1;
    params.links.masks[0] = 1ull << link;
    params.errCfg[link].errType = ERR_TYPE_LINK_ERR;
    params.errCfg[link].errSettings = LINK_ERR_FORCE_DOWN;

    printf("*** FORCE_LINK_DOWN on GPU %d link %d ***\n", minor, link);
    kmsg(
        "NVLINK_HWINJECT_MARK begin gpu=%d link=%d mode=%s\n",
        minor,
        link,
        mode
    );

    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int result = rmcontrol(
        CMD_NVLINK_SET_HW_ERR_INJECT, &params, sizeof(params), &status
    );
    clock_gettime(CLOCK_MONOTONIC, &end);
    kmsg("NVLINK_HWINJECT_MARK end status=0x%x\n", status);

    printf(
        "  SET_HW_ERROR_INJECT -> ioctl=%d status=0x%04x %s\n",
        result,
        status,
        nvstatus(status)
    );
    printf(
        "  ioctl took %.3f ms\n",
        ((end.tv_sec + end.tv_nsec / 1e9) -
         (start.tv_sec + start.tv_nsec / 1e9)) *
            1e3
    );
    if (result != 0)
        perror("NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT");

    close(gpufd);
    close(ctlfd);
    return result == 0 && status == 0 ? 0 : 3;
}
