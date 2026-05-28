// priority_boost.c — minimal Vulkan layer that injects
// VkDeviceQueueGlobalPriorityCreateInfoKHR into every queue at device
// creation. Built as part of the host-GPU CUDA / Vulkan contention
// investigation (see bench/display_fps.sh + display_fps_probe.lua):
// mpv gpu-next does not expose Vulkan queue priority, and on GB10 CUDA
// stream priority is already at its lowest, so elevating the Vulkan
// present queue is the only remaining "let Vulkan win the GPU
// scheduler" lever. Wired up via an explicit layer manifest under
// ~/.local/share/vulkan/explicit_layer.d/ and toggled per-run with
// VK_INSTANCE_LAYERS=VK_LAYER_PRIORITY_BOOST.
//
// Env tunables:
//   VK_PRIORITY_BOOST_LEVEL = low|medium|high|realtime (default: high)
//   VK_PRIORITY_BOOST_VERBOSE = 1 enables stderr trace
//
// HIGH/REALTIME on the NVIDIA driver normally require CAP_SYS_NICE on
// the calling process — without it vkCreateDevice returns
// VK_ERROR_NOT_PERMITTED_KHR. We retry once at MEDIUM on that error
// and warn, so the layer is safe to leave loaded even without the
// capability (just no-op at MEDIUM == driver default).
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYER_NAME "VK_LAYER_PRIORITY_BOOST"

// One-shot globals; mpv only ever spins up one Vulkan instance + one
// device. A real layer would key these per-handle in a hash table.
static PFN_vkGetInstanceProcAddr g_next_gipa = NULL;
static PFN_vkCreateDevice        g_next_create_device = NULL;
static VkInstance                g_instance = VK_NULL_HANDLE;
// The next layer's vkGetDeviceProcAddr — captured at CreateDevice via
// the device-chain's layerInfo. Routing my_GetDeviceProcAddr through
// the instance-side gipa returns the loader terminator instead of the
// next layer's GDPA, which breaks dispatch for many device-level
// functions (mpv uses far more device calls than vulkaninfo, which is
// why vulkaninfo survives the same wrapper but mpv crashes).
static PFN_vkGetDeviceProcAddr   g_next_gdpa = NULL;

static int verbose(void) {
    const char *e = getenv("VK_PRIORITY_BOOST_VERBOSE");
    return e != NULL && *e == '1';
}

static VkQueueGlobalPriorityKHR want_priority(void) {
    // Default MEDIUM (= driver default) when the env is unset: the layer
    // is a no-op unless explicitly opted in via VK_PRIORITY_BOOST_LEVEL,
    // so we can leave it installed system-wide without affecting other
    // Vulkan apps. Only mpv (the cap-holding consumer) opts into HIGH/
    // REALTIME via its wrapper env.
    const char *e = getenv("VK_PRIORITY_BOOST_LEVEL");
    if (!e)                              return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
    if (strcmp(e, "low") == 0)           return VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
    if (strcmp(e, "medium") == 0)        return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
    if (strcmp(e, "high") == 0)          return VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
    if (strcmp(e, "realtime") == 0)      return VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
    return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
}

static const char* priority_name(VkQueueGlobalPriorityKHR p) {
    switch (p) {
        case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:      return "LOW";
        case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:   return "MEDIUM";
        case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:     return "HIGH";
        case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR: return "REALTIME";
        default: return "?";
    }
}

// Allocate the modified queue-create-info array + the per-queue
// VkDeviceQueueGlobalPriorityCreateInfoKHR records. Returns a fresh
// copy of pCreateInfo with pQueueCreateInfos pointing into the new
// arrays. Caller owns the out_alloc_block and must free() it after
// the wrapped CreateDevice returns.
static void build_boosted_create_info(
        const VkDeviceCreateInfo *src,
        VkQueueGlobalPriorityKHR prio,
        VkDeviceCreateInfo       *out,
        void                    **out_alloc_block)
{
    uint32_t n = src->queueCreateInfoCount;
    size_t qci_bytes = n * sizeof(VkDeviceQueueCreateInfo);
    size_t gp_bytes  = n * sizeof(VkDeviceQueueGlobalPriorityCreateInfoKHR);
    char *block = calloc(1, qci_bytes + gp_bytes);
    *out_alloc_block = block;

    VkDeviceQueueCreateInfo *new_qcis = (VkDeviceQueueCreateInfo*)block;
    VkDeviceQueueGlobalPriorityCreateInfoKHR *gps =
        (VkDeviceQueueGlobalPriorityCreateInfoKHR*)(block + qci_bytes);

    for (uint32_t i = 0; i < n; i++) {
        new_qcis[i] = src->pQueueCreateInfos[i];
        gps[i].sType =
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
        gps[i].pNext = (void*)new_qcis[i].pNext;  // chain in front of existing pNext
        gps[i].globalPriority = prio;
        new_qcis[i].pNext = &gps[i];
    }

    *out = *src;
    out->pQueueCreateInfos = new_qcis;
}

VKAPI_ATTR VkResult VKAPI_CALL my_CreateDevice(
        VkPhysicalDevice              physicalDevice,
        const VkDeviceCreateInfo     *pCreateInfo,
        const VkAllocationCallbacks  *pAllocator,
        VkDevice                     *pDevice)
{
    // Find device-layer link info, advance the chain.
    VkLayerDeviceCreateInfo *layerInfo =
        (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    while (layerInfo
           && !(layerInfo->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                && layerInfo->function == VK_LAYER_LINK_INFO))
        layerInfo = (VkLayerDeviceCreateInfo*)layerInfo->pNext;
    if (!layerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa =
        layerInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   next_gdpa =
        layerInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    layerInfo->u.pLayerInfo = layerInfo->u.pLayerInfo->pNext;
    PFN_vkCreateDevice next_create =
        (PFN_vkCreateDevice)next_gipa(g_instance, "vkCreateDevice");
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    // True passthrough when the env is unset: pass the original
    // pCreateInfo through unmodified so we can tell injection bugs
    // apart from chain-wrapping bugs in the bench.
    const char *level_env = getenv("VK_PRIORITY_BOOST_LEVEL");
    if (!level_env) {
        if (verbose())
            fprintf(stderr, "[priority_boost] vkCreateDevice: PASSTHRU "
                            "(VK_PRIORITY_BOOST_LEVEL unset)\n");
        VkResult r0 = next_create(physicalDevice, pCreateInfo,
                                  pAllocator, pDevice);
        if (r0 == VK_SUCCESS) g_next_gdpa = next_gdpa;
        return r0;
    }

    // Inject our pNext block into a copy of pQueueCreateInfos.
    VkQueueGlobalPriorityKHR prio = want_priority();
    VkDeviceCreateInfo boosted;
    void *alloc_block = NULL;
    build_boosted_create_info(pCreateInfo, prio, &boosted, &alloc_block);

    if (verbose())
        fprintf(stderr, "[priority_boost] vkCreateDevice: requesting %s "
                        "on %u queue(s)\n",
                priority_name(prio), pCreateInfo->queueCreateInfoCount);

    VkResult r = next_create(physicalDevice, &boosted, pAllocator, pDevice);

    // NOT_PERMITTED → fall back to MEDIUM (== driver default) so mpv
    // still starts. Common cause: missing CAP_SYS_NICE for HIGH/REALTIME.
    if (r == VK_ERROR_NOT_PERMITTED_KHR
            && prio != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR) {
        free(alloc_block); alloc_block = NULL;
        fprintf(stderr, "[priority_boost] %s denied "
                        "(VK_ERROR_NOT_PERMITTED — need CAP_SYS_NICE on mpv?), "
                        "retrying at MEDIUM\n", priority_name(prio));
        build_boosted_create_info(pCreateInfo,
                                  VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
                                  &boosted, &alloc_block);
        r = next_create(physicalDevice, &boosted, pAllocator, pDevice);
    }

    if (verbose())
        fprintf(stderr, "[priority_boost] vkCreateDevice result=%d\n", r);

    if (r == VK_SUCCESS) g_next_gdpa = next_gdpa;
    free(alloc_block);
    return r;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL my_GetInstanceProcAddr(
        VkInstance instance, const char *pName);

VKAPI_ATTR VkResult VKAPI_CALL my_CreateInstance(
        const VkInstanceCreateInfo  *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkInstance                  *pInstance)
{
    VkLayerInstanceCreateInfo *layerInfo =
        (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (layerInfo
           && !(layerInfo->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
                && layerInfo->function == VK_LAYER_LINK_INFO))
        layerInfo = (VkLayerInstanceCreateInfo*)layerInfo->pNext;
    if (!layerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa =
        layerInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    layerInfo->u.pLayerInfo = layerInfo->u.pLayerInfo->pNext;

    PFN_vkCreateInstance next_create =
        (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = next_create(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS) return r;

    g_next_gipa = next_gipa;
    g_instance  = *pInstance;
    g_next_create_device =
        (PFN_vkCreateDevice)next_gipa(*pInstance, "vkCreateDevice");

    if (verbose())
        fprintf(stderr, "[priority_boost] instance created, next "
                        "CreateDevice=%p\n", (void*)g_next_create_device);
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL my_GetDeviceProcAddr(
        VkDevice device, const char *pName)
{
    // Route through the device-chain's next GDPA captured at
    // CreateDevice (NOT the instance gipa, which returns the loader
    // terminator for device-level calls and breaks dispatch).
    if (g_next_gdpa) return g_next_gdpa(device, pName);
    return NULL;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL my_GetInstanceProcAddr(
        VkInstance instance, const char *pName)
{
    if (!pName) return NULL;
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)my_GetInstanceProcAddr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0)
        return (PFN_vkVoidFunction)my_GetDeviceProcAddr;
    if (strcmp(pName, "vkCreateInstance") == 0)
        return (PFN_vkVoidFunction)my_CreateInstance;
    if (strcmp(pName, "vkCreateDevice") == 0)
        return (PFN_vkVoidFunction)my_CreateDevice;
    if (g_next_gipa) return g_next_gipa(instance, pName);
    return NULL;
}

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(
        VkNegotiateLayerInterface *pVersionStruct)
{
    if (pVersionStruct->loaderLayerInterfaceVersion > 2)
        pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetInstanceProcAddr     = my_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr       = my_GetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
