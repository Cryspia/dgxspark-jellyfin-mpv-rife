// Native C++ VapourSynth filter — split-dual dispatch shell.
//
// The shell exposes `core.dgxspark_split_dual.SplitDual(clip, scale=...,
// interp_mult=..., compute_id=...)` which outputs (n_src-1)*mult frames
// at scale·input resolution. Each output frame fires a Python compute
// callback that submits work into the queue_mgr DAG (CCSR / INTERP /
// SR_INTERP, possibly across host + remote worker) and blocks on
// frame_done. Per-frame compute lives entirely in Python; this C++
// shell only handles VideoInfo, frame-prop forwarding, and the cross-
// language plane-pointer pass-through.
//
// Historical note: pre-M5 the project had multiple modes (LEGACY_PAIR,
// "Mode A", "Mode B"); the surviving path is the split-task dual-
// machine pipeline. The C++ shell kept the legacy "ModeB" name into
// M6; M7 renamed it to `SplitDual`. Build with the sibling
// CMakeLists.txt.

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace py = pybind11;

namespace {

struct PassthroughData {
    VSNode *node = nullptr;
};

const VSFrame *VS_CC passthroughGetFrame(int n, int activationReason,
                                          void *instanceData,
                                          void ** /*frameData*/,
                                          VSFrameContext *frameCtx,
                                          VSCore * /*core*/,
                                          const VSAPI *vsapi) {
    auto *d = static_cast<PassthroughData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        // Just hand back the source frame; vapoursynth refcounts handle
        // ownership. The point of this stage is to confirm we can serve
        // frames from C++.
        return vsapi->getFrameFilter(n, d->node, frameCtx);
    }

    return nullptr;
}

void VS_CC passthroughFree(void *instanceData, VSCore * /*core*/,
                            const VSAPI *vsapi) {
    auto *d = static_cast<PassthroughData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC passthroughCreate(const VSMap *in, VSMap *out,
                              void * /*userData*/, VSCore *core,
                              const VSAPI *vsapi) {
    int err = 0;
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) {
        vsapi->mapSetError(out, "SplitDual.Passthrough: missing 'clip' arg");
        return;
    }
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    if (!vsh::isConstantVideoFormat(vi)) {
        vsapi->freeNode(node);
        vsapi->mapSetError(out, "SplitDual.Passthrough: variable format / "
                                  "dimensions not supported");
        return;
    }

    auto *d = new PassthroughData{};
    d->node = node;

    VSFilterDependency deps[] = {{node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Passthrough", vi,
                              passthroughGetFrame, passthroughFree,
                              fmParallel, deps, 1, d, core);
}

// =============================================================================
// Stage 2: SplitDual shell — 2× temporal × 2× spatial output. Each output frame
// requests its pair (src_a = src[n/2], src_b = src[n/2 + 1]) but emits a
// black frame of the upscaled dims. Stage 3 will plug in real compute.
// =============================================================================

struct SplitDualData {
    VSNode *node = nullptr;     // source clip
    int scale = 2;              // spatial upscale factor
    int interp_mult = 2;        // M7: temporal multiplier (2 = current
                                //   x2 frame doubling; 3 = x3 mode with
                                //   2 interp frames per source pair)
    VSVideoInfo vi{};           // output video info (mult× temporal+spatial)
    int n_src = 0;              // source-clip frame count (cached)
    // Python compute callable. Signature (called with GIL held inside
    // splitDualGetFrame):
    //   compute(pair_k:int, phase:int,
    //           sa_y:int, sa_u:int, sa_v:int,
    //           sb_y:int, sb_u:int, sb_v:int,
    //           dst_y:int, dst_u:int, dst_v:int,
    //           sa_y_stride:int, sa_uv_stride:int,
    //           dst_y_stride:int, dst_uv_stride:int) -> None
    // *_y/_u/_v are raw plane addresses (int handles), Python wraps via
    // numpy.frombuffer. Heavy lifts (torch ops) release the GIL on
    // their own — pybind11's GIL stays held in our wrapper, which is
    // fine because we want each compute callback to behave like one
    // ModifyFrame call but free of Python-side dispatch overhead.
    py::object compute_callable;
};

const VSFrame *VS_CC splitDualGetFrame(int n, int activationReason,
                                    void *instanceData,
                                    void ** /*frameData*/,
                                    VSFrameContext *frameCtx,
                                    VSCore *core,
                                    const VSAPI *vsapi) {
    auto *d = static_cast<SplitDualData *>(instanceData);
    const int mult = d->interp_mult;
    const int pair_k = n / mult;
    const int src_a_idx = std::min(pair_k, d->n_src - 1);
    const int src_b_idx = std::min(pair_k + 1, d->n_src - 1);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(src_a_idx, d->node, frameCtx);
        if (src_b_idx != src_a_idx)
            vsapi->requestFrameFilter(src_b_idx, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        const VSFrame *src_a = vsapi->getFrameFilter(src_a_idx, d->node, frameCtx);
        const VSFrame *src_b = (src_b_idx != src_a_idx)
            ? vsapi->getFrameFilter(src_b_idx, d->node, frameCtx) : src_a;

        // dst inherits src_a's frame props so _Matrix / _ColorRange /
        // _MP_IMAGE survive. But that ALSO copies src_a's
        // _DurationNum/_DurationDen (which encode the source's
        // per-frame timestamp gap). Since we emit mult× the frames per
        // pair, each output frame must carry 1/mult of src_a's duration.
        // Without this fix mpv schedules our output at the source's
        // 24fps interval (≈42 ms / frame instead of 21), and ffmpeg
        // PSNR against the single-machine baseline silently misaligns.
        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width,
                                              d->vi.height, src_a, core);
        // Scale per-frame duration by mult so the mult× output stream
        // is tagged at 1/(mult·fps) per frame. Multiply den by mult
        // instead of dividing num by mult to avoid losing precision.
        VSMap *props = vsapi->getFramePropertiesRW(dst);
        if (props) {
            int err_n = 0, err_d = 0;
            int64_t dur_num = vsapi->mapGetInt(props, "_DurationNum",
                                                 0, &err_n);
            int64_t dur_den = vsapi->mapGetInt(props, "_DurationDen",
                                                 0, &err_d);
            if (!err_n && !err_d) {
                vsapi->mapSetInt(props, "_DurationDen", dur_den * mult,
                                  maReplace);
            }
        }

        // phase ∈ [0, mult). phase=0 = src passthrough SR (CCSR);
        // phase=k>0 = interp SR at timestep=k/mult.
        const int phase = n % mult;

        if (d->compute_callable) {
            // Call into Python compute. Plane pointers are passed as
            // ints; Python wraps via numpy.frombuffer.
            try {
                py::gil_scoped_acquire gil;
                d->compute_callable(
                    pair_k, phase,
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_a, 0)),
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_a, 1)),
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_a, 2)),
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_b, 0)),
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_b, 1)),
                    reinterpret_cast<intptr_t>(vsapi->getReadPtr(src_b, 2)),
                    reinterpret_cast<intptr_t>(vsapi->getWritePtr(dst, 0)),
                    reinterpret_cast<intptr_t>(vsapi->getWritePtr(dst, 1)),
                    reinterpret_cast<intptr_t>(vsapi->getWritePtr(dst, 2)),
                    static_cast<ptrdiff_t>(vsapi->getStride(src_a, 0)),
                    static_cast<ptrdiff_t>(vsapi->getStride(src_a, 1)),
                    static_cast<ptrdiff_t>(vsapi->getStride(dst, 0)),
                    static_cast<ptrdiff_t>(vsapi->getStride(dst, 1)));
            } catch (const py::error_already_set &e) {
                vsapi->setFilterError(
                    (std::string("SplitDual compute callback raised: ") +
                     e.what()).c_str(),
                    frameCtx);
                if (src_b != src_a) vsapi->freeFrame(src_b);
                vsapi->freeFrame(src_a);
                vsapi->freeFrame(dst);
                return nullptr;
            }
        } else {
            // No callback wired up — emit zeros (skeleton mode).
            for (int p = 0; p < d->vi.format.numPlanes; ++p) {
                uint8_t *ptr = vsapi->getWritePtr(dst, p);
                ptrdiff_t stride = vsapi->getStride(dst, p);
                int h = vsapi->getFrameHeight(dst, p);
                std::memset(ptr, 0, static_cast<size_t>(stride) * h);
            }
        }

        if (src_b != src_a) vsapi->freeFrame(src_b);
        vsapi->freeFrame(src_a);
        return dst;
    }

    return nullptr;
}

void VS_CC splitDualFree(void *instanceData, VSCore * /*core*/,
                      const VSAPI *vsapi) {
    auto *d = static_cast<SplitDualData *>(instanceData);
    vsapi->freeNode(d->node);
    // Releasing the Python ref must be done with the GIL held. If the
    // interpreter is finalising this can throw, but the alternative
    // (leaking) is worse — bound the catch to keep mpv shutdown sane.
    try {
        py::gil_scoped_acquire gil;
        d->compute_callable.release();
    } catch (...) {
        // intentionally swallowed: interpreter likely finalised
    }
    delete d;
}

void VS_CC splitDualCreate(const VSMap *in, VSMap *out, void * /*userData*/,
                        VSCore *core, const VSAPI *vsapi) {
    int err = 0;
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) {
        vsapi->mapSetError(out, "SplitDual: missing 'clip' arg");
        return;
    }
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    if (!vsh::isConstantVideoFormat(vi)) {
        vsapi->freeNode(node);
        vsapi->mapSetError(out, "SplitDual: variable format / dims not supported");
        return;
    }
    // Accept 10-bit YUV at 4:2:0 / 4:2:2 / 4:4:4 input. Output
    // is always YUV420P10 (mpv expects this); the Python compute
    // callback writes 4:2:0-shaped plane data to dst regardless of
    // input subsampling.
    if (vi->format.colorFamily != cfYUV ||
        vi->format.bitsPerSample != 10 ||
        vi->format.sampleType != stInteger ||
        vi->format.subSamplingW < 0 || vi->format.subSamplingW > 1 ||
        vi->format.subSamplingH < 0 || vi->format.subSamplingH > 1) {
        vsapi->freeNode(node);
        vsapi->mapSetError(out,
            "SplitDual: only 10-bit YUV at 4:2:0 / 4:2:2 / 4:4:4 is supported");
        return;
    }

    int scale = vsapi->mapGetIntSaturated(in, "scale", 0, &err);
    if (err) scale = 2;
    // M5: accept any scale in [1, 4]. 1 = 4K input downsample path
    // (Python make_dispatcher uses downsample_pre=2 internally); 2 =
    // 1080p×2; 3 = 720p×3; 4 = 480p×4. Output frame dims = vi->width
    // * scale (line below) already use the variable correctly.
    if (scale < 1 || scale > 4) {
        vsapi->freeNode(node);
        vsapi->mapSetError(out,
            "SplitDual: scale must be in [1, 4]");
        return;
    }

    // Temporal multiplier. 1 = no temporal upscale (only phase=0
    // CCSR per pair — F9 OFF "SR-only" in dual); 2 = x2 frame
    // doubling; 3 = x3 (t=1/3, 2/3); 4 = x4 (t=1/4, 2/4, 3/4).
    int interp_mult = vsapi->mapGetIntSaturated(in, "interp_mult", 0, &err);
    if (err) interp_mult = 2;
    if (interp_mult < 1 || interp_mult > 4) {
        vsapi->freeNode(node);
        vsapi->mapSetError(out,
            "SplitDual: interp_mult must be 1, 2, 3, or 4");
        return;
    }

    // Optional `compute` arg: a Python callable (vapoursynth's
    // VSMap holds Python callables under type "func" — but its
    // VSFunction wrapper is not the same as a py::object. The
    // canonical idiom for plugins that need a real Python callable
    // is to read it via the embedding Python interpreter's globals,
    // or to register the filter via Python (which is what we'll do
    // long-term). For stage 3 we read it via VSMap's data field,
    // expecting a stringified id("compute") — caller stashes the
    // callable in a module-level dict the Python wrapper checks.
    // To keep this stage simple, we accept the callable as an int
    // handle (id()) and look it up via a Python helper module.
    py::object compute_callable;
    int64_t cb_id = vsapi->mapGetInt(in, "compute_id", 0, &err);
    if (!err && cb_id != 0) {
        try {
            py::gil_scoped_acquire gil;
            // The dispatcher Python module registers callbacks keyed
            // by id() at install time. We look up by id here.
            py::module_ helper = py::module_::import("dgxspark_native_helper");
            py::object cb = helper.attr("get_callback")(cb_id);
            if (!cb.is_none()) compute_callable = cb;
        } catch (const py::error_already_set &e) {
            vsapi->freeNode(node);
            vsapi->mapSetError(out,
                (std::string("SplitDual: failed to resolve compute callback: ") +
                 e.what()).c_str());
            return;
        }
    }

    auto *d = new SplitDualData{};
    d->node = node;
    d->scale = scale;
    d->interp_mult = interp_mult;
    d->compute_callable = std::move(compute_callable);
    d->n_src = vi->numFrames;
    d->vi = *vi;
    d->vi.width  = vi->width  * scale;
    d->vi.height = vi->height * scale;
    // Output format:
    //   • scale != 1 (1080p / 720p / 480p paths upscaling to a larger
    //     target): force YUV420P10. mpv consumes 4:2:0 at the 4K
    //     display surface, and the Python compute callback writes
    //     4:2:0-shaped plane data into dst on these paths.
    //   • scale == 1 (4K-like passthrough variant for 4K + non-standard
    //     resolutions): output dim == src dim, preserve src chroma
    //     subsampling. Python compute callback writes
    //     src-chroma-shaped planes (the real-frame passthrough copies
    //     src YUV verbatim; interp frames downsample chroma to dst
    //     dims via F.interpolate inside SR_INTERP).
    if (scale != 1 &&
        (vi->format.subSamplingW != 1 || vi->format.subSamplingH != 1)) {
        VSVideoFormat out_format;
        if (!vsapi->queryVideoFormat(&out_format, cfYUV, stInteger,
                                       10, 1, 1, core)) {
            vsapi->freeNode(node);
            vsapi->mapSetError(out,
                "SplitDual: failed to query YUV420P10 output format");
            delete d;
            return;
        }
        d->vi.format = out_format;
    }
    // Output length = (N_src - 1) * mult so the final source frame has
    // a "next" frame to interpolate against; drop the dangling tail.
    d->vi.numFrames = std::max(0, (vi->numFrames - 1) * interp_mult);
    // mult× temporal: fps scales by mult.
    d->vi.fpsNum = vi->fpsNum * interp_mult;
    d->vi.fpsDen = vi->fpsDen;

    VSFilterDependency deps[] = {{node, rpGeneral}};
    vsapi->createVideoFilter(out, "SplitDual", &d->vi,
                              splitDualGetFrame, splitDualFree,
                              fmParallel, deps, 1, d, core);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin,
                                              const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "io.dgxspark.split_dual",      // unique identifier
        "dgxspark_split_dual",             // namespace
        "DGX-Spark dual-machine Mode B (C++)",
        VS_MAKE_VERSION(0, 1),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin);

    vspapi->registerFunction("Passthrough", "clip:vnode;",
                              "clip:vnode;",
                              passthroughCreate, nullptr, plugin);
    vspapi->registerFunction("SplitDual",
                              "clip:vnode;scale:int:opt;compute_id:int:opt;"
                              "interp_mult:int:opt;",
                              "clip:vnode;",
                              splitDualCreate, nullptr, plugin);
}
