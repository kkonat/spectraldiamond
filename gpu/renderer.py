"""Device, pipeline and uniform packing. The kernel lives in kernel.wgsl."""
import math, os, sys, time

import numpy as np

from cpu import geometry, lighting, optics, spectrum

from gpu.colour import SKY_GROUND, SKY_TINT, working_space

SHADER = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "kernel.wgsl"), encoding="utf8").read()

F = np.float32
LMIN, LMAX = spectrum.LMIN, spectrum.LMAX
MAX_LIGHTS = 5
WORKGROUP = 64
MAX_WG_PER_DIM = 65535


PARAMS = np.dtype([
    ("eye",    "4f4"),   # xyz camera position
    ("fwd",    "4f4"),   # xyz camera forward
    ("right",  "4f4"),   # xyz camera right
    ("up",     "4f4"),   # xyz camera up
    ("wnorm",  "4f4"),   # xyz white-normalisation of the CIE weights
    ("bs_c",   "4f4"),   # xyz bounding-sphere centre, w radius
    ("cauchy", "4f4"),   # A, B
    ("misc",   "4f4"),   # ambient, tan(fov/2), pixel footprint, shadow strength
    ("floor_p", "4f4"),  # plane y, pattern scale, fade radius, mode
    ("floor_s", "4f4"),  # brightness, F0, gloss width, ambient-occlusion
    ("floor_a", "4f4"),  # pattern colour A
    ("floor_b", "4f4"),  # pattern colour B
    ("x2ws",   "(3,4)f4"),   # XYZ -> working-space primaries, one row per vec4
    ("sky",    "(2,4)f4"),   # sky tint, ground bounce, in the working space
    ("cfg",    "4u4"),   # W, H, n_triangles, bounces
    ("cfg2",   "4u4"),   # seed, sample_base, samples_this_pass, n_lights
    ("cfg3",   "4u4"),   # grid_x, n_pixels
    ("lights", "(5,4)f4"),   # xyz direction, w angular width
    ("lint",   "(5,4)f4"),   # x intensity
])

# Floor presets: mode id, pattern colour A, colour B, pattern scale, Fresnel F0,
# gloss lobe width, brightness. `mirror` is the black glossy acrylic that gem
# and jewellery photography is normally shot on -- the stone's reflection is the
# whole point of it. The patterned modes exist to be looked at *through* the
# stone: refraction shears them into the dispersion fan.
FLOORS = {
    "none":    (0, (0, 0, 0),              (0, 0, 0),               1.00, 0.00, 0.000, 0.00),
    "mirror":  (1, (0.015, 0.015, 0.020),  (0.015, 0.015, 0.020),   1.00, 0.35, 0.0040, 0.35),
    "sweep":   (2, (0.60, 0.62, 0.70),     (0.020, 0.020, 0.030),   1.60, 0.09, 0.0090, 0.35),
    "checker": (3, (0.62, 0.63, 0.67),     (0.045, 0.045, 0.055),   0.22, 0.06, 0.0100, 0.35),
    "stripes": (4, (0.70, 0.70, 0.75),     (0.030, 0.030, 0.040),   0.13, 0.07, 0.0080, 0.35),
    "rings":   (5, (0.72, 0.74, 0.82),     (0.030, 0.030, 0.050),   0.11, 0.07, 0.0080, 0.35),
    "grid":    (6, (0.020, 0.020, 0.030),  (0.75, 0.80, 0.95),      0.30, 0.10, 0.0060, 0.40),
}


def _lazy_wgpu():
    try:
        import wgpu
    except ImportError:
        sys.exit("wgpu not installed; run:  pip install wgpu")
    return wgpu


def list_adapters():
    wgpu = _lazy_wgpu()
    for a in wgpu.gpu.enumerate_adapters_sync():
        i = a.info
        print(f"  {i['adapter_type']:<12} {i['backend_type']:<8} "
              f"{i['vendor']} {i['device']}")


class FloorSpec:
    """Resolved floor parameters: a preset with the CLI overrides applied.
    Constructed with no arguments it is mode 0, i.e. no floor at all."""

    def __init__(self, name="none", y=-0.43, scale=0.0, bright=-1.0, gloss=-1.0,
                 fade=6.0, ao=0.55):
        mode, a, b, s, f0, gw, br = FLOORS[name]
        self.name = name
        self.mode = mode
        self.col_a, self.col_b = a, b
        self.pattern_scale = scale if scale > 0 else s
        self.f0 = gloss if gloss >= 0 else f0
        self.gloss_w = gw
        self.bright = bright if bright >= 0 else br
        self.y, self.fade, self.ao = y, fade, ao

    def __bool__(self):
        return self.mode != 0

    def __str__(self):
        if not self:
            return "no floor"
        return (f"floor {self.name} at y={self.y:g} "
                f"(scale {self.pattern_scale:g}, F0 {self.f0:g}, "
                f"bright {self.bright:g})")


def camera(azim, elev, dist, fov):
    az, el = math.radians(azim), math.radians(elev)
    eye = np.array([dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                    dist * math.cos(el) * math.cos(az)], F)
    fwd = np.array([0.0, -0.035, 0.0], F) - eye
    fwd /= np.linalg.norm(fwd)
    # world-up degenerates when looking straight down; fall back to +Z
    world_up = np.array([0, 1, 0], F)
    if abs(float(fwd @ world_up)) > 0.999:
        world_up = np.array([0, 0, 1], F)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, fwd, right, up, F(math.tan(math.radians(fov * 0.5)))


class GPURenderer:
    """Owns the device, pipeline and triangle buffer; reusable across frames."""

    def __init__(self, force_fallback=False):
        self.wgpu = wgpu = _lazy_wgpu()
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="low-power" if force_fallback else "high-performance")
        if adapter is None:
            sys.exit("no GPU adapter available")
        self.adapter = adapter
        self.device = device = adapter.request_device_sync()

        self.ntri = len(geometry.TRI_BASE)
        self.tri_buf = device.create_buffer(
            size=self.ntri * 4 * 4 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self._spin = None
        self._upload_gem(geometry.SPIN)

        self.ubo = device.create_buffer(
            size=PARAMS.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        C = wgpu.ShaderStage.COMPUTE
        self.bgl = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": C,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": C,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": C,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
        ])
        self.pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            compute={"module": device.create_shader_module(code=SHADER),
                     "entry_point": "main"})
        self._acc = None
        self._acc_n = 0

    @property
    def name(self):
        i = self.adapter.info
        return f"{i['device']} ({i['backend_type']})"

    def _upload_gem(self, spin):
        """Re-send the triangles when the stone has turned. 110 triangles is
        7 kB, so doing this per animation frame costs nothing measurable."""
        if self._spin == spin:
            return
        geometry.set_spin(spin)
        tri = np.zeros((self.ntri, 4, 4), F)
        tri[:, 0, :3] = geometry.V0
        tri[:, 1, :3] = geometry.E1
        tri[:, 2, :3] = geometry.E2
        tri[:, 3, :3] = geometry.NRM
        self.device.queue.write_buffer(self.tri_buf, 0, tri.tobytes())
        self._spin = spin

    def _accum_buffer(self, npix):
        if self._acc is None or self._acc_n != npix:
            wgpu = self.wgpu
            self._acc = self.device.create_buffer(
                size=npix * 3 * 4,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                       | wgpu.BufferUsage.COPY_DST))
            self._acc_n = npix
            self._bg = self.device.create_bind_group(layout=self.bgl, entries=[
                {"binding": 0, "resource":
                    {"buffer": self.ubo, "offset": 0, "size": self.ubo.size}},
                {"binding": 1, "resource":
                    {"buffer": self.tri_buf, "offset": 0, "size": self.tri_buf.size}},
                {"binding": 2, "resource":
                    {"buffer": self._acc, "offset": 0, "size": self._acc.size}},
            ])
        self.device.queue.write_buffer(self._acc, 0, np.zeros(npix * 3, F).tobytes())
        return self._acc

    def _submit(self, gx, gy):
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self._bg)
        cp.dispatch_workgroups(gx, gy, 1)
        cp.end()
        self.device.queue.submit([enc.finish()])

    def _sync(self):
        """Block until the queue drains. queue.on_submitted_work_done_sync() is
        broken in some wgpu-py builds (callback signature mismatch), so prefer
        the device poller and fall back only if it is absent."""
        dev = self.device
        if hasattr(dev, "_poll_wait"):
            dev._poll_wait()
        else:
            dev.queue.on_submitted_work_done_sync()

    def render(self, W, H, spp, bounces, seed, fire, azim, elev, dist, fov,
               ambient, nlights, pass_spp=0, progress=None, floor=None,
               space=None, spin=0.0, shadow=0.0):
        space = space or working_space("srgb")
        self._upload_gem(spin)
        npix = W * H
        eye, fwd, right, up, scale = camera(azim, elev, dist, fov)
        A_C, B_C = optics.cauchy_coeffs(fire)

        pr = np.zeros((), PARAMS)
        pr["eye"][:3] = eye
        pr["fwd"][:3] = fwd
        pr["right"][:3] = right
        pr["up"][:3] = up
        pr["wnorm"][:3] = space.wnorm
        pr["x2ws"][:, :3] = space.xyz_to_rgb
        pr["sky"][0][:3] = space.from_srgb(SKY_TINT)
        pr["sky"][1][:3] = space.from_srgb(SKY_GROUND)
        pr["bs_c"][:3] = geometry.BS_C
        pr["bs_c"][3] = geometry.BS_R
        pr["cauchy"][:2] = (A_C, B_C)
        # misc.z is the angular size of one pixel: the floor uses it to widen
        # its pattern edges with distance instead of aliasing at the horizon.
        pr["misc"][:] = (ambient, scale, 2.0 * scale / W, shadow)
        fl = floor or FloorSpec()
        pr["floor_p"][:] = (fl.y, fl.pattern_scale, fl.fade, float(fl.mode))
        pr["floor_s"][:] = (fl.bright, fl.f0, fl.gloss_w, fl.ao)
        pr["floor_a"][:3] = space.from_srgb(fl.col_a)
        pr["floor_b"][:3] = space.from_srgb(fl.col_b)
        pr["cfg"][:] = (W, H, self.ntri, bounces)
        pr["cfg2"][3] = nlights
        pr["cfg2"][0] = (int(seed) * 2654435761 + 1) & 0xFFFFFFFF
        for k, (d, w, i) in enumerate(lighting.LIGHTS[:MAX_LIGHTS]):
            pr["lights"][k][:3] = d
            pr["lights"][k][3] = w
            pr["lint"][k][0] = i

        total_wg = (npix + WORKGROUP - 1) // WORKGROUP
        gx = min(total_wg, MAX_WG_PER_DIM)
        gy = (total_wg + gx - 1) // gx
        pr["cfg3"][:2] = (gx, npix)

        self._accum_buffer(npix)

        # Keep every dispatch short: the Windows GPU watchdog resets the driver
        # at ~2 s per command. Calibrate on the first pass, then hold.
        chunk = pass_spp if pass_spp > 0 else 1
        done, t0 = 0, time.time()
        while done < spp:
            n = min(chunk, spp - done)
            pr["cfg2"][1] = done
            pr["cfg2"][2] = n
            self.device.queue.write_buffer(self.ubo, 0, pr.tobytes())
            t1 = time.time()
            self._submit(gx, gy)
            self._sync()
            dt = time.time() - t1
            done += n
            if pass_spp <= 0:
                per = dt / n
                chunk = max(1, min(spp, int(0.35 / per) if per > 1e-6 else spp))
            if progress is not None:
                progress(done, spp, time.time() - t0)

        raw = self.device.queue.read_buffer(self._acc)
        img = np.frombuffer(raw, F).astype(np.float64).reshape(H, W, 3)
        return img / spp
