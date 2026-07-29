"""Render a rotation sequence and mux it with ffmpeg."""
import os, shutil, subprocess, sys, tempfile, time

import numpy as np

from cpu import geometry, imaging, trace

def animate(a):
    """Render a.anim_frames frames, each rotated a.anim_step degrees in azimuth
    from the previous one, then stitch them into an .mp4 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH; install it or drop --anim")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("pillow not installed; --anim needs it to write frames")

    tmp = tempfile.mkdtemp(prefix="diamond_anim_")
    print(f"{len(geometry.TRI)} triangles | {a.width}x{a.width} | {a.spp} spp | "
          f"{a.anim_frames} frames x {a.anim_step:g} deg | fire x{a.fire:g} | "
          f"rig {a.rig} ({a.lights} lights) | ambient {a.ambient:g}")
    t0 = time.time()
    try:
        for i in range(a.anim_frames):
            # The stone turns; the camera, lights and floor do not. --azimuth
            # stays what the user asked for, on every frame.
            img = trace.render_parallel(a.width, a.width, a.spp, a.bounces, a.seed, a.fire,
                                  a.azimuth, a.elevation, a.distance, a.fov, a.chunk,
                                  a.jobs, progress=imaging.anim_progress(i, a.anim_frames, t0),
                                  rig=a.rig, ambient=a.ambient, nlights=a.lights,
                                  spin=i * a.anim_step)
            frame = imaging.tonemap(img, a.exposure)
            Image.fromarray((frame * 255).astype(np.uint8)).save(
                os.path.join(tmp, f"frame_{i:05d}.png"))
        sys.stderr.write("\n")
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(a.fps),
             "-i", os.path.join(tmp, "frame_%05d.png"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-movflags", "+faststart", a.anim],
            check=True)
    except KeyboardInterrupt:
        sys.stderr.write("\nanimation aborted by user\n")
        sys.exit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.anim} ({a.anim_frames} frames) in {time.time()-t0:.1f}s")
