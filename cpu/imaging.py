"""Display transform, atomic file writes and the progress lines."""
import os, sys, tempfile, time

import numpy as np

def tonemap(x, exposure):
    x = np.maximum(x, 0.0) * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1) ** (1 / 2.2)


def atomic_write(path, write):
    """Build the file under a temporary name in the same directory, then swap
    it into place.

    Writing straight to the destination truncates it to zero length before the
    first byte of image data goes out, so for the length of the write the file
    on disk is a valid PNG header followed by nothing. Any reader arriving
    inside that window fails with "EOF while reading chunk IDAT" -- an editor's
    image preview auto-reloading is the usual one, since the change event fires
    at the truncate, not at the close. The render was never at fault. Swapping
    a finished file in closes the window: a reader sees either the old file or
    the new one, never a partial one.

    Returns the path actually written, which is not `path` in the one case the
    swap cannot be done -- see below.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)     # rendered/ on a fresh checkout
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_",
                               suffix=os.path.splitext(path)[1])
    os.close(fd)
    try:
        write(tmp)
        for _ in range(25):                      # ~5 s
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                # Windows refuses the swap while another process holds the
                # destination open. Normally that is a preview or thumbnailer
                # passing through, so waiting clears it.
                time.sleep(0.2)
        # Still locked. Falling back to an in-place write would truncate the
        # destination and reintroduce exactly the torn read this function
        # exists to prevent, so park the finished render beside it instead --
        # nothing is lost and nothing is corrupted.
        keep = path + ".new"
        os.replace(tmp, keep)
        sys.stderr.write(f"\n{path} is held open by another process; "
                         f"wrote {keep} instead\n")
        return keep
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_png(arr, path):
    from PIL import Image
    return atomic_write(path, lambda p: Image.fromarray(arr).save(p))


def save_npy(arr, path):
    if not path.endswith(".npy"):
        path += ".npy"          # else np.save appends it to the temporary name

    def write(p):
        # Handed a file object, np.save neither appends .npy nor closes it --
        # and a still-open temporary file cannot be renamed on Windows.
        with open(p, "wb") as f:
            np.save(f, arr)

    return atomic_write(path, write)


def fmt_hms(s):
    s = int(max(s, 0.0))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def anim_progress(frame, nframes, t0):
    """Progress line for one frame of an animation, with the ETA measured over
    the whole sequence instead of the current frame: what you want to know on a
    60-frame run is when the .mp4 lands, not when frame 7 does.

    Every frame costs the same -- only the azimuth changes -- so once any frame
    has finished, the mean completed-frame time is the estimator, and it covers
    the per-frame PNG encode for free. Only the ffmpeg mux at the end is
    outside it, and that is seconds.

    Frame 1 has no completed frame to go on and must extrapolate from partial
    work, which reads badly on the GPU path: the dispatch auto-tuner starts at
    one sample per pass and calibrates upward, so the first few updates are far
    slower per sample than the steady state and the estimate starts several
    times too high. It converges within a second or two, and every frame after
    the first is stable.

    Accepts the CPU renderer's (done, total) and the GPU's (done, total,
    elapsed) alike; the elapsed it is handed is per-frame, so it is ignored."""
    t_frame = time.time()                   # this frame starts now
    def show(done, total, *_):
        now = time.time()
        frac = done / max(total, 1)
        left = nframes - frame - frac       # frames still to render
        if frame > 0:
            eta = (t_frame - t0) / frame * left
        elif frac > 1e-9:
            eta = (now - t_frame) / frac * left
        else:
            eta = 0.0
        sys.stderr.write(f"\r  frame {frame+1}/{nframes} | {done}/{total} spp | "
                         f"{fmt_hms(now - t0)} elapsed | eta {fmt_hms(eta)}   ")
        sys.stderr.flush()
    return show
