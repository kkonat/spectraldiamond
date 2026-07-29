"""The two one-line status helpers shared by the single render and --anim."""
import sys


def shadow_note(a):
    return f" + shadow {a.shadow:g}" if a.shadow > 0.0 else ""


def _progress(prefix=""):
    def show(done, total, elapsed):
        eta = elapsed / done * (total - done) if done else 0.0
        sys.stderr.write(f"\r  {prefix}{done}/{total} spp  {elapsed:6.1f}s "
                         f"(eta {eta:6.1f}s)")
        sys.stderr.flush()
    return show
