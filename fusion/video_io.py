"""Video reading and overlay writing shared by every stage in this experiment.

The frame-selection arithmetic here is a deliberate, exact copy of the convention the existing
mono-pipeline uses in ``run_wilor_3d.py`` / ``run_mediapipe_2d.py``:

    step    = 1 if sample_fps <= 0 else max(1, round(fps / sample_fps))
    start_f = round(start_sec * fps)
    end_f   = total if duration_sec >= 1e8 else min(total, start_f + round(duration_sec * fps))

and ``frame_idx`` is always the **absolute** frame number in the source video. Two stages that
disagree about any of this silently fuse nothing: the wrist matcher finds no pairs and the run
"succeeds" with a degenerate deliverable. ``fuse_2d_3d.py`` guards against it by comparing
``width``/``height``/``step``, so every producer must write those keys.
"""

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Everything a producer stage needs to write a schema-compatible npz."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    step: int
    start_frame: int
    end_frame: int
    sample_fps: float

    @property
    def output_fps(self):
        return self.fps / self.step

    @property
    def n_selected(self):
        """How many frames this selection will process, if the container is honest."""
        span = max(0, self.end_frame - self.start_frame)
        return (span + self.step - 1) // self.step


def open_video(path, start_sec=0.0, duration_sec=1e9, sample_fps=0.0, require_cfr=True):
    """Open a video and resolve the frame selection. Returns ``(capture, VideoInfo)``.

    ``require_cfr`` reproduces the existing pipeline's hard failures. A VFR or broken container
    reports ``fps=0`` or ``frame_count=0``; without the guard the read loop silently does nothing and
    the job "succeeds" with an empty deliverable.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f'cannot open video: {path}')

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not (fps and fps > 0):
        if require_cfr:
            raise SystemExit(f'video reports fps={fps}; re-mux to CFR before running')
        fps = 30.0
        print(f'[warn] video reports fps={fps}; assuming 30')
    if total <= 0:
        if require_cfr:
            raise SystemExit(f'video reports frame_count={total}; re-mux (missing nb_frames) first')
        total = 1 << 30

    step = 1 if sample_fps <= 0 else max(1, int(round(fps / sample_fps)))
    start_frame = int(round(start_sec * fps))
    end_frame = total if duration_sec >= 1e8 else min(total, start_frame + int(round(duration_sec * fps)))

    info = VideoInfo(path=str(path), width=width, height=height, fps=float(fps), total_frames=total,
                     step=step, start_frame=start_frame, end_frame=end_frame,
                     sample_fps=float(sample_fps))
    return capture, info


def iter_frames(capture, info):
    """Yield ``(absolute_frame_index, bgr_frame)`` for the selected frames.

    Frames outside the stride are read and discarded rather than sought past: seeking per frame on
    h264 is both slower and less reliable than a linear read.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, info.start_frame)
    index = info.start_frame
    while index < info.end_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if (index - info.start_frame) % info.step != 0:
            index += 1
            continue
        yield index, frame
        index += 1


class OverlayVideoWriter:
    """Write BGR frames to an mp4, preferring an ffmpeg h264 pipe.

    OpenCV's bundled ``mp4v`` encoder is the fallback when ffmpeg is not on PATH. It works, but
    produces files some players refuse to scrub - which matters when the whole point of the artifact
    is a human scrubbing through it looking for failures.
    """

    def __init__(self, path, width, height, fps):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.proc = None
        self.writer = None

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is not None:
            cmd = [
                ffmpeg, '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}', '-r', f'{self.fps:.6f}', '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.backend = 'ffmpeg/libx264'
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                raise RuntimeError(f'could not open an OpenCV VideoWriter for {self.path}')
            self.backend = 'opencv/mp4v'

    def write(self, frame_bgr):
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if self.proc is not None:
            self.proc.stdin.write(frame_bgr.tobytes())
        else:
            self.writer.write(frame_bgr)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
