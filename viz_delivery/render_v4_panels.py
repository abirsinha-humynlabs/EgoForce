#!/usr/bin/env python
"""
v4 combined renderer — matches the customer's reference layout (example_viz_SLAM.mp4):

  LEFT  : egocentric video + 21-keypoint hand skeletons + WORLD-FRAME wrist trail (v3)
  RIGHT : four 3D plots of the hands in the world/gravity frame — Front / Side / Top / 3/4 view
          (Left=blue, Right=red, Spine=green head->wrists link from the SLAM head pose)

Built on the v3 world-frame fix: the wrist trail on the left is anchored in 3D via the SLAM head
pose T_world_cam, and the right-hand 3D panels show the same metric hands in a gravity-aligned
world frame, so the two sides agree and together verify the camera SLAM.

Usage: render_v4_panels.py --video left_eye.mp4 --npz *_enhanced_keypoints.npz \
                           --head head_pose_6dof.npz --out DIR [--past-sec 4]
"""
import argparse, os, shutil, subprocess, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v3 import (build_tracks, filter_tracks, assign_handedness,
                       draw_skeleton, draw_traj_world, color, WRIST, HAND_EDGES)
from render_v2 import clip_foot_scores   # CLIP ViT-B/32 hand-vs-foot classifier (v2_clip parity)

VIEWS = [("Front View", 8, -90), ("Side View", 8, 0),
         ("Top View", 89, -90), ("3/4 View", 24, -60)]     # (title, elev, azim)


def imu_gravity_up(imu_csv, T):
    """TRUE world-up from the ZED IMU (fix #1). Averaging the egocentric image-up is garbage on a
    rotating head; the accelerometer's mean specific force is gravity. Return world-up unit vector."""
    import csv
    rows = list(csv.reader(open(imu_csv)))
    hdr = [h.strip().lower() for h in rows[0]]
    ix, iy, iz = hdr.index('x'), hdr.index('y'), hdr.index('z')
    acc = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows[1:] if len(r) > iz])
    # at rest the accelerometer reads specific force = -gravity, i.e. it already points UP
    up_cam = acc.mean(0); up_cam /= (np.linalg.norm(up_cam) + 1e-9)
    n = min(len(T), 400)
    up_world = np.mean([T[f][:3, :3] @ up_cam for f in range(n)], 0)  # -> DROID world
    return up_world / (np.linalg.norm(up_world) + 1e-9)


def infer_world_up(T):
    ups = T[:, :3, :3] @ np.array([0.0, -1.0, 0.0])         # FALLBACK ONLY (unreliable on egocentric)
    up = ups.mean(0)
    return up / (np.linalg.norm(up) + 1e-9)


def align_basis(up):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(up @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, ref); e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(up, e1);  e2 /= np.linalg.norm(e2) + 1e-9
    return np.stack([e1, e2, up], 0)                        # rows: world -> [x, y, up]


def hand_world(track, f, T):
    """21x3 hand keypoints of `track` at frame f, in world coords (or None)."""
    if f not in track['kp3'] or track['kp3'][f] is None or f >= len(T):
        return None
    kc = np.asarray(track['kp3'][f], float)                 # (21,3) camera frame
    return (T[f][:3, :3] @ kc.T).T + T[f][:3, 3]            # -> world


EYE_H = 1.55                                                # nominal standing eye height (floor ref)


def disp(P, R, center3d):
    """world point -> gravity-aligned display coords about a FIXED scene centre (head keeps its
    true world position and moves through the frame as the person walks)."""
    c = R @ (np.asarray(P) - center3d)
    return np.array([c[0], c[1], EYE_H + float(np.asarray(P) @ R[2])])   # z = floor-ref height


def draw_panels(fig, axes, tracks, t, T, R, center3d, lim, zlim, head_path=None):
    """World-fixed panels: fixed gravity-aligned frame; the head (camera) moves along its path as
    the person walks, with a faint head trail so the locomotion is visible. Axes auto-scaled once
    to contain the whole head trajectory + arm reach."""
    col_rgb = {0: "#1f5fd0", 1: "#e0342f"}                  # left blue, right red
    head_w = T[t][:3, 3] if t < len(T) else center3d
    hd = disp(head_w, R, center3d)
    hands = [(w, tr['hand_at'].get(t, tr['hand']))
             for tr in tracks for w in [hand_world(tr, t, T)] if w is not None]
    for (title, elev, azim), ax in zip(VIEWS, axes):
        ax.clear()
        ax.set_title(title, fontsize=11, pad=2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(zlim[0], zlim[1])
        ax.set_box_aspect((2 * lim, 2 * lim, zlim[1] - zlim[0]))   # true metric proportions
        ax.grid(True); ax.tick_params(labelsize=7)
        for w, isr in hands:                                  # spine: head -> each wrist
            wr = disp(w[WRIST], R, center3d)
            ax.plot([hd[0], wr[0]], [hd[1], wr[1]], [hd[2], wr[2]], color="#1f9d4d", lw=2)
        ax.scatter([hd[0]], [hd[1]], [hd[2]], color="#111", s=55, marker='o')   # HEAD (moves)
        ax.scatter([hd[0]], [hd[1]], [hd[2]], color="#1f9d4d", s=18, marker='o')
        for w, isr in hands:                                  # hands
            P = np.array([disp(p, R, center3d) for p in w])
            c = col_rgb[isr]
            for a, b in HAND_EDGES:
                ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]], color=c, lw=1.6)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=c, s=8)


def add_legend(fig):
    """Add the Left/Right/Spine/Head legend ONCE (figure-level; ax.clear() never removes it,
    so calling this per-frame would stack legends and slow drawing to O(N^2))."""
    fig.legend(handles=[plt.Line2D([], [], color="#1f5fd0", label="Left"),
                        plt.Line2D([], [], color="#e0342f", label="Right"),
                        plt.Line2D([], [], color="#1f9d4d", label="Spine (head→wrists)"),
                        plt.Line2D([], [], color="#111", marker='o', ls='', label="Head")],
               loc="lower center", ncol=4, fontsize=9, frameon=False)


def fig_to_bgr(fig, w, h):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--npz', required=True)
    ap.add_argument('--head', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--imu', default=None, help='imu_accel.csv for gravity-aligned panels (STRONGLY recommended)')
    ap.add_argument('--past-sec', type=float, default=0.0)
    ap.add_argument('--kp-frames', type=int, default=0, help='total frame count of the video the keypoints were decoded from (for timeline remap); 0 = infer from max frame_idx')
    ap.add_argument('--future-sec', type=float, default=3.0)   # customer reference shows the FUTURE 3s wrist path
    ap.add_argument('--max-frames', type=int, default=0)      # 0 = all
    ap.add_argument('--gate', type=float, default=150.0); ap.add_argument('--maxgap', type=int, default=10)
    ap.add_argument('--min-len', type=int, default=10)
    ap.add_argument('--min-shape', type=float, default=0.35); ap.add_argument('--min-mp', type=float, default=0.25)
    ap.add_argument('--foot-ymin', type=float, default=0.95); ap.add_argument('--foot-shape', type=float, default=0.6)
    ap.add_argument('--no-filter', action='store_true')
    ap.add_argument('--shape-filter', action='store_true', help='apply the bad_shape/foot drops (only for clips that may contain feet)')
    ap.add_argument('--clip-foot', action='store_true', help='CLIP image hand-vs-foot rejection (v2_clip parity)')
    ap.add_argument('--clip-thr', type=float, default=0.55); ap.add_argument('--clip-ymin', type=float, default=0.5)
    ap.add_argument('--clip-nsamp', type=int, default=6)
    args = ap.parse_args()

    npz = np.load(args.npz)
    fi, k2 = npz['frame_idx'], npz['kp2d']
    kp3d = npz['kp3d_cam']; K = npz['K']
    src = npz['source'] if 'source' in npz else None

    head = np.load(args.head); T = head['T'].astype(np.float64)
    hfi = head['frame_idx'].astype(int) if 'frame_idx' in head else np.arange(len(T))
    if not np.array_equal(hfi, np.arange(len(T))):
        Tmap = {int(f): T[i] for i, f in enumerate(hfi)}
        maxf = max(int(hfi.max()), int(fi.max())) + 1
        Td = np.tile(np.eye(4), (maxf, 1, 1))
        for f in range(maxf):
            if f in Tmap: Td[f] = Tmap[f]
        T = Td

    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    past_win = int(round(args.past_sec * fps)); future_win = int(round(args.future_sec * fps))

    # ---- frame-timeline sync ----
    # keypoint frame_idx values are ABSOLUTE video-frame indices (the extractor stamps the real frame
    # number and simply emits nothing for frames with no hand). So DO NOT infer the source length from
    # max(frame_idx): hands routinely drop out before the clip ends, which made the old heuristic guess
    # a short length (e.g. 1796 vs a real 1823-frame video) and then STRETCH every keypoint, sliding the
    # skeleton off the hand. Only remap when the caller explicitly passes --kp-frames (i.e. the keypoints
    # were genuinely decoded from a differently-lengthed copy). Otherwise treat frame_idx as absolute and
    # just drop any out-of-range detection so it can't index past the video/pose timeline.
    Nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.kp_frames and args.kp_frames > 1 and Nvid > 1 and abs(int(args.kp_frames) - Nvid) >= 2:
        fi = np.rint(fi.astype(np.float64) * (Nvid - 1) / (int(args.kp_frames) - 1)).astype(int)
        print(f"[sync] keypoints on {int(args.kp_frames)}-frame timeline -> remapped to video/pose {Nvid} frames")
    else:
        oob = int(np.sum(fi >= Nvid))
        if oob:
            keep = fi < Nvid
            fi = fi[keep]; k2 = k2[keep]; kp3d = kp3d[keep]
            if src is not None: src = src[keep]
            print(f"[sync] absolute frame_idx; dropped {oob} detection(s) beyond video ({Nvid} frames)")
        else:
            print(f"[sync] absolute frame_idx; {Nvid} video frames — no remap")

    # by-index lookups built AFTER the sync block so the drop branch stays consistent
    kp3d_by_gi = {i: kp3d[i] for i in range(len(fi))}
    src_by_gi = {i: str(src[i]) for i in range(len(fi))} if src is not None else {}

    tracks = build_tracks(fi, k2, kp3d, args.gate, args.maxgap, args.min_len)
    if not args.no_filter:
        tracks, _ = filter_tracks(tracks, kp3d_by_gi, src_by_gi, W, H, args)
        if args.clip_foot and tracks:                          # v2_clip parity: drop CLIP-detected feet
            scores = clip_foot_scores(tracks, args.video, args.clip_nsamp)
            keep, ndrop = [], 0
            for tr, sc in zip(tracks, scores):
                if sc == sc and sc > args.clip_thr and tr['feat']['wrist_y'] > args.clip_ymin:
                    ndrop += 1                                  # looks like a foot AND sits low in frame
                else:
                    keep.append(tr)
            print(f"[clip_foot] dropped {ndrop} foot-track(s); {len(keep)} kept")
            tracks = keep
    assign_handedness(tracks, W)

    if args.imu:
        up = imu_gravity_up(args.imu, T); print(f"[gravity] IMU up = {np.round(up,3)}")
    else:
        up = infer_world_up(T); print("[gravity] WARNING: no --imu; using unreliable image-up fallback")
    # orientation guard: the head (camera) is above the working hands — flip up if the sign is off
    wpos = [hand_world(tr, f, T)[WRIST] for tr in tracks for f in tr['frames']
            if hand_world(tr, f, T) is not None]
    if wpos and (T[:, :3, 3].mean(0) - np.mean(wpos, 0)) @ up < 0:
        up = -up; print("[gravity] flipped up so head sits above hands")
    R = align_basis(up)

    # ---- world-fixed panel frame: one scene centre + auto-scaled axes that contain the WHOLE head
    # path (so the head/camera MOVES through the frame as the person walks, instead of being pinned
    # to the centre) plus arm reach. Head height is ~constant while walking; hands hang below. ----
    heads_w = T[:, :3, 3]
    center3d = heads_w.mean(0)
    head_path = np.array([disp(h, R, center3d) for h in heads_w])   # display coords, per frame
    REACH = 0.7                                                     # arm reach beyond the head (m)
    lim = float(min(max(np.abs(head_path[:, :2]).max() + REACH, 0.6), 3.0))
    # fit z to the actual head path + room for hands hanging below; NO floor clamp, so if the SLAM
    # head drifts (a real signal the customer wants to see) the head stays in frame instead of
    # vanishing under a fake floor.
    zmax = float(head_path[:, 2].max() + 0.20)
    zmin = float(head_path[:, 2].min() - 0.70)
    zlim = (zmin, zmax)
    print(f"[panels] scene centre={np.round(center3d,2)} lim=±{lim:.2f}m z=[{zmin:.2f},{zmax:.2f}] "
          f"head-path span={np.round(np.ptp(head_path[:, :2], 0),2)}m")

    RW = int(round(H * 0.9))                                  # right (panels) width
    fig = plt.figure(figsize=(RW / 100, H / 100), dpi=100)
    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.08, wspace=0.15, hspace=0.20)
    add_legend(fig)                                          # once, not per frame

    stem = os.path.splitext(os.path.basename(args.video))[0]; os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, f'{stem}.v4.tmp.mp4'); final = os.path.join(args.out, f'{stem}_wrist_traj_panels.mp4')
    OW = W + RW
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OW, H))
    nL = sum(1 for t in tracks if t['hand'] == 0)
    print(f"[{stem}] v4 panels {OW}x{H}@{fps:.1f} | tracks={len(tracks)} (L={nL} R={len(tracks)-nL}) | poses={len(T)}")

    t = 0; TRAIL_ALPHA = 0.72
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and t >= args.max_frames: break
        # LEFT: ego + world-frame trail + 21-kp skeleton
        overlay = frame.copy(); drew = False
        for tr in tracks:
            drew |= draw_traj_world(overlay, tr, t, T, K, past_win, future_win, W, H)
        if drew: cv2.addWeighted(overlay, TRAIL_ALPHA, frame, 1 - TRAIL_ALPHA, 0, frame)
        for tr in tracks:
            if t in tr['kp']:
                draw_skeleton(frame, tr['kp'][t], color(tr['hand_at'].get(t, tr['hand'])))
        # RIGHT: 4 world-frame panels (head moves along its path)
        draw_panels(fig, axes, tracks, t, T, R, center3d, lim, zlim, head_path)
        panels = fig_to_bgr(fig, RW, H)
        canvas = np.zeros((H, OW, 3), np.uint8); canvas[:] = (255, 255, 255)
        canvas[:, :W] = frame; canvas[:, W:] = panels
        wr.write(canvas); t += 1
    cap.release(); wr.release(); plt.close(fig)
    if shutil.which('ffmpeg'):
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '20', '-g', '15', '-movflags', '+faststart', final],
                           capture_output=True)
        os.remove(tmp) if r.returncode == 0 else shutil.move(tmp, final)
    print(f"[{stem}] wrote {final} ({t} frames)")


if __name__ == '__main__':
    main()
