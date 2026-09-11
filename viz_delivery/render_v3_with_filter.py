#!/usr/bin/env python
"""
v3 wrist-trajectory renderer — WORLD-FRAME trajectory (fixes customer feedback).

v2 drew the trajectory by painting each frame's wrist IMAGE PIXEL (kp2d) onto a different
frame, so the trail was locked to the camera/image plane and slid as the head moved. The
customer requires the trajectory in the WORLD frame, projected into the CURRENT camera view,
so it stays "frozen" in 3D space — which is exactly what verifies the SLAM camera track.

v3 keeps ALL of v2's cleanup (per-frame handedness, foot/false-positive rejection) and changes
ONLY the trajectory: each wrist position is lifted to the world using the SLAM head pose
T_world_cam at the frame it was observed, then re-projected into the camera at the CURRENT
frame with the pinhole intrinsics K. If SLAM is correct the breadcrumbs stick to the physical
scene; if the camera track drifts they visibly slide (the intended diagnostic).

    world_f = T[f][:3,:3] @ kp3d_cam_wrist[f] + T[f][:3,3]     # observed wrist -> world
    cam_t   = T[t][:3,:3].T @ (world_f - T[t][:3,3])           # world -> current camera
    u = fx*cam_t[0]/cam_t[2] + cx ;  v = fy*cam_t[1]/cam_t[2] + cy      # (cam_t[2] > 0)

Inputs: the SAME enhanced npz as v2 (needs kp3d_cam + K) PLUS the head 6-DoF poses
(head_pose_6dof.npz: T, frame_idx) from the DROID-SLAM head-pose job.

Usage: render_v3.py --video left_eye.mp4 --npz *_enhanced_keypoints.npz \
                    --head head_pose_6dof.npz --out DIR [--past-sec 4] [--future-sec 0]
"""
import argparse, os, shutil, subprocess
import numpy as np
import cv2

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20)]
BLUE = (255, 0, 0); RED = (0, 0, 255)          # BGR: left=blue, right=red
def color(is_right): return RED if is_right else BLUE
WRIST = 0
MCPS = [1, 5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]


# ---------------------------------------------------------------- tracking (v2, + carries kp3d)
def build_tracks(fi, k2, k3, gate=150.0, maxgap=10, min_len=8):
    """Greedy wrist-proximity tracking. Carries per-frame 3D wrist (k3, camera frame) so the
    trajectory can be lifted to the world. handedness assigned later."""
    order = np.argsort(fi, kind='stable'); tk = []
    for gi in order:
        f = int(fi[gi]); xy = k2[gi, 0]
        best, bd = None, gate
        for t in tk:
            if 0 < f - t['lf'] <= maxgap:
                d = float(np.hypot(*(t['lxy'] - xy)))
                if d < bd: best, bd = t, d
        if best is None:
            tk.append(dict(lxy=xy, lf=f, idx=[gi]))
        else:
            best['lxy'] = xy; best['lf'] = f; best['idx'].append(gi)
    out = []
    for t in tk:
        if len(t['idx']) < min_len: continue
        kp = {int(fi[i]): k2[i] for i in t['idx']}
        kp3 = {int(fi[i]): (k3[i] if k3 is not None else None) for i in t['idx']}
        out.append(dict(kp=kp, kp3=kp3, frames=sorted(kp), idx=t['idx']))
    return out


# ---------------------------------------------------------------- plausibility (v2, unchanged)
def hand_shape_score(kp):
    palm = np.linalg.norm(kp[MCPS[1]] - kp[MCPS[4]]) + 1e-6
    spans = [np.linalg.norm(kp[t] - kp[WRIST]) for t in TIPS]
    r = np.array(spans) / palm
    if r.mean() < 1.0 or r.mean() > 6.0: return 0.0
    cv = r.std() / (r.mean() + 1e-6)
    return float(max(0.0, 1.0 - cv))


def track_features(tr, kp3d_by_gi, src_by_gi, H):
    kps = [tr['kp'][f] for f in tr['frames']]
    shape = np.median([hand_shape_score(k) for k in kps])
    wrist_y = np.median([k[WRIST, 1] for k in kps]) / H
    zs = [kp3d_by_gi[i][WRIST, 2] for i in tr['idx'] if i in kp3d_by_gi]
    depth = float(np.median(zs)) if zs else np.nan
    srcs = [src_by_gi.get(i, 'wilor') for i in tr['idx']]
    mp_support = np.mean([s in ('fused', 'lifted_2d') for s in srcs]) if srcs else 0.0
    return dict(shape=shape, wrist_y=wrist_y, depth=depth, mp_support=mp_support, n=len(tr['frames']))


def filter_tracks(tracks, kp3d_by_gi, src_by_gi, W, H, args):
    kept, dropped = [], []
    for tr in tracks:
        f = track_features(tr, kp3d_by_gi, src_by_gi, H)
        tr['feat'] = f
        reasons = []
        if f['n'] < args.min_len:                                  reasons.append('short')
        # only drop a WiLoR-only track if it ALSO fails the hand-shape check — a well-shaped hand that
        # MediaPipe simply MISSED (e.g. gripping a tool from a back-of-hand egocentric view) is real and
        # must be kept (mirrors the mp_support gate already applied to the shape/foot drops below).
        if f['mp_support'] < args.min_mp and f['shape'] < args.min_shape:  reasons.append('wilor_only')
        # shape-based drops (bad_shape, foot): applied ONLY to clips flagged --shape-filter (the ones
        # that may contain feet) AND only to poorly-corroborated tracks. A hand fully corroborated by
        # MediaPipe (mp_support high) is real even when it GRIPS a tool — clenched fingers sit near the
        # wrist so hand_shape_score() returns 0, which used to nuke every construction/gripping clip.
        # _with_filter build: the shape/foot drops are ALWAYS applied (this variant targets episodes
        # where a foot was tracked as a hand). Still guarded by mp_support < 0.5 so a MediaPipe-confirmed
        # tool-gripping hand (clenched fingers -> shape~0) is never mistaken for a foot.
        if f['mp_support'] < 0.5:
            if f['shape'] < args.min_shape:                        reasons.append('bad_shape')
            if f['wrist_y'] > args.foot_ymin and f['shape'] < args.foot_shape:  reasons.append('foot')
        if reasons: tr['drop'] = reasons; dropped.append(tr)
        else: kept.append(tr)
    return kept, dropped


# ---------------------------------------------------------------- handedness (v2, unchanged)
def assign_handedness(tracks, W):
    byf = {}
    for ti, tr in enumerate(tracks):
        for fr in tr['frames']:
            byf.setdefault(fr, []).append((ti, float(tr['kp'][fr][WRIST, 0])))
    per = [dict() for _ in tracks]
    for fr, lst in byf.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda z: z[1])
        per[lst[0][0]][fr] = 0
        per[lst[-1][0]][fr] = 1
        for ti, x in lst[1:-1]:
            per[ti][fr] = 0 if x < W / 2 else 1
    for ti, tr in enumerate(tracks):
        lab = list(per[ti].values())
        maj = (int(np.mean(lab) >= 0.5) if lab
               else int(np.median([tr['kp'][f][WRIST, 0] for f in tr['frames']]) >= W / 2))
        tr['hand'] = maj
        for fr in tr['frames']:
            per[ti].setdefault(fr, maj)
        tr['hand_at'] = per[ti]


# ---------------------------------------------------------------- drawing
def draw_skeleton(img, kp, col):
    for a, b in HAND_EDGES:
        cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), col, 2, cv2.LINE_AA)
    for x, y in kp:
        cv2.circle(img, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)


def draw_traj_world(overlay, track, t, T, K, past_win, future_win, W, H):
    """WORLD-FRAME wrist trajectory projected into the CURRENT (frame t) camera view.

    Each observed wrist (camera-frame metric 3D at its own frame f) is lifted to the world via
    the SLAM pose T[f], then re-projected into the frame-t camera via T[t] and K. World-anchored
    breadcrumbs => the trail stays glued to the physical scene as the head moves."""
    if t >= len(T) or t not in track['kp3'] or track['kp3'][t] is None:
        return False
    Rt = T[t][:3, :3]; tt = T[t][:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    col = color(track['hand_at'].get(t, track['hand']))          # per-frame L/R (v2 fix)

    frames = [f for f in track['frames']
              if (t - past_win) <= f <= (t + future_win)
              and f < len(T) and track['kp3'].get(f) is not None]
    if len(frames) < 2:
        return False

    proj = []                                                    # (u, v, f) in front of camera
    for f in frames:
        w = np.asarray(track['kp3'][f][WRIST], float)            # wrist, camera frame @ f (metres)
        world = T[f][:3, :3] @ w + T[f][:3, 3]                   # -> world
        camt = Rt.T @ (world - tt)                               # -> current camera
        Z = camt[2]
        if Z <= 1e-3:                                            # behind the current camera
            continue
        u = fx * camt[0] / Z + cx; v = fy * camt[1] / Z + cy
        if -40 <= u <= W + 40 and -40 <= v <= H + 40:            # keep only points in/near the frame
            proj.append((u, v, f))                               # (world points off-screen are culled,
    if len(proj) < 2:
        return False

    # DOTS ONLY (no connecting lines) — like the customer's reference. Connecting lines over a
    # walking window traced the forearm and looked like an arm; frozen breadcrumb dots read clean.
    for (u, v, f) in proj:
        fade = 1.0 - 0.6 * (abs(f - t) / max(past_win, future_win, 1))
        c = tuple(int(255 - (255 - ch) * max(fade, 0.2)) for ch in col)
        r = 6 if f == t else 3
        cv2.circle(overlay, (int(u), int(v)), r, c, -1, cv2.LINE_AA)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--npz', required=True)
    ap.add_argument('--head', required=True, help='head_pose_6dof.npz (T, frame_idx) from the SLAM head-pose job')
    ap.add_argument('--out', required=True)
    ap.add_argument('--past-sec', type=float, default=4.0)       # trailing world breadcrumbs
    ap.add_argument('--future-sec', type=float, default=0.0)     # optional upcoming path
    ap.add_argument('--no-skeleton', action='store_true')
    ap.add_argument('--gate', type=float, default=150.0); ap.add_argument('--maxgap', type=int, default=10)
    ap.add_argument('--min-len', type=int, default=10)
    ap.add_argument('--min-shape', type=float, default=0.35)
    ap.add_argument('--min-mp', type=float, default=0.25)
    ap.add_argument('--foot-ymin', type=float, default=0.95)
    ap.add_argument('--foot-shape', type=float, default=0.6)
    ap.add_argument('--no-filter', action='store_true')
    ap.add_argument('--meta-out', default=None)
    args = ap.parse_args()

    npz = np.load(args.npz)
    fi, k2 = npz['frame_idx'], npz['kp2d']
    kp3d = npz['kp3d_cam'] if 'kp3d_cam' in npz else None
    if kp3d is None:
        raise SystemExit("npz has no kp3d_cam — world-frame trajectory needs metric 3D keypoints")
    K = npz['K'] if 'K' in npz else None
    if K is None:
        raise SystemExit("npz has no K (intrinsics) — needed to project world points into the view")
    src = npz['source'] if 'source' in npz else None
    kp3d_by_gi = {i: kp3d[i] for i in range(len(fi))}
    src_by_gi = {i: str(src[i]) for i in range(len(fi))} if src is not None else {}

    head = np.load(args.head)
    T = head['T'].astype(np.float64)                             # (N,4,4) camera-to-world, frame-indexed
    # align: head_pose frame_idx should be 0..N-1 matching the video; index T by absolute frame number
    hfi = head['frame_idx'].astype(int) if 'frame_idx' in head else np.arange(len(T))
    if not np.array_equal(hfi, np.arange(len(T))):
        Tmap = {int(f): T[i] for i, f in enumerate(hfi)}         # sparse -> dense by frame number
        maxf = max(hfi.max(), int(fi.max())) + 1
        Td = np.tile(np.eye(4), (maxf, 1, 1))
        for f in range(maxf):
            if f in Tmap: Td[f] = Tmap[f]
        T = Td

    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    past_win = int(round(args.past_sec * fps)); future_win = int(round(args.future_sec * fps))

    tracks = build_tracks(fi, k2, kp3d, args.gate, args.maxgap, args.min_len)
    dropped = []
    if not args.no_filter:
        tracks, dropped = filter_tracks(tracks, kp3d_by_gi, src_by_gi, W, H, args)
        for tr in dropped:
            print(f"  [drop {','.join(tr['drop'])}] n={tr['feat']['n']} shape={tr['feat']['shape']:.2f} "
                  f"y={tr['feat']['wrist_y']:.2f} mp={tr['feat']['mp_support']:.2f}")
    assign_handedness(tracks, W)

    stem = os.path.splitext(os.path.basename(args.video))[0]; os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, f'{stem}.v3.tmp.mp4'); final = os.path.join(args.out, f'{stem}_wrist_traj_v3.mp4')
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    nL = sum(1 for t in tracks if t['hand'] == 0); nR = len(tracks) - nL
    print(f"[{stem}] v3 (WORLD-frame traj) {W}x{H}@{fps:.1f} | {len(tracks)} tracks (L={nL} R={nR}) | poses={len(T)}")
    t = 0; TRAIL_ALPHA = 0.72
    while True:
        ok, frame = cap.read()
        if not ok: break
        overlay = frame.copy(); drew = False
        for tr in tracks:
            drew |= draw_traj_world(overlay, tr, t, T, K, past_win, future_win, W, H)
        if drew: cv2.addWeighted(overlay, TRAIL_ALPHA, frame, 1 - TRAIL_ALPHA, 0, frame)
        if not args.no_skeleton:
            for tr in tracks:
                if t in tr['kp']: draw_skeleton(frame, tr['kp'][t], color(tr['hand_at'].get(t, tr['hand'])))
        cv2.putText(frame, f'f{t}  BLUE=L  RED=R  world-frame trail {args.past_sec:.0f}s  [v3]', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        wr.write(frame); t += 1
    cap.release(); wr.release()
    if shutil.which('ffmpeg'):
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '20', '-g', '15', '-movflags', '+faststart', final], capture_output=True)
        os.remove(tmp) if r.returncode == 0 else shutil.move(tmp, final)
    print(f"[{stem}] wrote {final} ({t} frames)")
    if args.meta_out:
        import json, collections
        drops = collections.Counter(r for tr in dropped for r in tr['drop'])
        json.dump(dict(version='v3', frame_of_reference='world (SLAM T_world_cam) projected into current view',
                       frames=t, tracks_kept=len(tracks), tracks_dropped=len(dropped),
                       drop_reasons=dict(drops), left_tracks=nL, right_tracks=nR,
                       past_sec=args.past_sec, future_sec=args.future_sec),
                  open(args.meta_out, 'w'), indent=2)


if __name__ == '__main__':
    main()
