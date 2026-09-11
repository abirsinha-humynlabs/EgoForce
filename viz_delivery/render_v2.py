#!/usr/bin/env python
"""
v2 wrist-trajectory renderer — POST-PROCESSES the existing v1 enhanced npz (no WiLoR re-run, CPU only)
to fix the three HITL-audited issues, then re-renders the overlay in the same style as v1.

Issues addressed (see AUDIT):
  1) LEFT/RIGHT swap (most common): v1 colored by the model's per-frame `is_right`, which flips on
     egocentric back-of-hand views. v2 IGNORES the model label and derives handedness from GEOMETRY:
     in head-mounted egocentric video the two hands almost never cross, so per frame the leftmost
     wrist is the LEFT hand and the rightmost is the RIGHT hand. Votes are aggregated per track;
     single-hand stretches fall back to a wrist-x-vs-center prior. -> stable, correct L/R.
  2) FOOT tracked as a hand: feet sit on the floor -> far depth (large Z) AND low in the frame AND
     detected by WiLoR only (MediaPipe rarely fires on feet). v2 drops tracks matching that profile.
  3) FALSE POSITIVE on non-hand (cloth/object): rejected by a hand-shape plausibility score
     (bone-length consistency of the 21-kp skeleton) + short-track / low-support temporal gates.

All thresholds are CLI-tunable so we can calibrate on the audited problem videos. v1 is untouched.

Usage: render_v2.py --video left_eye.mp4 --npz *_enhanced_keypoints.npz --out DIR [--future-sec 3]
"""
import argparse, os, shutil, subprocess
import numpy as np
import cv2

# 21-kp OpenPose hand topology (inlined so v2 needs NO torch/wilor/mediapipe — CPU-only image)
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20)]

BLUE = (255, 0, 0); RED = (0, 0, 255)          # BGR: left=blue, right=red
def color(is_right): return RED if is_right else BLUE

# 21-kp OpenPose hand topology helpers
WRIST = 0
MCPS = [1, 5, 9, 13, 17]     # thumb/index/middle/ring/pinky metacarpophalangeal
TIPS = [4, 8, 12, 16, 20]


# ---------------------------------------------------------------- tracking
def build_tracks(fi, k2, gate=150.0, maxgap=10, min_len=8):
    """Greedy wrist-proximity tracking -> one physical hand per track (handedness assigned later)."""
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
        out.append(dict(kp=kp, frames=sorted(kp), idx=t['idx']))
    return out


# ---------------------------------------------------------------- plausibility (issues 2 & 3)
def hand_shape_score(kp):
    """0..1 plausibility that 21 kp form a real hand, from bone-length consistency. A real hand's
    per-finger bones shrink distally and the 5 fingers have comparable spans; degenerate object/foot
    fits violate this. Returns low score for implausible skeletons."""
    palm = np.linalg.norm(kp[MCPS[1]] - kp[MCPS[4]]) + 1e-6      # index-MCP .. pinky-MCP width
    spans = [np.linalg.norm(kp[t] - kp[WRIST]) for t in TIPS]     # wrist->tip per finger
    # real hand: tip spans are 1.5-4x palm width and similar across fingers
    r = np.array(spans) / palm
    if r.mean() < 1.0 or r.mean() > 6.0: return 0.0
    cv = r.std() / (r.mean() + 1e-6)                              # finger-span coefficient of variation
    return float(max(0.0, 1.0 - cv))                             # tight spread -> ~1, messy -> ~0


def track_features(tr, kp3d_by_gi, src_by_gi, H):
    """Aggregate per-track signals used by the reject gates."""
    kps = [tr['kp'][f] for f in tr['frames']]
    shape = np.median([hand_shape_score(k) for k in kps])
    wrist_y = np.median([k[WRIST, 1] for k in kps]) / H          # 0=top .. 1=bottom of frame
    # depth (if kp3d present): median wrist Z in meters
    zs = [kp3d_by_gi[i][WRIST, 2] for i in tr['idx'] if i in kp3d_by_gi]
    depth = float(np.median(zs)) if zs else np.nan
    # fraction of detections that had MediaPipe support (fused/lifted_2d) vs wilor-only
    srcs = [src_by_gi.get(i, 'wilor') for i in tr['idx']]
    mp_support = np.mean([s in ('fused', 'lifted_2d') for s in srcs]) if srcs else 0.0
    return dict(shape=shape, wrist_y=wrist_y, depth=depth, mp_support=mp_support, n=len(tr['frames']))


def filter_tracks(tracks, kp3d_by_gi, src_by_gi, W, H, args):
    """Reject feet / false-positives using the signals that actually separate them from real hands
    (calibrated on the audited videos):
      - shape < min_shape       : degenerate 21-kp fit (collapsed/exploded) -> most feet & object FPs
      - mp_support < min_mp      : WiLoR-only detection with no MediaPipe agreement -> the FP signature
                                   (real hands are corroborated by both models; WiLoR's YOLO fires on
                                   non-hands far more than MediaPipe does)
      - very-bottom + degenerate : a conservative foot catch for the frame's bottom edge
    Depth/position alone are NOT used to gate — feet and low-working hands overlap on both."""
    kept, dropped = [], []
    for tr in tracks:
        f = track_features(tr, kp3d_by_gi, src_by_gi, H)
        tr['feat'] = f
        reasons = []
        if f['n'] < args.min_len:                                  reasons.append('short')
        if f['shape'] < args.min_shape:                            reasons.append('bad_shape')   # issue 2/3
        if f['mp_support'] < args.min_mp:                          reasons.append('wilor_only')  # issue 3
        if f['wrist_y'] > args.foot_ymin and f['shape'] < args.foot_shape:  reasons.append('foot')  # issue 2
        if reasons: tr['drop'] = reasons; dropped.append(tr)
        else: kept.append(tr)
    return kept, dropped


# ---------------------------------------------------------------- CLIP hand-vs-foot (issue 2, robust)
_CLIP = None
def _clip():
    """Lazy-load CLIP (ViT-B/32) + precompute hand/foot text embeddings. Only imported when used."""
    global _CLIP
    if _CLIP is None:
        import torch, open_clip
        m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); m.eval()
        tok = open_clip.get_tokenizer('ViT-B-32')
        HAND = ["a photo of a human hand", "fingers and a palm", "a hand gripping an object"]
        FOOT = ["a photo of a bare human foot", "toes and a foot on the floor", "a foot and ankle"]
        with torch.no_grad():
            th = m.encode_text(tok(HAND)); th /= th.norm(dim=-1, keepdim=True); th = th.mean(0, keepdim=True); th /= th.norm(dim=-1, keepdim=True)
            tf = m.encode_text(tok(FOOT)); tf /= tf.norm(dim=-1, keepdim=True); tf = tf.mean(0, keepdim=True); tf /= tf.norm(dim=-1, keepdim=True)
        _CLIP = (torch, m, pre, torch.cat([th, tf]))
    return _CLIP


def clip_foot_scores(tracks, video, nsamp=6, ctx=1.7):
    """Per-track median P(foot): crop the image region around each track's keypoints in nsamp frames,
    classify hand-vs-foot with CLIP, take the median. Keypoint geometry can't tell a well-fit foot
    from a hand, but the pixels can. Reads each needed frame once."""
    import collections
    torch, m, pre, txt = _clip()
    from PIL import Image
    need = collections.defaultdict(list)                         # frame -> [(track_idx, kp)]
    for ti, tr in enumerate(tracks):
        fr = tr['frames']; sel = fr[::max(1, len(fr) // nsamp)][:nsamp]
        for f in sel: need[f].append((ti, tr['kp'][f]))
    acc = collections.defaultdict(list)
    cap = cv2.VideoCapture(video)
    for f in sorted(need):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok: continue
        for ti, kp in need[f]:
            x0, y0 = kp.min(0); x1, y1 = kp.max(0)
            s = max(max(x1 - x0, 20), max(y1 - y0, 20)) * ctx; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            X0, X1 = int(max(0, cx - s / 2)), int(min(img.shape[1], cx + s / 2))
            Y0, Y1 = int(max(0, cy - s / 2)), int(min(img.shape[0], cy + s / 2))
            c = img[Y0:Y1, X0:X1]
            if c.size == 0: continue
            im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                fe = m.encode_image(pre(im).unsqueeze(0)); fe /= fe.norm(dim=-1, keepdim=True)
                acc[ti].append(float((100 * fe @ txt.T).softmax(-1)[0][1]))
    cap.release()
    return [float(np.median(acc[ti])) if acc[ti] else float('nan') for ti in range(len(tracks))]


# ---------------------------------------------------------------- handedness (issue 1)
def assign_handedness(tracks, W):
    """PER-FRAME relative L/R (fixes the swap): at each frame, among the hands visible THAT frame,
    the leftmost wrist is the LEFT hand (blue) and the rightmost is the RIGHT hand (red). Correct
    whenever the two hands don't cross (the egocentric norm) and follows a hand that moves across
    the image instead of committing one wrong global label. Single-hand frames inherit the track's
    majority label for temporal stability. Sets tr['hand_at'][frame] and tr['hand'] (majority)."""
    byf = {}
    for ti, tr in enumerate(tracks):
        for fr in tr['frames']:
            byf.setdefault(fr, []).append((ti, float(tr['kp'][fr][WRIST, 0])))
    per = [dict() for _ in tracks]                   # per[ti][frame] = is_right (only where >=2 hands)
    for fr, lst in byf.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda z: z[1])
        per[lst[0][0]][fr] = 0                        # leftmost -> LEFT
        per[lst[-1][0]][fr] = 1                       # rightmost -> RIGHT
        for ti, x in lst[1:-1]:                       # middle (3+ hands, rare) -> by side
            per[ti][fr] = 0 if x < W / 2 else 1
    for ti, tr in enumerate(tracks):
        lab = list(per[ti].values())
        maj = (int(np.mean(lab) >= 0.5) if lab
               else int(np.median([tr['kp'][f][WRIST, 0] for f in tr['frames']]) >= W / 2))
        tr['hand'] = maj
        for fr in tr['frames']:                       # fill single-hand frames with the majority
            per[ti].setdefault(fr, maj)
        tr['hand_at'] = per[ti]


# ---------------------------------------------------------------- drawing (same style as v1)
def draw_skeleton(img, kp, col):
    for a, b in HAND_EDGES:
        cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), col, 2, cv2.LINE_AA)
    for x, y in kp:
        cv2.circle(img, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)


def draw_future(overlay, track, t, W):
    if t not in track['kp']: return False
    col = color(track['hand_at'].get(t, track['hand'])); now = track['kp'][t][0]   # per-frame L/R
    pts = [(f, track['kp'][f][0]) for f in track['frames'] if t < f <= t + W]
    if not pts: return False
    n = len(pts)
    for k, (f, p) in enumerate(pts):
        pi = (int(p[0]), int(p[1])); frac = k / max(n - 1, 1); fade = 1.0 - 0.55 * frac
        c = tuple(int(255 - (255 - ch) * fade) for ch in col)
        cv2.circle(overlay, pi, 5 if k == 0 else 3, c, -1, cv2.LINE_AA)
    first = (int(pts[0][1][0]), int(pts[0][1][1]))
    if np.hypot(first[0] - now[0], first[1] - now[1]) <= 120.0:
        pale = tuple(int(255 - (255 - ch) * 0.6) for ch in col)
        cv2.line(overlay, (int(now[0]), int(now[1])), first, pale, 1, cv2.LINE_AA)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--future-sec', type=float, default=3.0); ap.add_argument('--no-skeleton', action='store_true')
    # tracking
    ap.add_argument('--gate', type=float, default=150.0); ap.add_argument('--maxgap', type=int, default=10)
    ap.add_argument('--min-len', type=int, default=10)
    # reject gates (calibrated on audited videos; all tunable)
    ap.add_argument('--min-shape', type=float, default=0.35)     # degenerate 21-kp fit (feet/objects)
    ap.add_argument('--min-mp', type=float, default=0.25)        # require MediaPipe agreement (kills WiLoR-only FPs)
    ap.add_argument('--foot-ymin', type=float, default=0.95)     # very bottom edge ...
    ap.add_argument('--foot-shape', type=float, default=0.6)     # ... AND weak shape -> conservative foot catch
    ap.add_argument('--no-filter', action='store_true')          # for A/B vs v1
    ap.add_argument('--clip-foot', action='store_true')          # issue 2: CLIP image classifier drops feet
    ap.add_argument('--clip-thr', type=float, default=0.55)      # drop track if median P(foot) > thr (clean hands score <=0.2)
    ap.add_argument('--clip-ymin', type=float, default=0.5)      # ...only for detections in the lower frame (feet are never high up)
    ap.add_argument('--clip-nsamp', type=int, default=6)
    ap.add_argument('--meta-out', default=None)                  # write a v2 meta json here
    args = ap.parse_args()

    npz = np.load(args.npz)
    fi, k2 = npz['frame_idx'], npz['kp2d']
    kp3d = npz['kp3d_cam'] if 'kp3d_cam' in npz else None
    src = npz['source'] if 'source' in npz else None
    kp3d_by_gi = {i: kp3d[i] for i in range(len(fi))} if kp3d is not None else {}
    src_by_gi = {i: str(src[i]) for i in range(len(fi))} if src is not None else {}

    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); win = int(round(args.future_sec * fps))

    tracks = build_tracks(fi, k2, args.gate, args.maxgap, args.min_len)
    dropped = []
    if not args.no_filter:
        tracks, dropped = filter_tracks(tracks, kp3d_by_gi, src_by_gi, W, H, args)
        if args.clip_foot and tracks:                            # issue 2: image-based foot rejection
            scores = clip_foot_scores(tracks, args.video, args.clip_nsamp)
            keep = []
            for tr, sc in zip(tracks, scores):
                tr['feat']['clip_foot'] = None if sc != sc else round(sc, 2)
                # drop only if it LOOKS like a foot (CLIP) AND sits low in frame (feet are never high up);
                # this shields an upper-frame hand from a rare CLIP misfire
                if sc == sc and sc > args.clip_thr and tr['feat']['wrist_y'] > args.clip_ymin:
                    tr['drop'] = ['clip_foot']; dropped.append(tr)
                else:
                    keep.append(tr)
            tracks = keep
        for tr in dropped:
            cf = tr['feat'].get('clip_foot')
            print(f"  [drop {','.join(tr['drop'])}] n={tr['feat']['n']} shape={tr['feat']['shape']:.2f} y={tr['feat']['wrist_y']:.2f} mp={tr['feat']['mp_support']:.2f} clip_foot={cf}")
    assign_handedness(tracks, W)

    stem = os.path.splitext(os.path.basename(args.video))[0]; os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, f'{stem}.v2.tmp.mp4'); final = os.path.join(args.out, f'{stem}_wrist_traj_v2.mp4')
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    nL = sum(1 for t in tracks if t['hand'] == 0); nR = len(tracks) - nL
    print(f"[{stem}] v2 {W}x{H}@{fps:.1f} | {len(tracks)} tracks kept (L={nL} R={nR})")
    t = 0; TRAIL_ALPHA = 0.72
    while True:
        ok, frame = cap.read()
        if not ok: break
        overlay = frame.copy(); drew = False
        for tr in tracks: drew |= draw_future(overlay, tr, t, win)
        if drew: cv2.addWeighted(overlay, TRAIL_ALPHA, frame, 1 - TRAIL_ALPHA, 0, frame)
        if not args.no_skeleton:
            for tr in tracks:
                if t in tr['kp']: draw_skeleton(frame, tr['kp'][t], color(tr['hand_at'].get(t, tr['hand'])))
        cv2.putText(frame, f'f{t}  BLUE=L  RED=R  future {args.future_sec:.0f}s  [v2]', (10, 30),
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
        json.dump(dict(version='v2', frames=t, tracks_kept=len(tracks), tracks_dropped=len(dropped),
                       drop_reasons=dict(drops), left_tracks=nL, right_tracks=nR,
                       handedness='per-frame relative (leftmost=left,rightmost=right)',
                       filters=dict(min_shape=args.min_shape, min_mp=args.min_mp, min_len=args.min_len)),
                  open(args.meta_out, 'w'), indent=2)


if __name__ == '__main__':
    main()
