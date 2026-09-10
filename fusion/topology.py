"""Shared OpenPose-21 hand topology, drawing helpers and colours.

Three producers in this experiment all emit the SAME 21-keypoint order, which is what makes the
comparison and the fusion apples-to-apples. Verified, not assumed:

* **EgoForce** — ``models/mano_layer.py`` maps the raw MANO joints through
  ``mano_joint_mapping = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]``
  with fingertips appended thumb, index, middle, ring, pinky
  (``MANO_FINGERTIP_VERT_INDICES``). Working that through lands exactly on the order below.
* **RTMPose-m Hand5** — trained with ``configs/_base_/datasets/coco_wholebody_hand.py``, whose
  keypoint names are ``wrist, thumb1..4, forefinger1..4, middle_finger1..4, ring_finger1..4,
  pinky_finger1..4``. Same order.
* **MediaPipe HandLandmarker** — same order (this is what the existing mono-pipeline relies on).

So no remapping is needed anywhere. Do NOT reorder without changing every producer.

    0            = wrist
    1,2,3,4      = thumb   (MCP, PIP, DIP, TIP)
    5,6,7,8      = index   (MCP, PIP, DIP, TIP)
    9,10,11,12   = middle
    13,14,15,16  = ring
    17,18,19,20  = pinky
"""

import numpy as np

NUM_HAND_JOINTS = 21
NUM_ARM_JOINTS = 3

JOINT_NAMES = [
    'wrist',
    'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

WRIST = 0
MCPS = [1, 5, 9, 13, 17]
PIPS = [2, 6, 10, 14, 18]
DIPS = [3, 7, 11, 15, 19]
TIPS = [4, 8, 12, 16, 20]

# (name, joint chain from the wrist outward, BGR colour)
FINGER_CHAINS = [
    ('thumb', (0, 1, 2, 3, 4), (60, 76, 231)),
    ('index', (0, 5, 6, 7, 8), (49, 176, 243)),
    ('middle', (0, 9, 10, 11, 12), (72, 201, 116)),
    ('ring', (0, 13, 14, 15, 16), (219, 152, 52)),
    ('pinky', (0, 17, 18, 19, 20), (182, 89, 155)),
]

# The 20 finger edges, in the same order as the existing pipeline's hand_topology.HAND_EDGES.
HAND_EDGES = [(a, b) for _, chain, _ in FINGER_CHAINS for a, b in zip(chain[:-1], chain[1:])]

# Optional palm cross-links. Not part of HAND_EDGES because the existing pipeline does not draw them;
# they only make the overlay easier to read.
PALM_CHAIN = (5, 9, 13, 17)
PALM_EDGES = list(zip(PALM_CHAIN[:-1], PALM_CHAIN[1:]))
PALM_COLOR = (190, 190, 190)

HAND_ORDER = ['left', 'right']
HAND_LABEL_COLORS = {'left': (255, 214, 120), 'right': (140, 210, 255)}

# Bones used for the bone-length consistency check. A fused hand whose bone lengths drift frame to
# frame is not a hand any more, which is the exact failure the rigid-PnP fusion was designed to avoid
# and that the articulated refit must not reintroduce.
BONES = HAND_EDGES


def skeleton_edges(include_palm=True):
    """Return the (E, 2) joint-index pairs used to draw the hand skeleton."""
    edges = list(HAND_EDGES) + (PALM_EDGES if include_palm else [])
    return np.asarray(edges, dtype=np.int32)


def bone_lengths(kp3d):
    """(..., 21, 3) -> (..., 20) Euclidean length of each finger bone."""
    kp3d = np.asarray(kp3d)
    a = np.asarray([e[0] for e in BONES])
    b = np.asarray([e[1] for e in BONES])
    return np.linalg.norm(kp3d[..., a, :] - kp3d[..., b, :], axis=-1)


def pinhole_K(fx, fy, cx, cy):
    """Match the existing pipeline's hand_topology.pinhole_K exactly (float64)."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_pinhole(kp3d, K):
    """Project (..., 3) camera-frame points through a pinhole K. Mirrors reprojection_qc()."""
    kp3d = np.asarray(kp3d, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.clip(kp3d[..., 2], 1e-6, None)
    return np.stack([fx * kp3d[..., 0] / z + cx, fy * kp3d[..., 1] / z + cy], axis=-1)


# ------------------------------------------------------------------ drawing


def _finite_point(xy, width, height, margin=64):
    """Return an int pixel tuple, or None when the point is unusable."""
    if not np.all(np.isfinite(xy)):
        return None
    x, y = float(xy[0]), float(xy[1])
    if x < -margin or y < -margin or x > width + margin or y > height + margin:
        return None
    return int(round(x)), int(round(y))


def draw_hand(frame_bgr, j2d, hand_name='right', line_thickness=2, joint_radius=3, draw_palm=True):
    """Draw one 21-joint hand skeleton onto a BGR frame, in place, per-finger coloured."""
    import cv2

    height, width = frame_bgr.shape[:2]
    points = [_finite_point(j2d[i], width, height) for i in range(NUM_HAND_JOINTS)]
    is_left = hand_name == 'left'

    if draw_palm:
        for a, b in PALM_EDGES:
            if points[a] is not None and points[b] is not None:
                cv2.line(frame_bgr, points[a], points[b], PALM_COLOR,
                         max(1, line_thickness - 1), cv2.LINE_AA)

    for _, chain, color in FINGER_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if points[a] is not None and points[b] is not None:
                cv2.line(frame_bgr, points[a], points[b], color, line_thickness, cv2.LINE_AA)

    for _, chain, color in FINGER_CHAINS:
        for idx in chain[1:]:
            pt = points[idx]
            if pt is None:
                continue
            cv2.circle(frame_bgr, pt, joint_radius, color, -1, cv2.LINE_AA)
            if not is_left:
                # Outer ring on the right hand so the two stay distinguishable where they overlap.
                cv2.circle(frame_bgr, pt, joint_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    wrist = points[WRIST]
    if wrist is not None:
        label_color = HAND_LABEL_COLORS[hand_name]
        cv2.circle(frame_bgr, wrist, joint_radius + 3, label_color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, wrist, joint_radius + 5, (20, 20, 20), 1, cv2.LINE_AA)
        tag = 'L' if is_left else 'R'
        org = (wrist[0] + 10, wrist[1] - 10)
        cv2.putText(frame_bgr, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA)


def draw_points(frame_bgr, j2d, color, radius=2):
    """Scatter one keypoint set (e.g. the raw RTMPose observation) for visual comparison."""
    import cv2

    height, width = frame_bgr.shape[:2]
    for i in range(min(NUM_HAND_JOINTS, len(j2d))):
        pt = _finite_point(j2d[i], width, height)
        if pt is not None:
            cv2.circle(frame_bgr, pt, radius, color, -1, cv2.LINE_AA)


def draw_forearm(frame_bgr, arm_j2d, line_thickness=2):
    """Draw the 3-joint forearm chain onto a BGR frame, in place."""
    import cv2

    height, width = frame_bgr.shape[:2]
    n = min(NUM_ARM_JOINTS, len(arm_j2d))
    points = [_finite_point(arm_j2d[i], width, height) for i in range(n)]
    color = (150, 150, 150)
    for a in range(n - 1):
        if points[a] is not None and points[a + 1] is not None:
            cv2.line(frame_bgr, points[a], points[a + 1], color, line_thickness, cv2.LINE_AA)
    for pt in points:
        if pt is not None:
            cv2.circle(frame_bgr, pt, line_thickness + 1, color, -1, cv2.LINE_AA)
