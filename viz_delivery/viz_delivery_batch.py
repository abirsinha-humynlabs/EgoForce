#!/usr/bin/env python
"""
viz_delivery_batch.py — fleet render driver for the 617 VIZ_Final clips (robust work-stealing).

Per clip, picks the renderer by the manifest's `with_filter`:
  with_filter=False ("original no filter")   -> render_v4_panels.py            (wilor_only fix, NO shape/foot drops)
  with_filter=True  ("drop feet + bad-shape")-> render_v4_panels_with_filter.py (bad_shape+foot drops ALWAYS on)
Both run --clip-foot --future-sec 3 --past-sec 0 (matches the validated working sample).

Design: N INDEPENDENT worker processes. Each loops: claim an unrendered+unlocked clip via an atomic
S3 lock (put-object if-none-match "*"), render it, upload, repeat — until nothing is claimable. No
shared pool, so one worker dying (crash/OOM) can never strand the others' work or cancel pending clips.
Stale locks (a dead worker's clip) are reclaimed after 12 min. Boxes coordinate across the fleet through
the same S3 lock space and self-terminate when the manifest drains. Output ->
labelling_results/viz_delivery/<clip>_wrist_traj_panels.mp4
"""
import argparse, json, os, subprocess, sys, tempfile, shutil, time, random
import multiprocessing as mp
import boto3
from botocore.exceptions import ClientError

B = "stage-humyn-egocentric-stereo-data"
REGION = "ap-south-1"
HERE = os.path.dirname(os.path.abspath(__file__))
PYBIN = sys.executable
VIZ = "labelling_results/viz_delivery"
LOCKS = f"{VIZ}/_locks"
STALE_SEC = 12 * 60
_C = None


def cli():
    global _C
    if _C is None:
        _C = boto3.client("s3", region_name=REGION)
    return _C


def have(key):
    try:
        cli().head_object(Bucket=B, Key=key); return True
    except ClientError:
        return False


def acquire_lock(clip):
    lk = f"{LOCKS}/{clip}.lock"
    try:
        cli().put_object(Bucket=B, Key=lk, Body=b"", IfNoneMatch="*")
        return True
    except ClientError:
        try:
            h = cli().head_object(Bucket=B, Key=lk)
            if time.time() - h["LastModified"].timestamp() > STALE_SEC:
                cli().put_object(Bucket=B, Key=lk, Body=b"")     # reclaim dead worker's clip
                return True
        except ClientError:
            pass
        return False


def out_key(clip):
    return f"{VIZ}/{clip}_wrist_traj_panels.mp4"


def render(rec):
    clip = rec["clip"]; c = cli(); w = tempfile.mkdtemp(prefix=f"viz_{clip[:20]}_")
    try:
        vp = os.path.join(w, "left_eye.mp4"); npz = os.path.join(w, "k.npz")
        hp = os.path.join(w, "h.npz"); imu = os.path.join(w, "imu.csv")
        ind = rec["input_dir"]
        c.download_file(B, ind + "left_eye.mp4", vp)
        c.download_file(B, rec["npz_key"], npz)
        c.download_file(B, rec["head_key"], hp)
        has_imu = have(ind + "imu_accel.csv")
        if has_imu:
            c.download_file(B, ind + "imu_accel.csv", imu)
        renderer = "render_v4_panels_with_filter.py" if rec["with_filter"] else "render_v4_panels.py"
        odir = os.path.join(w, "out")
        cmd = [PYBIN, os.path.join(HERE, renderer), "--video", vp, "--npz", npz, "--head", hp,
               "--out", odir, "--future-sec", "3", "--past-sec", "0", "--clip-foot"]
        if has_imu:
            cmd += ["--imu", imu]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MPLBACKEND="Agg")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        made = os.path.join(odir, "left_eye_wrist_traj_panels.mp4")
        if r.returncode != 0 or not os.path.exists(made):
            return f"render_err:{(r.stderr or '')[-200:]}"
        web = os.path.join(w, "web.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", made, "-vf",
                        "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
                        "-c:v", "libx264", "-profile:v", "high", "-crf", "18", "-preset", "veryfast",
                        "-movflags", "+faststart", web], check=True)
        c.upload_file(web, B, out_key(clip), ExtraArgs={"ContentType": "video/mp4"})
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"exc:{type(e).__name__}:{str(e)[-160:]}"
    finally:
        shutil.rmtree(w, ignore_errors=True)


def claim_next(recs, seen):
    n = len(recs); start = random.randrange(n)
    for k in range(n):
        r = recs[(start + k) % n]; clip = r["clip"]
        if clip in seen:
            continue
        if have(out_key(clip)):
            seen.add(clip); continue
        if acquire_lock(clip):
            return r
    return None


def worker_loop(wid, recs):
    seen = set(); misses = 0; nrender = 0
    while True:
        r = claim_next(recs, seen)
        if r is None:
            misses += 1
            if misses >= 3:
                print(f"[w{wid}] drained after {nrender} renders", flush=True); return
            time.sleep(15 + random.random() * 15)     # let peers finish / stale reclaim
            continue
        misses = 0
        st = render(r)
        nrender += 1
        seen.add(r["clip"])                            # don't re-scan this clip; leave lock on failure
        if st != "ok":
            print(f"[w{wid}] {r['clip']}: {st}", flush=True)


def remaining(recs):
    return sum(1 for r in recs if not have(out_key(r["clip"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "final_manifest.json"))
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    recs = json.load(open(a.manifest))
    host = os.uname().nodename
    print(f"viz_delivery: {len(recs)} clips, procs={a.procs}, host={host}, remaining={remaining(recs)}", flush=True)
    ctx = mp.get_context("spawn")
    for outer in range(6):                              # respawn passes to mop up crash-orphaned clips
        rem = remaining(recs)
        if rem == 0:
            print("ALL DONE", flush=True); break
        print(f"[pass {outer}] remaining={rem}; launching {a.procs} workers", flush=True)
        procs = [ctx.Process(target=worker_loop, args=(i, recs)) for i in range(a.procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        after = remaining(recs)
        print(f"[pass {outer}] done; remaining now {after}", flush=True)
        if after == 0:
            break
        if after >= rem:                                # no progress -> peers own the rest; wait out stale
            time.sleep(60)
    try:
        cli().put_object(Bucket=B, Key=f"{VIZ}/_reports/{host}_{int(time.time())}.json",
                         Body=json.dumps({"host": host, "remaining": remaining(recs)}).encode())
    except Exception:
        pass
    print("worker host exiting", flush=True)


if __name__ == "__main__":
    main()
