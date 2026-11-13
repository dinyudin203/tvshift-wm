"""Project +x/+y/+z EEF displacements (1/2/3/5 cm, robot base frame) into the agentview
observation for each pair5 camera view, draw arrows, and dump pixel deltas.

Pixel numbers are reported both at native 512x512 and at the policy input size
(Qwen2.5-VL smart_resize with max_pixels=128*28*28 -> 308x308)."""
import json, sys
from pathlib import Path
import numpy as np, av, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

DATA = Path('/dataset/yujin/data/tvshift-libero-pair5-lerobot')
CAL = Path('/dataset/yujin/vla/icwm_plucker_tv6/calibration_pair5')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FLIP = sys.argv[2] if len(sys.argv) > 2 else 'none'   # none | v | hv
FRAME_IN_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SETS = [('Task1', 'libero_spatial_theta000_no_noops_lerobot', 0),
        ('Task2', 'libero_goal_theta015_no_noops_lerobot', 15),
        ('Task3', 'libero_object_theta030_no_noops_lerobot', 30),
        ('Task4', 'libero_10_theta330_no_noops_lerobot', 330),
        ('Task5', 'libero_10_theta345_no_noops_lerobot', 345)]
DISTS = [float(x) / 100 for x in (sys.argv[4] if len(sys.argv) > 4 else '1,2,3,5').split(',')]
CMS = [int(round(x * 100)) for x in DISTS]
AXES = {'+x': (np.array([1, 0, 0.]), (230, 40, 40)),
        '+y': (np.array([0, 1, 0.]), (40, 200, 40)),
        '+z': (np.array([0, 0, 1.]), (50, 110, 255))}
W_IN = 308; NATIVE = 512; S = W_IN / NATIVE


def read_frame(root, ep_row, k):
    key = 'observation.images.image'
    fi = int(ep_row[f'videos/{key}/file_index']); ci = int(ep_row[f'videos/{key}/chunk_index'])
    t0 = float(ep_row[f'videos/{key}/from_timestamp'])
    path = root / f'videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4'
    target = t0 + k / 20.0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(max(target - 1, 0) / st.time_base), stream=st)
        for fr in c.decode(st):
            if fr.time >= target - 1e-3:
                return fr.to_ndarray(format='rgb24')
    raise RuntimeError('frame not found')


def project(p_w, c2w, K):
    w2c = np.linalg.inv(c2w)
    pc = w2c[:3, :3] @ p_w + w2c[:3, 3]          # OpenGL cam: x right, y up, looks -z
    u = K[0, 2] + K[0, 0] * pc[0] / -pc[2]
    v = K[1, 2] - K[1, 1] * pc[1] / -pc[2]
    if FLIP in ('v', 'hv'): v = NATIVE - v
    if FLIP == 'hv': u = NATIVE - u
    return np.array([u, v])


def arrow(d, a, b, col, w=3):
    d.line([tuple(a), tuple(b)], fill=col, width=w)
    v = b - a; n = np.linalg.norm(v)
    if n < 1e-6: return
    v /= n; p = np.array([-v[1], v[0]]); L = 9
    d.polygon([tuple(b), tuple(b - L * v + 0.5 * L * p), tuple(b - L * v - 0.5 * L * p)], fill=col)


results = {}
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    fsmall = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except OSError:
    font = fsmall = ImageFont.load_default()
panels = []
for name, ds, theta in SETS:
    root = DATA / ds
    eps = pq.read_table(sorted((root / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()
    ep = eps[0]
    gidx = int(ep['dataset_from_index']) + FRAME_IN_EP
    data = pq.read_table(sorted((root / 'data').glob('*/*.parquet'))[0], columns=['index', 'observation.state', 'task_index'])
    idx = np.asarray(data['index'].to_pylist()); row = int(np.flatnonzero(idx == gidx)[0])
    eef = np.asarray(data['observation.state'][row].as_py()[:3], dtype=np.float64)
    c2w = np.load(CAL / ds / 'camera_to_world.npy', mmap_mode='r')[gidx, 0].astype(np.float64)
    K = np.load(CAL / ds / 'intrinsics.npy', mmap_mode='r')[gidx, 0].astype(np.float64)
    img = read_frame(root, ep, FRAME_IN_EP)
    # base frame rotation == world rotation in LIBERO (robot_base_to_world is a pure translation)
    base = json.loads((CAL / ds / 'robot_base.json').read_text())['tasks'][0]['robot_base_to_world']
    R_b2w = np.asarray(base)[:3, :3]
    o = project(eef, c2w, K)
    rec = {'theta': theta, 'frame_index': gidx, 'eef_world': eef.round(4).tolist(),
           'eef_px_native': o.round(1).tolist(), 'eef_px_input308': (o * S).round(1).tolist(), 'axes': {}}
    # draw on model-input-resolution image, upscaled x2 for readability
    UP = 2
    small = Image.fromarray(img).resize((W_IN, W_IN), Image.BICUBIC).resize((W_IN * UP, W_IN * UP), Image.NEAREST)
    d = ImageDraw.Draw(small)
    for ax, (vec, col) in AXES.items():
        pts = []; rec['axes'][ax] = {}
        for dist in DISTS:
            p = project(eef + R_b2w @ (vec * dist), c2w, K)
            dn = p - o; di = dn * S
            rec['axes'][ax][f'{int(dist*100)}cm'] = {
                'du_dv_native': dn.round(2).tolist(), 'len_native': round(float(np.linalg.norm(dn)), 2),
                'du_dv_input308': di.round(2).tolist(), 'len_input308': round(float(np.linalg.norm(di)), 2)}
            pts.append(p * S * UP)
        oo = o * S * UP
        arrow(d, oo, pts[-1], col)
        for dist, p in zip(DISTS, pts):
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 255, 255), outline=col)
        L5 = rec['axes'][ax][f'{CMS[-1]}cm']['len_input308']
        d.text(tuple(pts[-1] + 6), f"{ax} {L5:.1f}px/{CMS[-1]}cm", fill=col, font=fsmall, stroke_width=2, stroke_fill=(0, 0, 0))
    d.ellipse([oo[0] - 4, oo[1] - 4, oo[0] + 4, oo[1] + 4], fill=(255, 255, 0))
    hdr = f"{name}  {ds.split('_no_noops')[0]}  ({theta} deg)"
    d.rectangle([0, 0, W_IN * UP, 40], fill=(0, 0, 0))
    d.text((6, 4), hdr, fill=(255, 255, 255), font=font)
    lens = '  '.join(f"{ax}: " + '/'.join(f"{rec['axes'][ax][f'{c}cm']['len_input308']:.1f}" for c in CMS) for ax in AXES)
    d.text((6, 22), f"px@308 for {'/'.join(map(str, CMS))}cm  {lens}", fill=(220, 220, 220), font=fsmall)
    small.save(OUT / f'{name}_theta{theta:03d}.png'); panels.append(small)
    results[name] = rec
    print(name, theta, 'eef_px308', rec['eef_px_input308'], lens, flush=True)

grid = Image.new('RGB', (W_IN * 2 * 3, W_IN * 2 * 2), (30, 30, 30))
for i, p in enumerate(panels):
    grid.paste(p, ((i % 3) * W_IN * 2, (i // 3) * W_IN * 2))
grid.save(OUT / 'grid_all_views.png')
(OUT / 'pixel_deltas.json').write_text(json.dumps(results, indent=2))
