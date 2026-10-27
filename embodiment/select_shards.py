"""Keep the first N successful demos (by demo index) of every task in a shard dir: symlink their files into <out>, which
pack_lerobot.py then packs. The full shard dir is left untouched."""
import argparse, glob, json, os
p = argparse.ArgumentParser(); p.add_argument('--shards', required=True); p.add_argument('--out', required=True); p.add_argument('--n', type=int, default=45); p.add_argument('--theta', type=int, default=0)
a = p.parse_args()
by_task = {}
for jf in glob.glob(f'{a.shards}/_done/*.json'):
    m = json.load(open(jf)); by_task.setdefault(m['task'], []).append(m)
for sub in ('_done', 'frames', f'theta{a.theta:03d}', 'wrist'): os.makedirs(f'{a.out}/{sub}', exist_ok=True)
for t, ms in sorted(by_task.items()):
    ms.sort(key=lambda m: m['demo']); keep = ms[:a.n]
    assert len(keep) == a.n, f'task {t}: only {len(keep)} successful demos (< {a.n})'
    for m in keep:
        st = m['stem']
        for sub, ext in (('_done', 'json'), ('frames', 'npz'), (f'theta{a.theta:03d}', 'mp4'), ('wrist', 'mp4')):
            dst = f'{a.out}/{sub}/{st}.{ext}'
            if not os.path.lexists(dst): os.symlink(os.path.abspath(f'{a.shards}/{sub}/{st}.{ext}'), dst)
    print(f'task {t}: {len(ms)} successful, kept demos {[m["demo"] for m in keep][:5]}... ({len(keep)})')
