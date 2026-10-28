"""The shared core of hint labels: one distance, one merge, one hint builder.

Used in three places, so "distinct saddle" means the same thing everywhere: building the training labels
(merging each minimum's verified saddles into regions), and at inference before Sella (grouping landing points)
and after it (merging the refined saddles before they are counted).

    saddle_distance(m, P, q)   largest per-atom minimum-image distance from each of P (k, N, 3) to q (N, 3),
                               with the overall shift of the cell's contents removed (no fixed atoms)
    merge(m, points, energies, priority, radius, de)
                               greedy merge against representatives only: points are taken in priority order
                               (most often reached first, then lowest energy); each joins the first
                               representative within `radius` A and `de` eV, else becomes a representative.
                               Comparing only with representatives means a chain A~B~C cannot pull together
                               two points that are far apart.
    make_hints(m, saddle, thr, min_move)
                               one hint (atom, unit direction, move) for every atom that moves at least `thr`
                               times the largest move and at least `min_move` A from the minimum to the saddle;
                               the largest mover always gives one
"""
import numpy as np

MERGE_RADIUS, MERGE_DE = 0.15, 0.05
HINT_THR, HINT_MIN_MOVE = 0.5, 0.2


def mic(cell, d):
    """Minimum-image displacement(s) d (..., 3) for the cell (3, 3)."""
    A = np.asarray(cell, float)
    f = np.asarray(d, float) @ np.linalg.inv(A)
    return (f - np.round(f)) @ A


def displacement(m, pos):
    """(N, 3) minimum-image displacement from the minimum to pos, overall shift removed."""
    d = mic(m.cell.array, np.asarray(pos) - m.positions)
    return d - d.mean(0)


def saddle_distance(m, P, q):
    d = mic(m.cell.array, np.asarray(P) - np.asarray(q)[None])
    d -= d.mean(-2, keepdims=True)
    return np.linalg.norm(d, axis=-1).max(-1)


def merge(m, points, energies, priority, radius=MERGE_RADIUS, de=MERGE_DE):
    """(group index per point, indices of the representatives). `priority`: larger first; ties by energy."""
    order = sorted(range(len(points)), key=lambda i: (-priority[i], energies[i]))
    reps, group = [], np.full(len(points), -1)
    for i in order:
        if reps:
            d = saddle_distance(m, np.array([points[r] for r in reps]), np.array(points[i]))
            ok = [k for k in np.argsort(d) if d[k] <= radius and abs(energies[i] - energies[reps[k]]) <= de]
            if ok:
                group[i] = ok[0]
                continue
        group[i] = len(reps)
        reps.append(i)
    return group, reps


def make_hints(m, saddle_pos, thr=HINT_THR, min_move=HINT_MIN_MOVE):
    """[(atom, unit direction (3,), move in A)] for the saddle at saddle_pos, largest move first."""
    d = displacement(m, saddle_pos)
    size = np.linalg.norm(d, axis=1)
    top = int(size.argmax())
    keep = np.flatnonzero((size >= thr * size[top]) & (size >= min_move)).tolist() or [top]
    keep = sorted(keep, key=lambda a: -size[a])
    return [(int(a), d[a] / size[a], float(size[a])) for a in keep]


def tilt(u, theta_deg, rng):
    """A unit vector drawn uniformly from the cone of half-angle theta around the unit vector u."""
    cos_t = rng.uniform(np.cos(np.radians(theta_deg)), 1.0)
    phi = rng.uniform(0, 2 * np.pi)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    sin_t = np.sqrt(1 - cos_t ** 2)
    return cos_t * u + sin_t * (np.cos(phi) * e1 + np.sin(phi) * e2)
