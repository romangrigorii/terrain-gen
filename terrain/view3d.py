"""Load a rendered terrain surface into a simple perspective view."""
import math

import numpy as np


def _fit_color(color, shape):
    color = np.asarray(color)
    if color.ndim == 2:
        color = np.repeat(color[:, :, None], 3, axis=2)
    color = color[..., :3]
    if color.shape[:2] == shape:
        return color.astype(np.float32, copy=False)
    ys = np.linspace(0, color.shape[0] - 1, shape[0]).astype(np.int32)
    xs = np.linspace(0, color.shape[1] - 1, shape[1]).astype(np.int32)
    return color[ys][:, xs].astype(np.float32, copy=False)


def _sample(field, z01, x01):
    """Bilinear sample. `z01`/`x01` are 0..1 across rows and columns."""
    rows, cols = field.shape[:2]
    rf = np.clip(z01, 0, 1) * (rows - 1)
    cf = np.clip(x01, 0, 1) * (cols - 1)
    r0 = np.floor(rf).astype(np.int32)
    c0 = np.floor(cf).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    r0 = np.clip(r0, 0, rows - 1)
    c0 = np.clip(c0, 0, cols - 1)
    tr = (rf - r0).astype(np.float32)[..., None] if field.ndim == 3 else (rf - r0).astype(np.float32)
    tc = (cf - c0).astype(np.float32)[..., None] if field.ndim == 3 else (cf - c0).astype(np.float32)
    v00 = field[r0, c0]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]
    return (v00 * (1 - tc) + v01 * tc) * (1 - tr) + (v10 * (1 - tc) + v11 * tc) * tr


def _mesh_shape(rows, cols, steps):
    """Vertex counts for one sheet. `steps` only chooses how fine that sheet is."""
    long_edge = max(24, min(max(rows, cols), int(steps) * 5))
    if cols >= rows:
        vc = min(cols, long_edge)
        vr = max(2, min(rows, int(round(long_edge * rows / max(cols, 1)))))
    else:
        vr = min(rows, long_edge)
        vc = max(2, min(cols, int(round(long_edge * cols / max(rows, 1)))))
    return vr, vc


def _fill_tris(px, py, pz, col, vx, vz, valid, rgb, zbuf, pick_y, pick_x):
    """Paint both triangles of every grid quad. Nearer depth wins, so one sheet shows."""
    rows, cols = px.shape
    view_h, view_w = zbuf.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (valid[r, c] < 0.5 or valid[r, c + 1] < 0.5
                    or valid[r + 1, c] < 0.5 or valid[r + 1, c + 1] < 0.5):
                continue
            for k in range(2):
                if k == 0:
                    r0 = r
                    c0 = c
                    r1 = r
                    c1 = c + 1
                    r2 = r + 1
                    c2 = c
                else:
                    r0 = r + 1
                    c0 = c
                    r1 = r
                    c1 = c + 1
                    r2 = r + 1
                    c2 = c + 1
                z0 = pz[r0, c0]
                z1 = pz[r1, c1]
                z2 = pz[r2, c2]
                if z0 < 0.05 or z1 < 0.05 or z2 < 0.05:
                    continue
                x0 = px[r0, c0]
                y0 = py[r0, c0]
                x1 = px[r1, c1]
                y1 = py[r1, c1]
                x2 = px[r2, c2]
                y2 = py[r2, c2]
                area = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)
                if area < 1.0e-3 and area > -1.0e-3:
                    continue
                minx = x0
                maxx = x0
                miny = y0
                maxy = y0
                if x1 < minx:
                    minx = x1
                if x2 < minx:
                    minx = x2
                if x1 > maxx:
                    maxx = x1
                if x2 > maxx:
                    maxx = x2
                if y1 < miny:
                    miny = y1
                if y2 < miny:
                    miny = y2
                if y1 > maxy:
                    maxy = y1
                if y2 > maxy:
                    maxy = y2
                ix0 = int(math.floor(minx))
                iy0 = int(math.floor(miny))
                ix1 = int(math.ceil(maxx)) - 1
                iy1 = int(math.ceil(maxy)) - 1
                if ix0 < 0:
                    ix0 = 0
                if iy0 < 0:
                    iy0 = 0
                if ix1 >= view_w:
                    ix1 = view_w - 1
                if iy1 >= view_h:
                    iy1 = view_h - 1
                if ix0 > ix1 or iy0 > iy1:
                    continue
                inv = 1.0 / area
                for yy in range(iy0, iy1 + 1):
                    pyy = yy + 0.5
                    for xx in range(ix0, ix1 + 1):
                        pxx = xx + 0.5
                        w0 = (pxx - x1) * (y2 - y1) - (pyy - y1) * (x2 - x1)
                        w1 = (pxx - x2) * (y0 - y2) - (pyy - y2) * (x0 - x2)
                        w2 = (pxx - x0) * (y1 - y0) - (pyy - y0) * (x1 - x0)
                        if area > 0.0:
                            if w0 < -1.0e-3 or w1 < -1.0e-3 or w2 < -1.0e-3:
                                continue
                        elif w0 > 1.0e-3 or w1 > 1.0e-3 or w2 > 1.0e-3:
                            continue
                        depth = (w0 * z0 + w1 * z1 + w2 * z2) * inv
                        if depth >= zbuf[yy, xx]:
                            continue
                        b0 = w0 * inv
                        b1 = w1 * inv
                        b2 = w2 * inv
                        zbuf[yy, xx] = depth
                        rgb[yy, xx, 0] = b0 * col[r0, c0, 0] + b1 * col[r1, c1, 0] + b2 * col[r2, c2, 0]
                        rgb[yy, xx, 1] = b0 * col[r0, c0, 1] + b1 * col[r1, c1, 1] + b2 * col[r2, c2, 1]
                        rgb[yy, xx, 2] = b0 * col[r0, c0, 2] + b1 * col[r1, c1, 2] + b2 * col[r2, c2, 2]
                        pick_y[yy, xx] = b0 * vz[r0, c0] + b1 * vz[r1, c1] + b2 * vz[r2, c2]
                        pick_x[yy, xx] = b0 * vx[r0, c0] + b1 * vx[r1, c1] + b2 * vx[r2, c2]


def _bind_fill():
    try:
        from numba import njit
        return njit(cache=True)(_fill_tris)
    except Exception:
        return _fill_tris

_fill_mesh = _bind_fill()


def pan_delta(yaw, pitch, distance, dx, dy, view_w, view_h):
    """World shift so the surface follows a click-drag."""
    pitch = float(np.clip(pitch, 0.15, 1.25))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    forward = np.array([-cp * sy, -sp, -cp * cy], np.float64)
    forward /= max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_len
    up = np.cross(right, forward)
    aspect = float(view_w) / max(float(view_h), 1.0)
    scale = float(distance) * 0.46 * aspect / max(float(view_w), 1.0) * 2.0
    shift_x = (-float(dx) * right[0] + float(dy) * up[0]) * scale
    shift_z = (-float(dx) * right[2] + float(dy) * up[2]) * scale
    return float(shift_x), float(shift_z)


def stitch_tiles(tiles):
    """Join `(tx, ty, height, color)` tiles into one sheet."""
    usable = []
    for tx, ty, height, color in tiles:
        height = np.asarray(height, np.float32)
        if height.ndim != 2 or height.shape[0] < 2 or height.shape[1] < 2:
            continue
        usable.append((int(tx), int(ty), height, color))
    if not usable:
        return None
    usable.sort(key=lambda item: (item[1], item[0]))
    rows, cols = usable[0][2].shape
    fitted = []
    for tx, ty, height, color in usable:
        if height.shape != (rows, cols):
            height = _sample(height,
                             np.linspace(0, 1, rows, dtype=np.float32)[:, None],
                             np.linspace(0, 1, cols, dtype=np.float32)[None, :]).astype(np.float32)
        color = _fit_color(color, (rows, cols))
        fitted.append((tx, ty, height, color))
    txs = [tx for tx, _, _, _ in fitted]
    tys = [ty for _, ty, _, _ in fitted]
    tx0, ty0 = min(txs), min(tys)
    nx = max(txs) - tx0 + 1
    ny = max(tys) - ty0 + 1
    out_h = ny * (rows - 1) + 1
    out_w = nx * (cols - 1) + 1
    sheet = np.zeros((out_h, out_w), np.float32)
    rgb = np.zeros((out_h, out_w, 3), np.float32)
    mask = np.zeros((out_h, out_w), np.float32)
    for tx, ty, height, color in fitted:
        y0 = (ty - ty0) * (rows - 1)
        x0 = (tx - tx0) * (cols - 1)
        sheet[y0:y0 + rows, x0:x0 + cols] = height
        rgb[y0:y0 + rows, x0:x0 + cols] = color
        mask[y0:y0 + rows, x0:x0 + cols] = 1
    return sheet, rgb, mask, (tx0, ty0, nx, ny)


def load_surface(height, color, yaw, pitch, distance, view_w, view_h,
                 steps=28, y_scale=0.35, origin=None, mask=None):
    """Drape `color` on one mesh of `height` and return `(rgb, pick_y, pick_x)`."""
    height = np.asarray(height, np.float32)
    view_w = int(max(8, view_w))
    view_h = int(max(8, view_h))
    rgb = np.empty((view_h, view_w, 3), np.uint8)
    rgb[:] = (10, 12, 16)
    pick_y = np.full((view_h, view_w), -1, np.float32)
    pick_x = np.full((view_h, view_w), -1, np.float32)
    if height.ndim != 2 or height.shape[0] < 2 or height.shape[1] < 2:
        return rgb, pick_y, pick_x
    color = _fit_color(color, height.shape)
    if mask is not None:
        mask = np.asarray(mask, np.float32)
        if mask.shape != height.shape:
            mask = None
    if mask is not None and float(mask.min()) < 0.5:
        holes = mask < 0.5
        if holes.any() and not holes.all():
            from scipy.ndimage import distance_transform_edt
            _dist, (iy, ix) = distance_transform_edt(holes, return_indices=True)
            height = np.array(height, np.float32, copy=True)
            color = np.array(color, np.float32, copy=True)
            height[holes] = height[iy[holes], ix[holes]]
            color[holes] = color[iy[holes], ix[holes]]
    light = np.array([0.35, 0.86, 0.28], np.float32)
    light /= np.linalg.norm(light)
    scale = float(y_scale)
    rows, cols = height.shape
    vr, vc = _mesh_shape(rows, cols, max(8, int(steps)))
    if vr == rows and vc == cols:
        z01 = np.linspace(0, 1, rows, dtype=np.float32)
        x01 = np.linspace(0, 1, cols, dtype=np.float32)
        hs = height
        sampled = color
        if mask is None:
            vmask = np.ones((vr, vc), np.float32)
        else:
            vmask = np.ascontiguousarray(mask, np.float32)
    else:
        z01 = np.linspace(0, 1, vr, dtype=np.float32)
        x01 = np.linspace(0, 1, vc, dtype=np.float32)
        zz, xx = np.meshgrid(z01, x01, indexing="ij")
        hs = np.asarray(_sample(height, zz, xx), np.float32)
        sampled = np.asarray(_sample(color, zz, xx), np.float32)
        if mask is None:
            vmask = np.ones((vr, vc), np.float32)
        else:
            ri = np.clip(np.rint(z01 * (rows - 1)).astype(np.int32), 0, rows - 1)
            ci = np.clip(np.rint(x01 * (cols - 1)).astype(np.int32), 0, cols - 1)
            vmask = np.ascontiguousarray(mask[np.ix_(ri, ci)], np.float32)
    gy, gx = np.gradient(np.asarray(hs, np.float32))
    nx = -gx * np.float32(scale * max(vc - 1, 1))
    nz = -gy * np.float32(scale * max(vr - 1, 1))
    ny = np.ones_like(nx, np.float32)
    length = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), np.float32(1e-8))
    lambert = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / length, 0, 1)
    lit = np.ascontiguousarray(
        np.clip(sampled * (np.float32(0.62) + np.float32(0.5) * lambert[..., None]), 0, 255),
        np.float32)

    pitch = float(np.clip(pitch, 0.15, 1.25))
    distance = float(max(0.35, distance))
    yaw = float(yaw)
    if origin is None:
        origin_x, origin_z = 0.5, 0.5
    else:
        origin_x, origin_z = float(origin[0]), float(origin[1])
    if 0.0 <= origin_x <= 1.0 and 0.0 <= origin_z <= 1.0:
        mid = float(_sample(height, np.array(origin_z), np.array(origin_x))) * scale
    else:
        mid = float(np.median(height)) * scale
    target = np.array([origin_x, mid, origin_z], np.float32)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    eye = target + np.array([distance * cp * sy, distance * sp, distance * cp * cy], np.float32)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0], np.float32))
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-5:
        right = np.array([1.0, 0.0, 0.0], np.float32)
    else:
        right = right / right_len
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    fov = np.float32(0.46)
    aspect = np.float32(view_w / view_h)
    zz, xx = np.meshgrid(z01, x01, indexing="ij")
    world_y = np.asarray(hs, np.float32) * np.float32(scale)
    relx = xx - eye[0]
    rely = world_y - eye[1]
    relz = zz - eye[2]
    cx = relx * right[0] + rely * right[1] + relz * right[2]
    cy = relx * up[0] + rely * up[1] + relz * up[2]
    cz = relx * forward[0] + rely * forward[1] + relz * forward[2]
    inv_z = np.float32(1.0) / np.maximum(cz, np.float32(1e-3))
    ndc_x = (cx * inv_z) / (fov * aspect)
    ndc_y = (cy * inv_z) / fov
    screen_x = np.ascontiguousarray((ndc_x + np.float32(1)) * np.float32(view_w * 0.5), np.float32)
    screen_y = np.ascontiguousarray((np.float32(1) - ndc_y) * np.float32(view_h * 0.5), np.float32)
    screen_z = np.ascontiguousarray(cz, np.float32)
    vx = np.empty((vr, vc), np.float32)
    vz = np.empty((vr, vc), np.float32)
    vx[:] = x01
    vz[:] = z01[:, None]
    shaded = np.zeros((view_h, view_w, 3), np.float32)
    zbuf = np.full((view_h, view_w), np.float32(1e9))
    global _fill_mesh
    try:
        _fill_mesh(screen_x, screen_y, screen_z, lit, vx, vz, vmask, shaded, zbuf, pick_y, pick_x)
    except Exception:
        if _fill_mesh is _fill_tris:
            raise
        _fill_mesh = _fill_tris
        _fill_mesh(screen_x, screen_y, screen_z, lit, vx, vz, vmask, shaded, zbuf, pick_y, pick_x)
    hit = zbuf < np.float32(1e8)
    rgb[hit] = np.clip(shaded[hit], 0, 255).astype(np.uint8)
    pick_y = np.where(hit, pick_y, np.float32(-1))
    pick_x = np.where(hit, pick_x, np.float32(-1))
    return rgb, pick_y, pick_x


def mark_strokes(rgb, pick_y, pick_x, strokes, cursor=None, y_aspect=1.0):
    """Tint ground pixels that fall inside a brush. Sky picks stay put."""
    strokes = list(strokes or ())
    if not strokes and cursor is None:
        return rgb
    out = np.array(rgb, copy=True)
    valid = pick_y >= 0
    aspect = float(y_aspect)

    def splat(points, radius, color, alpha):
        radius = float(radius)
        if radius <= 0 or not points:
            return
        near = np.zeros(pick_y.shape, np.bool_)
        limit = radius * radius
        for y, x in points:
            dy = (pick_y - float(y)) * aspect
            dx = pick_x - float(x)
            near |= valid & (dx * dx + dy * dy <= limit)
        if not near.any():
            return
        base = out[near].astype(np.float32)
        tint = np.asarray(color, np.float32)
        out[near] = np.clip(base * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)

    for stroke in strokes:
        kind = stroke.get("kind")
        color = {
            "mountain": (210, 120, 70),
            "glacial": (190, 214, 228),
            "dune": (210, 176, 110),
            "coast": (90, 170, 160),
            "cave": (140, 110, 170),
        }.get(kind, (70, 150, 210))
        splat(stroke.get("points") or (), stroke.get("radius", 0.0), color, 0.55)
    if cursor is not None:
        y, x, radius, kind = cursor
        color = {
            "mountain": (240, 176, 120),
            "glacial": (220, 234, 242),
            "dune": (230, 200, 140),
            "coast": (140, 210, 200),
            "cave": (180, 150, 210),
        }.get(kind, (150, 206, 240))
        splat(((y, x),), radius, color, 0.4)
    return out
