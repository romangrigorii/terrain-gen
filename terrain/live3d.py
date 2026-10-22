"""Live land and sea for the studio's 3D view.

The flowing-terrain viewer draws a lit land mesh and a separate sea that moves.
This does the same with the studio's own height and color: land keeps the map
colors, and the sea is a second mesh at sea level (height 0) with a moving surface.
The software rasterizer in view3d stays as the fallback when OpenGL is missing.
"""
import math

import numpy as np

from terrain.export import METRES_PER_UNIT
from terrain.view3d import _fit_color, _sample

# Fallback only. A real sheet uses metres, so 12 km of relief is 12 km on the ground.
Y_SCALE = 0.35
FOV = 0.46
SKY = (0.65, 0.77, 0.90)
LIGHT = (0.35, 0.86, 0.28)
# The drape is the preview image. Keep that whole grid unless a many-tile sheet
# would make the mesh too heavy. Cutting a 640-wide preview down to 512 was what
# made coasts and shading look coarser than the map.
MAX_EDGE = 2048

_BG_VERT = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_BG_FRAG = """
#version 330
uniform vec3 sky;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_pick;
void main() {
    out_color = vec4(sky, 1.0);
    out_pick = vec2(-1.0, -1.0);
}
"""

_LAND_VERT = """
#version 330
in vec2 xz;
in float h;
in vec2 uv;
in vec3 nrm;
uniform mat4 mvp;
uniform float y_scale;
uniform vec2 plane;
out vec3 v_n;
out vec2 v_uv;
out float v_h;
out vec2 v_pick;
void main() {
    vec3 pos = vec3(xz.x * plane.x, h * y_scale, xz.y * plane.y);
    v_n = nrm;
    v_uv = uv;
    v_h = h;
    v_pick = xz;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

_LAND_FRAG = """
#version 330
uniform sampler2D color_tex;
uniform vec3 light_dir;
in vec3 v_n;
in vec2 v_uv;
in float v_h;
in vec2 v_pick;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_pick;
void main() {
    vec3 n = normalize(v_n);
    vec3 light = normalize(light_dir);
    vec3 fill = normalize(vec3(-0.25, 0.9, 0.35));
    float ndl = clamp(dot(n, light), 0.0, 1.0);
    float ndf = clamp(dot(n, fill), 0.0, 1.0);
    vec3 albedo = texture(color_tex, v_uv).rgb;
    vec3 col = albedo * (0.40 + 0.58 * ndl + 0.16 * ndf);
    if (v_h < 0.0) {
        col *= mix(vec3(1.0), vec3(0.42, 0.55, 0.62), clamp(-v_h / 0.25, 0.0, 1.0));
    }
    out_color = vec4(clamp(col, 0.0, 1.0), 1.0);
    out_pick = v_pick;
}
"""

_SEA_VERT = """
#version 330
in vec2 xz;
in vec2 uv;
uniform mat4 mvp;
uniform float y_scale;
uniform vec2 plane;
uniform float time;
uniform sampler2D height_tex;
out vec3 v_n;
out vec2 v_uv;
out vec3 v_world;
out float v_bed;

float waves(vec2 p) {
    float w = sin(p.x * 34.0 + time * 1.3) * sin(p.y * 29.0 + time * 0.6);
    w += sin(p.x * 16.0 - p.y * 21.0 + time * 0.85) * 0.55;
    w += sin((p.x + p.y) * 8.0 + time * 0.45) * 0.35;
    return w;
}

void main() {
    float bed = texture(height_tex, uv).r;
    float offshore = smoothstep(0.02, -0.08, bed);
    vec2 p = xz;
    float amp = 0.015 * offshore;
    float wh = waves(p) * amp;
    float e = 0.004;
    float dx = (waves(p + vec2(e, 0.0)) - waves(p - vec2(e, 0.0))) * amp;
    float dz = (waves(p + vec2(0.0, e)) - waves(p - vec2(0.0, e))) * amp;
    vec3 pos = vec3(xz.x * plane.x, wh * y_scale, xz.y * plane.y);
    v_n = normalize(vec3(-dx * 18.0, 1.0, -dz * 18.0));
    v_uv = uv;
    v_world = pos;
    v_bed = bed;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

_SEA_FRAG = """
#version 330
uniform vec3 light_dir;
uniform vec3 eye;
uniform vec3 sky;
uniform float time;
in vec3 v_n;
in vec2 v_uv;
in vec3 v_world;
in float v_bed;
layout(location = 0) out vec4 out_color;

void main() {
    if (v_bed > 0.012) {
        discard;
    }
    vec3 n = normalize(v_n);
    vec3 light = normalize(light_dir);
    vec3 view = normalize(eye - v_world);
    float ndl = clamp(dot(n, light), 0.0, 1.0);
    float fres = pow(1.0 - clamp(dot(view, n), 0.0, 1.0), 3.0);
    float spec = pow(clamp(dot(reflect(-light, n), view), 0.0, 1.0), 48.0);
    float depth = clamp(-v_bed / 0.45, 0.0, 1.0);
    vec3 deep = vec3(0.03, 0.16, 0.32);
    vec3 shallow = vec3(0.10, 0.42, 0.46);
    vec3 col = mix(shallow, deep, depth);
    col *= 0.55 + 0.6 * ndl;
    col += fres * sky * 0.85;
    col += spec * vec3(0.85, 0.9, 0.95);
    float alpha = mix(0.42, 0.88, depth);
    alpha = clamp(alpha + fres * 0.2, 0.35, 0.95);
    out_color = vec4(clamp(col, 0.0, 1.0), alpha);
}
"""


def _sharp_texture(texture, moderngl, ctx):
    """Mipmaps and anisotropy, so an angled view does not smear the preview."""
    texture.repeat_x = False
    texture.repeat_y = False
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    limit = 1.0
    info = getattr(ctx, "info", None) or {}
    for key in ("GL_MAX_TEXTURE_MAX_ANISOTROPY", "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT"):
        try:
            limit = float(info[key])
        except (KeyError, TypeError, ValueError):
            continue
        break
    if limit > 1.0:
        texture.anisotropy = min(16.0, limit)


def _resolution(rows, cols, limit):
    long_edge = max(int(rows), int(cols))
    if long_edge <= int(limit):
        return int(rows), int(cols)
    scale = float(limit) / float(long_edge)
    return max(2, int(round(rows * scale))), max(2, int(round(cols * scale)))


def vertical_scale(rows, cols, layout, world_width_m):
    """Height and the two ground axes, in the same metres.

    The sheet's x axis runs 0..1 across every tile. One height unit is
    `METRES_PER_UNIT`, so it has to be divided by the sheet's width in metres
    or the peaks stand several times taller than the ground they sit on.
    North-south uses the same metres per pixel as east-west.
    """
    nx = 1 if not layout else max(1, int(layout[2]))
    width_m = float(world_width_m) * nx
    if not math.isfinite(width_m) or width_m <= 0:
        width_m = 200_000.0 * nx
    y_scale = float(METRES_PER_UNIT) / width_m
    z_span = float(rows - 1) / float(max(int(cols) - 1, 1))
    return y_scale, z_span


def build_geometry(height, color, mask=None, max_edge=MAX_EDGE, y_scale=Y_SCALE, z_span=1.0):
    """Vertex buffer, triangle indices, albedo, and the full-resolution bed."""
    height = np.ascontiguousarray(np.asarray(height, np.float32))
    if height.ndim != 2 or height.shape[0] < 2 or height.shape[1] < 2:
        raise ValueError("height must be a 2D grid")
    rows, cols = height.shape
    if mask is None:
        mask = np.ones((rows, cols), np.float32)
    else:
        mask = np.asarray(mask, np.float32)
        if mask.shape != height.shape:
            mask = np.ones((rows, cols), np.float32)
    color = _fit_color(color, height.shape)
    vr, vc = _resolution(rows, cols, max_edge)
    z01 = np.linspace(0, 1, vr, dtype=np.float32)
    x01 = np.linspace(0, 1, vc, dtype=np.float32)
    if (vr, vc) == (rows, cols):
        hs = height
        ms = np.ascontiguousarray(mask, np.float32)
    else:
        zz, xx = np.meshgrid(z01, x01, indexing="ij")
        hs = np.asarray(_sample(height, zz, xx), np.float32)
        ms = np.asarray(_sample(mask, zz, xx), np.float32)
    gy, gx = np.gradient(hs)
    nx = -gx * np.float32(y_scale * max(vc - 1, 1))
    nz = -gy * np.float32(y_scale * max(vr - 1, 1) / max(float(z_span), 1e-6))
    ny = np.ones_like(nx, np.float32)
    length = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), np.float32(1e-8))
    zz, xx = np.meshgrid(z01, x01, indexing="ij")
    verts = np.empty((vr * vc, 8), np.float32)
    verts[:, 0] = xx.ravel()
    verts[:, 1] = zz.ravel()
    verts[:, 2] = hs.ravel()
    verts[:, 3] = xx.ravel()
    verts[:, 4] = zz.ravel()
    verts[:, 5] = (nx / length).ravel()
    verts[:, 6] = (ny / length).ravel()
    verts[:, 7] = (nz / length).ravel()
    valid = ((ms[:-1, :-1] >= 0.5) & (ms[:-1, 1:] >= 0.5)
             & (ms[1:, :-1] >= 0.5) & (ms[1:, 1:] >= 0.5))
    rr, cc = np.nonzero(valid)
    i00 = (rr * np.uint32(vc) + cc).astype(np.uint32)
    idx = np.empty((i00.size, 6), np.uint32)
    idx[:, 0] = i00
    idx[:, 1] = i00 + np.uint32(1)
    idx[:, 2] = i00 + np.uint32(vc + 1)
    idx[:, 3] = i00
    idx[:, 4] = i00 + np.uint32(vc + 1)
    idx[:, 5] = i00 + np.uint32(vc)
    albedo = np.ascontiguousarray(np.clip(np.rint(color), 0, 255).astype(np.uint8))
    return verts, idx.ravel(), albedo, np.ascontiguousarray(height, np.float32)


def _mvp(yaw, pitch, distance, origin, height, view_w, view_h, y_scale=Y_SCALE, plane=(1.0, 1.0)):
    """Match the software camera: yaw, pitch, and distance orbit a point on the sheet."""
    pitch = float(np.clip(pitch, 0.15, 1.25))
    distance = float(max(0.35, distance))
    yaw = float(yaw)
    if origin is None:
        ox, oz = 0.5, 0.5
    else:
        ox, oz = float(origin[0]), float(origin[1])
    if 0.0 <= ox <= 1.0 and 0.0 <= oz <= 1.0:
        mid = float(_sample(height, np.float32(oz), np.float32(ox))) * y_scale
    else:
        mid = float(np.median(height)) * y_scale
    target = np.array([ox * float(plane[0]), mid, oz * float(plane[1])], np.float64)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    eye = target + np.array([distance * cp * sy, distance * sp, distance * cp * cy], np.float64)
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-8)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(up, eye))
    view[2, 3] = float(np.dot(forward, eye))
    aspect = float(view_w) / max(float(view_h), 1.0)
    f = 1.0 / FOV
    near = max(0.001, min(0.02, distance * 0.02))
    far = max(40.0, distance * 8.0)
    proj = np.zeros((4, 4), np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    mvp = (proj @ view).astype(np.float32)
    return np.ascontiguousarray(mvp), eye.astype(np.float32)


def frame_distance(yaw, pitch, plane, view_w, view_h, origin=(0.5, 0.5),
                   y_scale=0.0, height_peak=0.0):
    """How far back the camera must sit for every corner of the sheet to stay in view."""
    plane = (float(plane[0]), float(plane[1]))
    peak = max(0.0, float(height_peak)) * float(y_scale)
    corners = [(x, z, y) for z in (0.0, 1.0) for x in (0.0, 1.0) for y in (0.0, peak)]
    flat = np.zeros((2, 2), np.float32)

    def fits(distance):
        mvp, _eye = _mvp(
            yaw, pitch, distance, origin, flat, view_w, view_h,
            y_scale=y_scale, plane=plane)
        for x, z, y in corners:
            clip = mvp.astype(np.float64) @ np.array(
                [x * plane[0], y, z * plane[1], 1.0], np.float64)
            if clip[3] <= 1e-4:
                return False
            ndc = clip[:3] / clip[3]
            # Depth is not the picture. A close near-plane pushes ndc z toward 1
            # for the whole sheet, and treating that as "off screen" threw the
            # camera back to the search cap.
            if abs(float(ndc[0])) > 0.92 or abs(float(ndc[1])) > 0.92:
                return False
            if abs(float(ndc[2])) > 1.0:
                return False
        return True

    lo, hi = 0.35, 1.0
    if not fits(hi):
        for _ in range(10):
            hi *= 1.7
            if fits(hi):
                break
    for _ in range(18):
        mid = (lo + hi) * 0.5
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


class LiveTerrain:
    """GPU picture of one stitched sheet. `draw` moves the sea every call."""

    def __init__(self):
        import moderngl

        self._moderngl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.bg_prog = self.ctx.program(vertex_shader=_BG_VERT, fragment_shader=_BG_FRAG)
        self.land = self.ctx.program(vertex_shader=_LAND_VERT, fragment_shader=_LAND_FRAG)
        self.sea = self.ctx.program(vertex_shader=_SEA_VERT, fragment_shader=_SEA_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self._quad = self.ctx.buffer(quad.tobytes())
        self.bg = self.ctx.vertex_array(self.bg_prog, [(self._quad, "2f", "in_pos")])
        self.vbo = None
        self.ibo = None
        self.land_vao = None
        self.sea_vao = None
        self.albedo = None
        self.bed = None
        self.fb_color = None
        self.pick_tex = None
        self.depth = None
        self.fbo = None
        self.sea_fbo = None
        self.index_count = 0
        self.sheet_key = None
        self.layout = None
        self.height = None
        self.y_scale = Y_SCALE
        self.plane = (1.0, 1.0)
        self.frame = None
        self.frame_key = None
        self._size = None

    def close(self):
        self._drop_mesh()
        self._drop_targets()
        if self._quad is not None:
            self._quad.release()
            self._quad = None
        self.ctx.release()

    def _drop_mesh(self):
        for name in ("land_vao", "sea_vao", "vbo", "ibo", "albedo", "bed"):
            obj = getattr(self, name)
            if obj is not None:
                obj.release()
                setattr(self, name, None)
        self.index_count = 0

    def _drop_targets(self):
        for name in ("fbo", "sea_fbo", "fb_color", "pick_tex", "depth"):
            obj = getattr(self, name)
            if obj is not None:
                obj.release()
                setattr(self, name, None)
        self._size = None

    def set_sheet(self, height, color, mask, layout, key, world_width_m=200_000.0):
        y_scale, z_span = vertical_scale(height.shape[0], height.shape[1], layout, world_width_m)
        self.y_scale = y_scale
        self.plane = (1.0, z_span)
        verts, indices, albedo, bed = build_geometry(
            height, color, mask, y_scale=y_scale, z_span=z_span)
        self._drop_mesh()
        mgl = self._moderngl
        self.vbo = self.ctx.buffer(verts.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        fmt = "2f 1f 2f 3f"
        names = ("xz", "h", "uv", "nrm")
        self.land_vao = self.ctx.vertex_array(
            self.land, [(self.vbo, fmt, *names)], self.ibo)
        self.sea_vao = self.ctx.vertex_array(
            self.sea, [(self.vbo, "2f 4x 2f 12x", "xz", "uv")], self.ibo)
        self.index_count = int(indices.size)
        self.albedo = self.ctx.texture((albedo.shape[1], albedo.shape[0]), 3, albedo.tobytes())
        _sharp_texture(self.albedo, mgl, self.ctx)
        self.bed = self.ctx.texture(
            (bed.shape[1], bed.shape[0]), 1, bed.tobytes(), dtype="f4")
        self.bed.filter = (mgl.LINEAR, mgl.LINEAR)
        self.bed.repeat_x = False
        self.bed.repeat_y = False
        self.height = bed
        self.layout = layout
        self.sheet_key = key
        self.frame = None
        self.frame_key = None

    def _ensure_targets(self, view_w, view_h):
        size = (int(view_w), int(view_h))
        if self._size == size:
            return
        self._drop_targets()
        w, h = size
        self.fb_color = self.ctx.texture((w, h), 4)
        self.pick_tex = self.ctx.texture((w, h), 2, dtype="f4")
        self.depth = self.ctx.depth_texture((w, h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.fb_color, self.pick_tex],
            depth_attachment=self.depth)
        self.sea_fbo = self.ctx.framebuffer(
            color_attachments=[self.fb_color],
            depth_attachment=self.depth)
        self._size = size

    def draw(self, yaw, pitch, distance, view_w, view_h, origin, time):
        if self.height is None or self.land_vao is None:
            raise RuntimeError("no terrain sheet")
        view_w = int(max(8, view_w))
        view_h = int(max(8, view_h))
        self._ensure_targets(view_w, view_h)
        mvp, eye = _mvp(
            yaw, pitch, distance, origin, self.height, view_w, view_h,
            y_scale=self.y_scale, plane=self.plane)
        mvp_bytes = np.ascontiguousarray(mvp).tobytes(order="F")
        mgl = self._moderngl
        self.fbo.use()
        self.fbo.clear(depth=1.0)
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.BLEND)
        self.bg_prog["sky"].value = SKY
        self.bg.render(mgl.TRIANGLE_STRIP)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        if self.index_count:
            self.land["mvp"].write(mvp_bytes)
            self.land["y_scale"].value = self.y_scale
            self.land["plane"].value = self.plane
            self.land["light_dir"].value = LIGHT
            self.albedo.use(0)
            self.land["color_tex"].value = 0
            self.ctx.depth_mask = True
            self.land_vao.render()
            self.sea_fbo.use()
            self.ctx.enable(mgl.BLEND)
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            self.ctx.depth_mask = False
            self.sea["mvp"].write(mvp_bytes)
            self.sea["y_scale"].value = self.y_scale
            self.sea["plane"].value = self.plane
            self.sea["time"].value = float(time)
            self.sea["light_dir"].value = LIGHT
            self.sea["eye"].value = (float(eye[0]), float(eye[1]), float(eye[2]))
            self.sea["sky"].value = SKY
            self.bed.use(1)
            self.sea["height_tex"].value = 1
            self.sea_vao.render()
            self.ctx.depth_mask = True
            self.ctx.disable(mgl.BLEND)
        rgb, pick_y, pick_x = self._read(view_w, view_h)
        self.frame = (rgb, pick_y, pick_x)
        return rgb, pick_y, pick_x

    def _read(self, view_w, view_h):
        self.fbo.use()
        raw = self.fbo.read(components=3, attachment=0, alignment=1)
        rgb = np.flipud(np.frombuffer(raw, np.uint8).reshape((view_h, view_w, 3))).copy()
        praw = self.fbo.read(components=2, attachment=1, dtype="f4", alignment=4)
        pick = np.flipud(np.frombuffer(praw, np.float32).reshape((view_h, view_w, 2))).copy()
        return rgb, pick[:, :, 1].astype(np.float32), pick[:, :, 0].astype(np.float32)
