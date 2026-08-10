"""pygame/SDL2 rendering of a composited FaceState.

Runs in two places from one code path:

* On the Pi, with ``SDL_VIDEODRIVER=kmsdrm``, SDL renders straight to the DSI
  panel through KMS/DRM. No X, no Wayland, no desktop session -- which is why
  the robot can run Ubuntu Server and still have a face.
* On a dev machine, in a plain window, so expressions can be iterated on
  without the robot or the panel being anywhere nearby.

The face is two eyes on black -- no mouth, no brows. Everything is drawn from
primitives (ellipses, rounded rects, polygon lid masks) rather than loaded from
images, which keeps the whole face continuously parameterised and leaves no
assets to fall out of sync with the code.
"""

import math
import os
from typing import Dict, Optional, Tuple

import pygame

from .easing import clamp
from .face_state import EyeParams, FaceState

# Screen regions reported back on a touch, matching TouchEvent constants.
REGION_OTHER = 0
REGION_LEFT_EYE = 1
REGION_RIGHT_EYE = 2

#: Scratch surfaces are sized up to a multiple of this so that a continuously
#: blending eye reuses one buffer instead of allocating a slightly different
#: one every frame. At 60 fps with eyes half the screen tall, that churn is
#: several megabytes a second of pointless allocation on the Pi.
_SURFACE_GRANULARITY = 32


def _quantize(value: int, step: int = _SURFACE_GRANULARITY) -> int:
    return ((max(1, value) + step - 1) // step) * step


def create_display(
    width: int = 0,
    height: int = 0,
    fullscreen: bool = True,
    video_driver: str = "",
    hide_cursor: bool = True,
) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """Bring up the SDL display and return (surface, (width, height)).

    Passing 0 for width/height adopts the panel's native mode, which is what
    you want on the Pi -- hard-coding 800x480 breaks the moment the panel is
    swapped or rotated.
    """
    if video_driver:
        os.environ["SDL_VIDEODRIVER"] = video_driver

    pygame.display.init()
    pygame.font.init()

    if width <= 0 or height <= 0:
        info = pygame.display.Info()
        width = width if width > 0 else info.current_w
        height = height if height > 0 else info.current_h
        # Some KMSDRM setups report a bogus mode before the first set_mode.
        if width <= 0 or height <= 0 or width > 8192 or height > 8192:
            width, height = 800, 480

    flags = pygame.FULLSCREEN if fullscreen else 0
    surface = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("burgerbot face")
    if hide_cursor:
        try:
            pygame.mouse.set_visible(False)
        except pygame.error:
            pass  # No cursor concept under KMSDRM.
    return surface, (width, height)


class Renderer:
    """Draws a FaceState. Stateless apart from surface caches and hit regions."""

    def __init__(
        self,
        surface: pygame.Surface,
        background=(0, 0, 0),
        supersample: int = 2,
    ):
        self.surface = surface
        self.background = background
        self.width, self.height = surface.get_size()

        # Supersample then downscale: pygame has no antialiased fills, and
        # aliased eye edges are very visible on a 7" panel at arm's length.
        # Costs fill rate, so drop to 1 if the Pi cannot hold frame rate.
        self.supersample = max(1, int(supersample))
        if self.supersample > 1:
            self._canvas = pygame.Surface(
                (self.width * self.supersample, self.height * self.supersample)
            )
        else:
            self._canvas = surface

        cw, ch = self._canvas.get_size()
        #: Uniform scale, so a circle is a circle on a non-square panel.
        self._scale = min(cw, ch) / 2.0
        self._cx = cw / 2.0
        self._cy = ch / 2.0

        self._eye_cache: Dict[Tuple, pygame.Surface] = {}
        self._hit_regions = {}

    # ---- coordinate helpers ------------------------------------------

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        """Normalised face coords -> canvas pixels (y is up in face space)."""
        return self._cx + x * self._scale, self._cy - y * self._scale

    def to_norm(self, px: float, py: float) -> Tuple[float, float]:
        """Screen pixels -> normalised face coords. Used for touch."""
        sx = px * self.supersample
        sy = py * self.supersample
        return (sx - self._cx) / self._scale, (self._cy - sy) / self._scale

    def _len(self, v: float) -> float:
        return v * self._scale

    def _scratch(self, key, size, clear: bool = True) -> pygame.Surface:
        """A reusable per-purpose scratch surface of at least `size`."""
        surf = self._eye_cache.get(key)
        if surf is None or surf.get_size() != size:
            surf = pygame.Surface(size, pygame.SRCALPHA)
            self._eye_cache[key] = surf
        elif clear:
            surf.fill((0, 0, 0, 0))
        return surf

    # ---- public API ---------------------------------------------------

    def draw(self, state: FaceState) -> None:
        canvas = self._canvas
        canvas.fill(self.background)
        self._hit_regions = {}

        color = tuple(int(clamp(c) * 255) for c in state.color[:3])
        tilt = state.face_tilt

        self._draw_eye(state, state.left, color, tilt, REGION_LEFT_EYE)
        self._draw_eye(state, state.right, color, tilt, REGION_RIGHT_EYE)

        if self.supersample > 1:
            pygame.transform.smoothscale(
                canvas, (self.width, self.height), self.surface
            )

    def present(self) -> None:
        pygame.display.flip()

    def hit_test(self, px: float, py: float) -> int:
        """Which feature is at this screen pixel, for touch reporting."""
        for region, rect in self._hit_regions.items():
            if rect.collidepoint(px * self.supersample, py * self.supersample):
                return region
        return REGION_OTHER

    # ---- feature drawing ----------------------------------------------

    def _face_transform(self, state: FaceState, x: float, y: float):
        """Apply the whole-face offset and scale to a point in face space."""
        return (
            x * state.face_scale_x + state.face_offset_x,
            y * state.face_scale_y + state.face_offset_y,
        )

    def _draw_eye(
        self,
        state: FaceState,
        eye: EyeParams,
        color,
        tilt: float,
        region: int,
    ) -> None:
        w = self._len(eye.width * state.face_scale_x)
        h = self._len(eye.height * state.face_scale_y)
        if w < 1.0 or h < 1.0:
            return

        cx, cy = self._face_transform(state, eye.center_x, eye.center_y)
        px, py = self.to_px(cx, cy)

        # Margin for the angled lid polygon to reach past the eye's corners.
        # Rotation needs no allowance here -- pygame.transform.rotate returns a
        # grown surface of its own.
        pad = int(max(w, h) * 0.12) + 4
        sw = _quantize(int(w) + 2 * pad)
        sh = _quantize(int(h) + 2 * pad)
        eye_surf = self._scratch(("eye", region), (sw, sh))

        body = pygame.Rect(0, 0, int(w), int(h))
        body.center = (sw // 2, sh // 2)
        if eye.corner_radius >= 0.999:
            # A fully rounded tall rect is a capsule -- straight sides with
            # domed ends. That is not the same silhouette as an oval, and on a
            # cartoon face the difference is very visible, so draw a real
            # ellipse at the top of the range.
            pygame.draw.ellipse(eye_surf, color, body)
        else:
            radius = int(min(w, h) / 2.0 * clamp(eye.corner_radius))
            pygame.draw.rect(eye_surf, color, body, border_radius=radius)

        if eye.pupil_radius > 0.001:
            pr = self._len(eye.pupil_radius)
            ppx = body.centerx + eye.pupil_x * (w / 2.0 - pr)
            ppy = body.centery - eye.pupil_y * (h / 2.0 - pr)
            pygame.draw.circle(
                eye_surf, self.background, (int(ppx), int(ppy)), int(pr)
            )

        self._apply_lids(eye_surf, body, eye, region)

        # Per-eye roll plus the whole-face tilt, in one rotation. Lids are
        # applied first so they rotate with the eye rather than staying level,
        # which is what makes a rolled eye still look like an eye.
        total_rot = eye.rotation + tilt
        if abs(total_rot) > 0.008:
            eye_surf = pygame.transform.rotate(eye_surf, math.degrees(total_rot))

        rect = eye_surf.get_rect(center=(int(px), int(py)))
        self._canvas.blit(eye_surf, rect)
        self._hit_regions[region] = rect

    def _apply_lids(
        self,
        eye_surf: pygame.Surface,
        body: pygame.Rect,
        eye: EyeParams,
        region: int,
    ) -> None:
        """Erase the lidded parts of an eye.

        Builds an alpha mask and multiplies it in, rather than painting lids in
        the background colour. Slightly more work, but it keeps the eye a
        genuinely transparent sprite so it can be rotated for the head tilt
        without dragging a rectangle of background around with it.
        """
        if eye.lid_upper <= 0.001 and eye.lid_lower <= 0.001:
            return

        sw, sh = eye_surf.get_size()
        mask = self._scratch(("mask", region), (sw, sh), clear=False)
        mask.fill((255, 255, 255, 255))

        # Just enough to reach the surface edges. Extending further makes an
        # angled lid's far end shoot off at a wild y and miss the eye entirely.
        overhang = (sw - body.width) / 2.0
        cut = (0, 0, 0, 0)

        if eye.lid_upper > 0.001:
            y = body.top + body.height * eye.lid_upper
            pts = self._lid_polygon(body, y, eye.lid_angle, overhang, upper=True)
            pygame.draw.polygon(mask, cut, pts)

        if eye.lid_lower > 0.001:
            y = body.bottom - body.height * eye.lid_lower
            # The lower lid stays level; angling both reads as a squint rather
            # than the expression the lid_angle was authored for.
            pts = self._lid_polygon(body, y, 0.0, overhang, upper=False)
            pygame.draw.polygon(mask, cut, pts)

        eye_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    @staticmethod
    def _lid_polygon(body: pygame.Rect, y: float, angle: float, overhang: float, upper: bool):
        """A lid as a half-plane polygon, tilted about the eye centre.

        The lid edge is the straight line through (centre_x, y) with slope
        tan(angle), evaluated at the polygon's x extents. `lid_angle` is
        authored as "inner corner down"; EyeParams mirroring already negates it
        for the right eye, so no extra sign handling belongs here.
        """
        cx = body.centerx
        slope = math.tan(angle)
        x0 = body.left - overhang
        x1 = body.right + overhang
        y0 = y + slope * (x0 - cx)
        y1 = y + slope * (x1 - cx)
        span = body.height + overhang * 4
        far = min(y0, y1) - span if upper else max(y0, y1) + span
        return [(x0, y0), (x1, y1), (x1, far), (x0, far)]
