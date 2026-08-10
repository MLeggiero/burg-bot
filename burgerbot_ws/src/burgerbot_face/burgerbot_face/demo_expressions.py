"""Drive the face with no robot and no ROS graph attached.

This is the iteration loop for animation work. Editing an expression and
watching it on the real robot means a build, a deploy and a nav stack; here it
is a keypress. Run it on the dev machine while tuning `expressions.py`.

    ros2 run burgerbot_face demo_expressions
    python3 -m burgerbot_face.demo_expressions --windowed

Keys:
    0-9         select an expression
    tab         cycle to the next expression
    arrows      steer the gaze; `g` releases it back to idle drift
    b           blink now
    r/s/x/n     fire a recoil / shake / squash / bounce reaction
    [ ]         nudge the simulated angular velocity (inertia + gaze lead)
    , .         nudge the simulated linear velocity (squash and stretch)
    space       reset simulated motion to zero
    f           toggle the fps readout
    esc / q     quit

Also runs headless to dump reference PNGs, which is how the renderer gets
checked without a display attached:

    python3 -m burgerbot_face.demo_expressions --dump-dir /tmp/faces

And headless to render actual animated GIFs -- real motion, not a still frame,
which is the only way to review blinking, breathing, sway and gaze drift
without hardware or a window:

    python3 -m burgerbot_face.demo_expressions --dump-gif /tmp/faces --gif-scenario idle
    python3 -m burgerbot_face.demo_expressions --dump-gif /tmp/faces --gif-scenario all

Requires Pillow (`pip install pillow`) -- a demo-only dependency, not part of
the robot's runtime.
"""

import argparse
import math
import os
import random
import sys
import time

from . import expressions
from .animator import Animator
from .layers import Compositor
from .renderer import Renderer, create_display


def _dump(args) -> int:
    """Render one PNG per expression, plus a blend strip. No display needed."""
    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.makedirs(args.dump_dir, exist_ok=True)

    pygame.display.init()
    surface = pygame.display.set_mode((args.width, args.height))
    renderer = Renderer(surface, supersample=args.supersample)

    names = expressions.names()
    for name in names:
        animator = Animator(name)
        compositor = Compositor()
        # Step past the blend so the pose is fully settled, and keep the
        # layers quiet so the reference images are reproducible.
        compositor.blink.enabled = False
        compositor.idle.amplitude = 0.0
        compositor.gaze.freeze()
        animator.update(5.0)
        state = compositor.compose(animator.state)
        renderer.draw(state)
        path = os.path.join(args.dump_dir, f"{name}.png")
        pygame.image.save(surface, path)
        print(f"  wrote {path}")

    # A blend strip: neutral -> startled sampled mid-transition, to confirm
    # intermediate poses are sane rather than only the endpoints.
    animator = Animator("neutral")
    compositor = Compositor()
    compositor.blink.enabled = False
    compositor.idle.amplitude = 0.0
    animator.set_expression("startled")
    for i in range(5):
        state = compositor.compose(animator.state)
        renderer.draw(state)
        path = os.path.join(args.dump_dir, f"blend_{i}.png")
        pygame.image.save(surface, path)
        print(f"  wrote {path}")
        animator.update(animator._duration / 4.0)

    print(f"\n{len(names)} expressions + 5 blend frames -> {args.dump_dir}")
    return 0


def _capture_frame(surface) -> "Image.Image":
    from PIL import Image
    import pygame

    # pygame.image.tostring does the pixel-format conversion explicitly, so
    # this does not depend on whatever native format the dummy driver's
    # surface happens to use (which varies by platform and is not guaranteed
    # to be 24-bit RGB packed the way a raw buffer read would assume).
    w, h = surface.get_size()
    tobytes = getattr(pygame.image, "tobytes", pygame.image.tostring)
    data = tobytes(surface, "RGB")
    return Image.frombytes("RGB", (w, h), data)


def _save_gif(frames, path: str, fps: float) -> None:
    frame_ms = max(20, round(1000.0 / fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )


def _dump_gif(args) -> int:
    """Render actual animated GIFs -- real motion, not a still frame.

    A still PNG cannot show blinking, breathing, sway or gaze drift, and those
    are most of what this file is about. This is how that gets reviewed
    without a window, a robot, or hardware attached.
    """
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print(
            "Pillow is required for --dump-gif. Install it with:\n"
            "  pip install pillow\n"
            "(a demo-only dependency -- not part of the robot's runtime)",
            file=sys.stderr,
        )
        return 1

    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.makedirs(args.dump_gif, exist_ok=True)

    pygame.display.init()
    surface = pygame.display.set_mode((args.gif_width, args.gif_height))
    renderer = Renderer(surface, supersample=args.supersample)
    dt = 1.0 / args.gif_fps

    def render_run(steps_and_holds, path, seed):
        """steps_and_holds: [(expression_name, hold_seconds), ...]"""
        animator = Animator(steps_and_holds[0][0])
        compositor = Compositor(rng=random.Random(seed))

        # Draw and capture the settled t=0 pose before the loop starts, or
        # frame 0 is whatever stale pixels were left on `surface` from the
        # previous run_render call (or an uninitialised black frame on the
        # very first one) rather than the actual starting expression.
        compositor.set_ambient(animator.ambient)
        renderer.draw(compositor.compose(animator.state))
        frames = [_capture_frame(surface)]

        for i, (name, hold) in enumerate(steps_and_holds):
            if i > 0:
                animator.set_expression(name, source="demo-gif", force=True)
            steps = max(1, round(hold / dt))
            for _ in range(steps):
                animator.update(dt)
                compositor.set_ambient(animator.ambient)
                compositor.update(dt)
                renderer.draw(compositor.compose(animator.state))
                frames.append(_capture_frame(surface))

        _save_gif(frames, path, args.gif_fps)
        print(f"  wrote {path}  ({len(frames)} frames, {len(frames) * dt:.1f}s)")

    if args.gif_scenario in ("single", "all"):
        name = args.gif_expression or expressions.DEFAULT
        path = os.path.join(args.dump_gif, f"{name}_idle.gif")
        render_run([(name, args.gif_duration)], path, seed=f"gif-{name}")

    if args.gif_scenario in ("cycle", "all"):
        # Every expression in turn, holding long enough to see its idle
        # character (breathing, sway, blink rate) as well as the blend in and
        # out of it. This one file is the fastest way to review the whole
        # emotional range at once.
        order = [n for n in expressions.names() if n != expressions.DEFAULT]
        sequence = [(expressions.DEFAULT, 1.0)]
        for name in order:
            sequence.append((name, 1.6))
        sequence.append((expressions.DEFAULT, 1.0))
        path = os.path.join(args.dump_gif, "expression_cycle.gif")
        render_run(sequence, path, seed="gif-cycle")

    print(f"\nGIFs written to {args.dump_gif}")
    return 0


def _interactive(args) -> int:
    import pygame

    surface, (w, h) = create_display(
        width=args.width,
        height=args.height,
        fullscreen=not args.windowed,
        video_driver=args.video_driver,
        hide_cursor=not args.windowed,
    )
    renderer = Renderer(surface, supersample=args.supersample)
    animator = Animator()
    compositor = Compositor()

    names = expressions.names()
    index = names.index(expressions.DEFAULT) if expressions.DEFAULT in names else 0
    print("Expressions:")
    for i, n in enumerate(names):
        print(f"  {i}  {n}")

    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    show_fps = True
    gaze = [0.0, 0.0]
    gaze_held = False
    lin = 0.0
    ang = 0.0
    running = True
    last = time.monotonic()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_0 <= k <= pygame.K_9:
                    i = k - pygame.K_0
                    if i < len(names):
                        index = i
                        animator.set_expression(names[i], source="demo")
                elif k == pygame.K_TAB:
                    index = (index + 1) % len(names)
                    animator.set_expression(names[index], source="demo")
                elif k == pygame.K_b:
                    compositor.blink.trigger()
                elif k == pygame.K_r:
                    compositor.reaction.fire("recoil")
                elif k == pygame.K_s:
                    compositor.reaction.fire("shake")
                elif k == pygame.K_x:
                    compositor.reaction.fire("squash")
                elif k == pygame.K_n:
                    compositor.reaction.fire("bounce")
                elif k == pygame.K_g:
                    gaze_held = False
                    compositor.gaze.release()
                elif k == pygame.K_LEFTBRACKET:
                    ang += 0.4
                elif k == pygame.K_RIGHTBRACKET:
                    ang -= 0.4
                elif k == pygame.K_COMMA:
                    lin -= 0.15
                elif k == pygame.K_PERIOD:
                    lin += 0.15
                elif k == pygame.K_SPACE:
                    lin = ang = 0.0
                elif k == pygame.K_f:
                    show_fps = not show_fps
            elif event.type == pygame.MOUSEBUTTONDOWN:
                nx, ny = renderer.to_norm(*event.pos)
                region = renderer.hit_test(*event.pos)
                print(f"touch at ({nx:+.2f}, {ny:+.2f}) region={region}")
                compositor.reaction.fire("bounce", 0.8)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            gaze[0] += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.05
            gaze[1] += (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 0.05
            gaze[0] = max(-1.0, min(1.0, gaze[0]))
            gaze[1] = max(-1.0, min(1.0, gaze[1]))
            gaze_held = True
        if gaze_held:
            compositor.gaze.look_at(gaze[0], gaze[1])

        now = time.monotonic()
        dt = min(now - last, 0.1)
        last = now

        compositor.motion.set_velocity(lin, ang)
        animator.update(dt)
        compositor.set_ambient(animator.ambient)
        compositor.update(dt)
        renderer.draw(compositor.compose(animator.state))

        if show_fps:
            label = (
                f"{names[index]}   fps {clock.get_fps():5.1f}   "
                f"v {lin:+.2f} w {ang:+.2f}"
            )
            surface.blit(font.render(label, True, (110, 110, 110)), (8, 6))

        renderer.present()
        clock.tick(args.fps)

    pygame.quit()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="burgerbot face demo")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--windowed", action="store_true", help="run in a window")
    p.add_argument("--video-driver", default="", help="e.g. kmsdrm, x11, dummy")
    p.add_argument("--dump-dir", default="", help="render still reference PNGs and exit")
    p.add_argument(
        "--dump-gif", default="",
        help="render animated GIFs (real motion, not stills) and exit; needs Pillow",
    )
    p.add_argument(
        "--gif-scenario", choices=["single", "cycle", "all"], default="cycle",
        help="'single' = one expression's idle motion, 'cycle' = every "
        "expression in turn (default), 'all' = both",
    )
    p.add_argument(
        "--gif-expression", default="",
        help="expression to use for the 'single' scenario (default: neutral)",
    )
    p.add_argument(
        "--gif-duration", type=float, default=5.0,
        help="hold seconds for the 'single' scenario",
    )
    p.add_argument("--gif-fps", type=float, default=18.0)
    p.add_argument("--gif-width", type=int, default=400)
    p.add_argument("--gif-height", type=int, default=240)
    # ros2 run passes through extra args; ignore anything unrecognised.
    args, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.dump_gif:
        return _dump_gif(args)
    if args.dump_dir:
        return _dump(args)
    if not args.windowed and not args.video_driver:
        # Interactive default on a dev machine is a window, not a takeover of
        # whatever display happens to be attached.
        args.windowed = True
    return _interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
