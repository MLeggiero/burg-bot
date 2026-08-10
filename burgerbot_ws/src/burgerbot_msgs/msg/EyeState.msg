# Rendering parameters for a single eye.
#
# The face is two eyes and nothing else -- no mouth, no brows -- so every
# emotion has to be expressible here. That is why this message carries as many
# levers as it does.
#
# Positions and sizes are in uniform normalized units so the face is resolution
# independent: y spans -1 (bottom) to +1 (top), x spans -aspect to +aspect, and
# a length of 1.0 is half the screen's shorter dimension. Uniform scaling is
# what keeps a circle circular on a non-square panel.

float32 center_x
float32 center_y

float32 width
float32 height

# 0 = sharp rectangle, 1 = a true ellipse; in between is a rounded rectangle.
float32 corner_radius 1.0

# Whole-eye roll in radians, positive = outer edge lifted. With no eyebrows,
# this does their job: ovals tilted toward each other read as a scowl.
float32 rotation

# Pupil offset from the eye centre, in units of the eye's own half-extent, so
# +-1 is the edge whatever the eye size. Radius 0 disables the pupil and the
# eye body itself carries the gaze, which is the default look.
float32 pupil_x
float32 pupil_y
float32 pupil_radius

# Fraction of the eye covered by each lid, 0..1. Both at 0.5 = fully closed.
float32 lid_upper
float32 lid_lower

# Upper-lid tilt in radians. Positive angles the inner corner down. This does
# more emotional work than any other single parameter here: inner-down is
# angry or determined, inner-up is sad or pleading.
float32 lid_angle
