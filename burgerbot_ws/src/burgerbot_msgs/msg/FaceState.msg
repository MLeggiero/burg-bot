# The fully composited face, after every animation layer has been applied.
#
# Published at a low rate purely for introspection, recording and replay -- the
# renderer does not consume it. Handy for `ros2 topic echo` when an expression
# looks wrong and you need to see which layer is responsible.

std_msgs/Header header

# The winning expression, which source won it, and the realised blend weight
# (not the requested one).
string expression
string source
float32 intensity

EyeState left_eye
EyeState right_eye

# Whole-face transform, driven by the squash-stretch and lean layers rather
# than authored into any keyframe.
float32 face_offset_x
float32 face_offset_y
float32 face_tilt
float32 face_scale_x 1.0
float32 face_scale_y 1.0

# Feature colour, RGBA 0..1.
float32[4] color

bool blinking
