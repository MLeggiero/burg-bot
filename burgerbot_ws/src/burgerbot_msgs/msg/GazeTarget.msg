# Where the eyes should look.
#
# Kept separate from ExpressionCommand because gaze updates far more often than
# mood does -- the robot can hold one expression while tracking a moving point.

uint8 MODE_IDLE=0       # release gaze; the idle layer resumes its drift
uint8 MODE_POINT=1      # track `point`, transformed through screen_optical_frame
uint8 MODE_DIRECTION=2  # look along `point` as a direction vector, not a position
uint8 MODE_SCREEN=3     # `point.x`/`point.y` are direct normalized screen coords

std_msgs/Header header

uint8 mode

# Interpreted per `mode`, in header.frame_id. The renderer transforms it into
# the screen's optical frame, so any frame in the TF tree works -- a point in
# `map`, an obstacle in `base_link`, a path pose in `odom`.
geometry_msgs/Point point

# How strongly to commit, 0..1. Partial weights read as a glance rather than a
# stare, which is usually what you want for anything peripheral.
float32 weight 1.0
