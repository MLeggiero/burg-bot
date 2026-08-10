# A request for the robot's face to adopt an expression.
#
# Many sources publish these concurrently (navigation status, obstacle
# proximity, localization health, battery, touch, behaviour trees). The mood
# arbiter in burgerbot_expressions picks the highest-priority candidate that
# has not yet expired -- ties break toward the most recent. Without that
# arbitration the face flickers between competing sources, which reads as
# broken rather than alive.

# ---- Named expressions (keyframes defined in burgerbot_face/expressions.py) ----
string NEUTRAL=neutral
string HAPPY=happy
string CURIOUS=curious
string FOCUSED=focused
string CONFUSED=confused
string STARTLED=startled
string SAD=sad
string SLEEPY=sleepy
string DETERMINED=determined
string ERROR=error
string NERVOUS=nervous

# ---- Priority bands ----
# AMBIENT  idle personality, lowest claim on the face
# TASK     what the robot is currently doing (navigating, mapping)
# CONCERN  something is going wrong but is recoverable (lost, stuck, low battery)
# ALERT    immediate physical events that should always show (bump, e-stop)
uint8 PRIORITY_AMBIENT=10
uint8 PRIORITY_TASK=50
uint8 PRIORITY_CONCERN=120
uint8 PRIORITY_ALERT=200

std_msgs/Header header

# Identifies the publisher, e.g. "nav_status", "proximity", "touch".
# A new command from a source replaces that source's previous candidate rather
# than stacking, so a chatty publisher cannot crowd out the arbiter.
string source

# One of the named expressions above.
string expression

# Blend weight toward the keyframe, 0..1. Values below 1 mix with neutral, so
# a source can express "slightly worried" without committing to the full pose.
float32 intensity 1.0

uint8 priority 10

# How long this candidate stays valid. Zero means it holds until the same
# source replaces or clears it. Anything event-driven should set this --
# otherwise a one-off startle sticks to the face forever.
builtin_interfaces/Duration duration

# Seconds to ease from the current pose into this one. Fast for reactions,
# slow for mood drift.
float32 blend_time 0.25
