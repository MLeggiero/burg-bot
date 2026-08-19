# What the companion behaviour is doing and, more usefully, why.
#
# Same role FaceState plays for the renderer: nothing consumes this, it exists
# so that when the robot does something inexplicable you can echo one topic and
# see the reasoning instead of inferring it from which way the wheels turned.
# For a behaviour driven by a person's movement -- something you cannot pause,
# rewind, or reproduce on demand -- that matters more than usual.

# ---- States ----
# IDLE         nobody around; ambient behaviour has the robot
# SEEKING      nobody around, heading for somewhere people are usually found
# WATCHING     someone is visible; looking at them, not committed to approach
# APPROACHING  driving to a standoff distance in front of the target
# ENGAGED      at a comfortable distance from the target, attending to them
# DANCING      performing a gesture; navigation is not driving
# WITHDRAWN    the target left repeatedly, so the robot has stopped following
# COMMANDED    told to go somewhere by burgerbot_dialog; the robot's own
#              target selection is suspended until it arrives or gives up
string IDLE=idle
string SEEKING=seeking
string WATCHING=watching
string APPROACHING=approaching
string ENGAGED=engaged
string DANCING=dancing
string WITHDRAWN=withdrawn
string COMMANDED=commanded

std_msgs/Header header

string state

# The person the robot has picked out of everyone visible. Empty in IDLE and
# SEEKING.
string target_id
string target_name
float32 target_distance

# How the robot feels about the target, -1 (has been repeatedly rebuffed) to
# +1 (this person sticks around and interacts). Decays toward 0 over time so
# one bad afternoon does not follow somebody around forever.
float32 affinity

# Consecutive approaches this person has walked out of. This is what turns
# into the sad face -- one walk-away is somebody being busy, three in a row is
# a message.
uint32 rejections

# How many people are visible right now, whether or not they are the target.
uint32 people_visible

# Plain-language reason for the current state, e.g. "target moved 1.4m away
# during approach". Written for a human reading a log at 2am.
string reason
