# One person the robot is currently keeping track of, in the map frame.
#
# The difference between this and PersonDetection2D is persistence. A
# detection is one glimpse; this is a thing with a history -- it has an
# identity that survives the person walking behind a chair, a velocity, and a
# record of how long the robot has known about it. Every social decision needs
# that history: you cannot tell "walking away from me" from "standing still"
# without a velocity, and you cannot tell "the same person keeps leaving" from
# "several different people passed by" without a stable id.

# Stable for the life of the track, e.g. "person_3". Reused only after the
# track has been deleted for good, never while it is merely occluded.
string track_id

# Recognised identity from face_identity, e.g. "mark". Empty when nobody
# matched, which is the normal case for a stranger and not an error -- the
# social stack works fine on anonymous tracks, it just cannot greet them by
# name or remember them across restarts.
string name
float32 name_confidence

# Map frame. Orientation is the person's estimated facing direction, which is
# only meaningful when `has_orientation` is true.
geometry_msgs/Pose pose
bool has_orientation

# Map frame, m/s. Smoothed by the tracker, not a raw frame-to-frame
# difference -- at close range, depth noise alone produces apparent speeds of
# a metre per second on a person standing perfectly still.
geometry_msgs/Vector3 velocity

# ---- Relationship to the robot, recomputed every publish ----
# Derived quantities, but derived once here rather than in each consumer:
# they need the robot's pose at the same instant as the track, and having
# three nodes each look that up separately is three chances to pair a track
# with a stale pose.

float32 distance         # metres, robot base to person
float32 bearing          # radians in the robot frame, 0 = dead ahead, +ve left
float32 range_rate       # m/s along the line of sight; positive = moving away

# 0..1 estimate of how much this person is oriented toward the robot: 1 is
# squarely facing it, 0 is facing directly away. Falls back to 0.5 ("no
# information") when neither keypoints nor motion can say -- deliberately
# neutral, so an unknown orientation neither invites nor forbids approach.
float32 engagement

float32 confidence       # running average detection confidence
uint32 observation_count

builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen

# False while the track is coasting on its predicted velocity because the
# detector has not seen this person for a frame or two. Consumers that must
# not act on a guess (approach goals, in particular) should check this;
# consumers that merely look at someone (gaze) are better off ignoring it,
# since eyes that snap away the instant a detection drops read as twitchy.
bool visible
