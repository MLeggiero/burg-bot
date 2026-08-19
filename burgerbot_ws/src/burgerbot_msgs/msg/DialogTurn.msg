# One exchange: what was said to the robot, and what it actually did about it.
#
# Same introspection role FaceState and SocialState play. It matters more here
# than for either of those, because a language model's output is the one part
# of this robot that is not reproducible -- you cannot re-run a conversation
# and get the same reply, so if the robot says or does something odd, this
# message is the only record of what it was asked and what came back.
#
# In particular `error` and `latency` are what distinguish the three failure
# modes that look identical from outside: the model said something strange, the
# model was unreachable, or the model was simply slow and the turn timed out.

std_msgs/Header header

# Who the robot believes it was talking to, from the companion's current
# target. Empty for a stranger, or when nobody is being tracked at all.
string speaker

# What was said to it, and what it replied.
string heard
string said

# What it asked the rest of the stack to do. Empty means it asked for nothing,
# which is the common case -- most turns are just talking.
string expression
string gesture

# The place name the model asked for, and whether that name resolved to
# somewhere real. An unresolved goal is not an error: "go to the kitchen" is a
# reasonable thing to say to a robot that has never been told where the kitchen
# is, and the reply should say so rather than the turn failing.
string goal
bool goal_resolved

# Seconds from sending the request to a validated reply. Worth watching: a
# local model that has started swapping goes from ~1s to ~30s with no other
# symptom.
float32 latency

# Empty on success. Set when the backend was unreachable, timed out, or
# returned something that could not be salvaged into a reply.
string error
