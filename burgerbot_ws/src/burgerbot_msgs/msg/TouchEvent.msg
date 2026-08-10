# A touch on the face panel.
#
# The renderer knows where it drew each feature this frame, so it can report
# what was actually poked rather than just a coordinate.

uint8 TYPE_DOWN=0
uint8 TYPE_UP=1
uint8 TYPE_MOVE=2

uint8 REGION_OTHER=0
uint8 REGION_LEFT_EYE=1
uint8 REGION_RIGHT_EYE=2

std_msgs/Header header

uint8 type
uint8 region

# Normalized screen coordinates, x in -1..1 (left to right), y in -1..1
# (bottom to top) -- the same space the face is authored in.
float32 x
float32 y
