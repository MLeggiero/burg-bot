# Everyone found in one camera frame.
#
# The header stamp is the stamp of the *image inference ran on*, not the time
# the message was published. person_tracker pairs this against the depth frame
# of the same instant, and with detection offloaded to a separate machine the
# gap between those two times can be a hundred milliseconds or more -- long
# enough that a walking person has moved several centimetres and a turning
# robot has moved them much further than that.

std_msgs/Header header
PersonDetection2D[] detections
