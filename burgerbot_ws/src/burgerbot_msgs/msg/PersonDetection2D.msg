# One person found in a single camera frame, before any tracking.
#
# vision_msgs/Detection2D would carry the bounding box perfectly well, but it
# has nowhere to put keypoints, and keypoints are what make the difference
# between "a person is 2m away" and "a person 2m away is facing me". Distance
# alone is a poor predictor of whether someone wants the robot's company --
# somebody walking past at 1.5m wants nothing, somebody standing at 2.5m
# looking straight at it does. Body orientation is what separates those, and
# it comes almost free from a pose model.
#
# Produced by either detector: person_detector_gpu fills keypoints in,
# person_detector_lite (which re-labels the Pi's existing bbox-only
# detections) leaves the array empty. Everything downstream treats missing
# keypoints as "orientation unknown" rather than as an error, so the same
# social stack runs on both.

# ---- COCO 17-keypoint indices, for indexing `keypoints` ----
uint8 KP_NOSE=0
uint8 KP_LEFT_EYE=1
uint8 KP_RIGHT_EYE=2
uint8 KP_LEFT_EAR=3
uint8 KP_RIGHT_EAR=4
uint8 KP_LEFT_SHOULDER=5
uint8 KP_RIGHT_SHOULDER=6
uint8 KP_LEFT_ELBOW=7
uint8 KP_RIGHT_ELBOW=8
uint8 KP_LEFT_WRIST=9
uint8 KP_RIGHT_WRIST=10
uint8 KP_LEFT_HIP=11
uint8 KP_RIGHT_HIP=12
uint8 KP_LEFT_KNEE=13
uint8 KP_RIGHT_KNEE=14
uint8 KP_LEFT_ANKLE=15
uint8 KP_RIGHT_ANKLE=16
uint8 NUM_KEYPOINTS=17

# Bounding box in the source image's pixel coordinates, not the model's input
# resolution -- the detector rescales before publishing, because the depth
# image this gets paired with is at the camera's native size.
float32 center_x
float32 center_y
float32 size_x
float32 size_y

float32 score

# Flat triples (x_pixels, y_pixels, confidence) in KP_* order, so length is
# either 0 (bbox-only detector) or 3 * NUM_KEYPOINTS. Flat rather than an
# array of a Keypoint message because it is read as a block by numpy on the
# consuming side, and a nested array costs a deserialization allocation per
# keypoint per person per frame for no gain.
float32[] keypoints
