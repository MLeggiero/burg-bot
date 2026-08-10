# A persistent, de-duplicated object in the semantic map layer.
#
# Raw per-frame detections (vision_msgs/Detection3D) are noisy and repeat --
# the robot walks past the same chair fifty times and produces fifty
# detections. One of these represents many of those folded into a single
# tracked object by burgerbot_perception's semantic_map node: same class,
# nearby position, clustered and confidence-weighted into one running
# estimate rather than fifty separate markers cluttering the map.

string id                        # stable id, e.g. "chair_3"
string label                     # class name, e.g. "chair"
geometry_msgs/Pose pose          # map-frame position; running weighted average
float32 confidence               # running average detection confidence
uint32 observation_count         # how many raw detections fed this estimate

builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
