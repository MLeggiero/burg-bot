# A snapshot of the semantic map layer: every currently-tracked object.
#
# Mirrors how the geometric map is handled -- nav2_map_server persists
# map.yaml/map.pgm alongside a SLAM session; burgerbot_perception's
# semantic_map node persists this alongside it as objects.yaml, so a saved
# map and its object labels travel together.

std_msgs/Header header
SemanticObject[] objects
