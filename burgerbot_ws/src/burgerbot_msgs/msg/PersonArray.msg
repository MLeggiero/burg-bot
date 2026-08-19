# Every person currently tracked, including those coasting out of sight.
#
# Published on a fixed timer rather than on each detection, so consumers see a
# steady rate whether the detector is running at 1.5 Hz on the Pi or 20 Hz on
# an offloaded GPU. An empty array is meaningful and is published as such: it
# says "nobody is here", which is exactly what the companion behaviour needs
# to hear before it decides to go looking.

std_msgs/Header header
Person[] people
