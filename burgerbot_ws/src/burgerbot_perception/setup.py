import os
from glob import glob

from setuptools import find_packages, setup

package_name = "burgerbot_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.txt")),
        # Populated by scripts/export_detection_model.sh, not checked into
        # git (a binary model file, same reasoning as the Gazebo world
        # assets in burgerbot_description). Empty at a fresh checkout --
        # colcon is fine installing nothing for an empty glob. Run the
        # export script, then re-run colcon build once to pick up the new
        # file (--symlink-install only needs this once; further edits to
        # the same file don't need another rebuild).
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mark Leggiero",
    maintainer_email="mark.leggiero1@gmail.com",
    description="Camera-based object detection and semantic mapping for the burgerbot.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "object_detector = burgerbot_perception.object_detector:main",
            "object_projector = burgerbot_perception.object_projector:main",
            "semantic_map = burgerbot_perception.semantic_map:main",
        ],
    },
)
