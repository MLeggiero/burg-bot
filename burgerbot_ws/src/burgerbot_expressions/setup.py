import os
from glob import glob

from setuptools import find_packages, setup

package_name = "burgerbot_expressions"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mark Leggiero",
    maintainer_email="mark.leggiero1@gmail.com",
    description="Mood arbitration and expressive body gestures for the burgerbot.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mood_arbiter = burgerbot_expressions.mood_arbiter:main",
            "gesture_server = burgerbot_expressions.gesture_server:main",
        ],
    },
)
