import os
from glob import glob

from setuptools import find_packages, setup

package_name = "burgerbot_face"

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
    description="Procedural animated face for the burgerbot DSI panel.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "face_node = burgerbot_face.face_node:main",
            "demo_expressions = burgerbot_face.demo_expressions:main",
        ],
    },
)
