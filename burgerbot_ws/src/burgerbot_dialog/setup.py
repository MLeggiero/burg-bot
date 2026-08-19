import os
from glob import glob

from setuptools import find_packages, setup

package_name = "burgerbot_dialog"

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
    description="Local-LLM conversation layer for the burgerbot.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dialog_manager = burgerbot_dialog.dialog_manager:main",
            "dialog_cli = burgerbot_dialog.dialog_cli:main",
        ],
    },
)
