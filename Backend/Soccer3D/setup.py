from setuptools import setup, find_packages

setup(
    name="soccer3d",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "tritonclient[all]",
        "supervision",
        "mediapipe",
        "torch",
        "torchvision",
    ],
    entry_points={
        "console_scripts": [
            "soccer3d=soccer3d.scripts.run_soccer3d:main",
        ],
    },
    author="SCAI Team",
    author_email="yazeed.alrubyli@gmail.com",
    description="Soccer player and ball detection and 3D pose estimation",
    keywords="soccer, 3d, pose, detection, tracking",
    python_requires=">=3.7",
)
