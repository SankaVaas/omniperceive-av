from setuptools import setup, find_packages

setup(
    name="omniperceive",
    version="0.1.0",
    description="End-to-End Multi-Task AV Perception: Detection, Lane, Depth, Segmentation",
    author="Your Name",
    packages=find_packages(exclude=["tests", "tools", "notebooks"]),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
