from setuptools import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="AE-ROM-Training",
    version=0.1,
    author="Christopher R. Wentland",
    author_email="chriswen@umich.edu",
    url="https://github.com/cwentland0/ae_rom_training",
    description="Framework for training autoencoder-based ROMs",
    long_description=readme,
    license=license,
    # install_requires=["numpy>=1.16.6", "scipy>=1.1.0", "matplotlib>=2.1.0"],
    entry_points={"console_scripts": ["ae_rom_training = ae_rom_training.driver:main"]},
    python_requires=">=3.6",
)
