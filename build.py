import os
import platform
import argparse

# CONFIG
package_name = 'xplainable'

# Load arguments
parser = argparse.ArgumentParser()

choices=['true', 'false']

parser.add_argument(
    "--install",
    type=str,
    default='false',
    choices=choices,
    help="Installs the whl file if true."
    )

parser.add_argument(
    "--force",
    type=str,
    default='false',
    choices=choices,
    help="Forces a reinstall of the whl file and dependencies if true."
    )

args = parser.parse_args()
install = args.install
force = args.force

# Load the latest version number
exec(open(f'{package_name}/_version.py').read())

# Build Wheel
os.system('python setup.py build_ext --inplace')
os.system('python setup.py bdist_wheel')

# Install Wheel
if install == 'true':

    if force == "true":
        os.system(f'pip uninstall {package_name}')

    # Find the current whl version filepath
    whl_prefix = f'{package_name}-{__version__}'
    target_whl = ''
    for file in os.listdir("dist"):
        if file.startswith(whl_prefix):
            target_whl = file

    # Normal install
    os.system(f'cd dist & pip install dist/{target_whl}')

# Remove .egg-info file
if platform.system() == 'Windows':
    os.system(f'rmdir /s /q build {package_name}.egg-info')

else:
    os.system(f'rm -r build {package_name}.egg-info')
