import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mab_dial_mpc_locomotion_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='piotrkicki',
    maintainer_email='piotr.m.kicki@gmail.com',
    description='Dial-MPC Locomotion controller for the MAB Honey Badger and MAB Silver Badger robots',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sb_dial_mpc_locomotion_controller = mab_dial_mpc_locomotion_controller.sb_dial_mpc_locomotion_controller_node:main'
        ],
    },
)
