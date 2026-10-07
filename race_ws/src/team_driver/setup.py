import os
from glob import glob

from setuptools import setup

package_name = 'team_driver'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Team',
    maintainer_email='you@e.ntu.edu.sg',
    description='Hackathon entry.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver = team_driver.driver:main',
            'map_viewer = team_driver.map_viewer:main',
        ],
    },
)
