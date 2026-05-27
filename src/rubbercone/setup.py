from setuptools import setup
import os
from glob import glob

package_name = 'rubbercone'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Rubbercone driving package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'orange_detection = rubbercone.orange_detection:main',
            'drive_pivot = rubbercone.drive_pivot:main',
            'planner = rubbercone.planner:main',
        ],
    },
)
