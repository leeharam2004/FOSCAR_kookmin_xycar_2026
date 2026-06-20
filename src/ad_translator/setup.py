from setuptools import find_packages, setup

package_name = 'ad_translator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='valueof@kookmin.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tf_translator = ad_translator.tf_translator:main',
            'lidar_translator = ad_translator.lidar_translator:main',
            'motor_translator = ad_translator.motor_translator:main',
            'goal_sender = ad_translator.goal_sender:main'
        ],
    },
)
