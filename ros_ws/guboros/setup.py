from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'guboros'

data_files=[(os.path.join('share', package_name), glob('launch/*.py')), # added for launch files
            (os.path.join('share', package_name, 'config'), glob('config/*')), # added for config files
            (os.path.join('share', package_name, 'urdf', 'meshes', 'crx20ia_l', 'visual'), glob('urdf/meshes/crx20ia_l/visual/*.stl')),
            (os.path.join('share', package_name, 'urdf', 'meshes', 'crx20ia_l', 'collision'), glob('urdf/meshes/crx20ia_l/collision/*.stl')),
            (os.path.join('share', package_name, 'urdf', 'meshes', 'others', 'collision'), glob('urdf/meshes/others/collision/*.stl')),
            (os.path.join('share', package_name, 'urdf', 'meshes', 'others', 'visual'), glob('urdf/meshes/others/collision/*.stl')),
            ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),]

package_data={package_name: ['urdf/meshes/*/*/*.stl']}

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=data_files,
    package_data=package_data,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gu',
    maintainer_email='borzoneg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
