- ros2 pkg create --build-type ament_python finger_pkg
- cd finger_pkg
- make this folder structure:

        finger_pkg/
        ├── launch/
        │   └── finger_launch.py
        ├── urdf/
        │   ├── finger.urdf
        │   └── meshes/
        │       ├── collision/
        │       │   ├── collision1.stl
        │       │   └── collision2.stl
        │       └── visual/
        │           ├── visual1.stl
        │           └── visual2.stl
        ├── finger_pkg/
        │   ├── __init__.py
        │   └── finger_node.py
        ├── resource/
        │   └── finger_pkg
        ├── package.xml
        ├── setup.cfg
        ├── setup.py
        └── README.md

Some things are already there
#### Setup.py:
- Add this:

        import os
        from glob import glob

        data_files=[(os.path.join('share', package_name), glob('launch/*.py')), # added for launch files
                    (os.path.join('share', package_name, 'config'), glob('config/*')), # added for config files
                    # left part is where the file is right is where we install it
                    (os.path.join('share', package_name, 'urdf', 'meshes', 'finger', 'visual'), glob('urdf/meshes/finger/visual/*.stl')), 
                    (os.path.join('share', package_name, 'urdf', 'meshes', 'finger', 'collision'), glob('urdf/meshes/finger/collision/*.stl')),
                    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                    ('share/' + package_name, ['package.xml']),]

        package_data={package_name: ['urdf/meshes/*/*/*.stl']}

- Use them in here: 

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

- colcon build (it will create 3 folders so maybe you want to put your package inside a ros_ws folder and then build inside there)