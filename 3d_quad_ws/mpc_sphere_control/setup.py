import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mpc_sphere_control'

setup(
    name=package_name,
    version='0.0.0',
    # find_packages is safer as it dynamically resolves nested modules like your simulation class
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        
        # EXPORT RVIZ CONFIGS: Allows your launch files to discover your saved display layouts
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),

        # Mesh assets for MESH_RESOURCE markers (render_node.ENTITY_STYLES)
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*'))),
    ],
    install_requires=['setuptools', 'scipy', 'numpy', 'osqp', 'casadi'],
    zip_safe=True,
    maintainer='Gesem Mejia, Ananya Acharya',
    maintainer_email='gesemgudino@gmail.com, ananya.ds.act@gmail.com',
    description='MPC and CBF for 3D navigation',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'render_node = mpc_sphere_control.render_node:main',
            'pursuit_evasion_node = mpc_sphere_control.pursuit_evasion_node:main',
        ],
    },
)
