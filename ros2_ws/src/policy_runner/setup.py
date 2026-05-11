from setuptools import find_packages, setup

package_name = 'policy_runner'

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
    maintainer='Emin Çağan Apaydın',
    maintainer_email='senin-emailin@example.com',
    description='ROS2 node for running trained IL/RL policy on Franka Panda via ONNX/TensorRT inference',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_inference_node = policy_runner.policy_inference_node:main'
        ],
    },
)
