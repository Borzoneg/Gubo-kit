from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import launch_ros.descriptions
import launch

# use from terminal: ros2 launch your_package your_launch_file.launch.py urdf_path:=/path/to/your.urdf rviz_config_path:=/path/to/config.rviz

def generate_launch_description():
    # declare argument and get value from it
    urdf_arg = DeclareLaunchArgument("urdf_path", default_value="/home/gu/Gubo-kit/urdf/pad_cell.urdf")
    rviz_arg = DeclareLaunchArgument("rviz_config_path", default_value="/home/gu/Gubo-kit/files/rviz_config/config.rviz")
    urdf_path = LaunchConfiguration("urdf_path")
    rviz_config_path = LaunchConfiguration("rviz_config_path")
    
    # execute the xacro/urdf and convert it to string
    robot_params={'robot_description': launch_ros.descriptions.ParameterValue(launch.substitutions.Command(['xacro ', urdf_path]), value_type=str)}

    # Nodes
    joint_state_publisher_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_params],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
    )

    return LaunchDescription([
        urdf_arg,
        rviz_arg,
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
    ])
