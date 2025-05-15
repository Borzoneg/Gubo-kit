sudo apt install terminator
sudo apt install git
# ssh-keygen
read -p "Press enter when you activate the ssh key in github and gitlab" 
git clone git@gitlab.sdu.dk:gubo/fluently_mem.git
# git clone git@github.com:Borzoneg/Gubo-kit.git
git clone git@github.com:Borzoneg/knowledge-transfer-fleuntly.git
echo "source $HOME/Gubo-kit/.bashgu" >> $HOME/.bashrc

sudo apt install python3.10-venv
python3 -m venv pyenv

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install software-properties-common                         
sudo add-apt-repository universe        
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop python3-colcon-common-extensions # ros-dev-tools
source /opt/ros/humble/setup.bash

cd $HOME/Gubo-kit/ros_ws
colcon build

sudo apt install ros-humble-joint_state_publisher_gui
sudo apt install ros-humble-rviz2
sudo apt install ros-humble-xacro
sudo apt install libogre-1.12-dev
pip install colcon-common-extensions catkin_pkg empy lark-parser rosdep rosdistro pyrealsense2 roboticstoolbox-python pytrees numpy scipy matplotlib open3d opencv-python ultralytics ur_rtde
sudo apt-get install python3-tk

source $HOME/Gubokit/.bashgu
