#!/usr/bin/bash
echo -e "\e[1;33mCalling script to building package, please waitting... \e[0m"
cd ~/catkin_ws/
catkin_make
echo -e "\e[1;33Complete! \e[0m"
