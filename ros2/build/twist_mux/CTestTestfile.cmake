# CMake generated Testfile for 
# Source directory: /workspaces/isaac_ros-dev/ros2/src/twist_mux
# Build directory: /workspaces/isaac_ros-dev/ros2/build/twist_mux
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_test_joystick_relay.py "/usr/bin/python3.8" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/workspaces/isaac_ros-dev/ros2/build/twist_mux/test_results/twist_mux/test_test_joystick_relay.py.xunit.xml" "--package-name" "twist_mux" "--output-file" "/workspaces/isaac_ros-dev/ros2/build/twist_mux/launch_test/test_test_joystick_relay.py.txt" "--command" "/usr/bin/python3" "-m" "launch_testing.launch_test" "/workspaces/isaac_ros-dev/ros2/src/twist_mux/test/test_joystick_relay.py" "--junit-xml=/workspaces/isaac_ros-dev/ros2/build/twist_mux/test_results/twist_mux/test_test_joystick_relay.py.xunit.xml" "--package-name=twist_mux")
set_tests_properties(test_test_joystick_relay.py PROPERTIES  LABELS "launch_test" TIMEOUT "60" WORKING_DIRECTORY "/workspaces/isaac_ros-dev/ros2/build/twist_mux" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/launch_testing_ament_cmake/cmake/add_launch_test.cmake;131;ament_add_test;/workspaces/isaac_ros-dev/ros2/src/twist_mux/CMakeLists.txt;67;add_launch_test;/workspaces/isaac_ros-dev/ros2/src/twist_mux/CMakeLists.txt;0;")
