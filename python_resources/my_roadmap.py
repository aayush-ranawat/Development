import numpy as np
from hal.products.mats import SDCSRoadMap
import matplotlib.pyplot as plt
import os
import math


def main():
    # Configuration parameters for example script.
    useSmallMap = False
    leftHandTraffic = False
    nodeSequence = [10,2,4,6,8,10]
    # Create a SDCSRoadMap instance with desired configuration.
    roadmap = SDCSRoadMap(
        leftHandTraffic=leftHandTraffic,
            useSmallMap=useSmallMap
    )
  # Generate the shortest path passing through the given sequence of nodes.
    # - nodeSequence can be a list or tuple of node indicies.
    # - The generated path takes the form of a 2xn numpy array
    path = roadmap.generate_path(nodeSequence=nodeSequence)*0.93
    path=rotate_waypoints(path,-9)
    x,y=path[0,:],path[1,:]
    
    yaw,speed=calculate_path_attributes(x,y)



    print(path)
    plt.plot(x,y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("X vs Y")
    plt.grid(True)
    plt.show()

def calculate_path_attributes(cx, cy, target_speed=1.0, max_lat_accel=0.4):
    """
    Calculates heading (yaw) and velocity (speed) based on x, y coordinates.
    
    Args:
        cx: List of x coordinates
        cy: List of y coordinates
        target_speed: Desired max speed (m/s)
        max_lat_accel: Maximum lateral acceleration allowed (determines cornering speed)
        
    Returns:
        cyaw: List of heading angles (radians)
        sp: List of target velocities (m/s)
    """
    
    # Ensure inputs are numpy arrays
    # cx = np.array(cx)
    # cy = np.array(cy)
    
    # --- 1. Calculate Heading (cyaw) ---
    # We use np.gradient to get the direction of the path at every point
    # dx/dt and dy/dt (assuming constant step size for geometry)
    dx = np.gradient(cx)
    dy = np.gradient(cy)
    
    # Compute yaw using arctan2 (handles all quadrants correctly)
    cyaw = np.arctan2(dy, dx)
    
    # Unwrap ensures the angle is continuous (e.g., doesn't jump from 3.14 to -3.14)
    # This helps controllers avoid sudden steering jerks
    cyaw = np.unwrap(cyaw)

    # --- 2. Calculate Velocity Profile (sp) ---
    # To find the right speed, we first need the curvature (kappa).
    # Formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Calculate curvature
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Handle NaN or Inf resulting from straight lines (zero division)
    curvature = np.nan_to_num(curvature)
    
    # Calculate Max Speed allowed based on curvature
    # v_max = sqrt(a_lat_max / curvature)
    # We add a small epsilon (1e-6) to curvature to avoid division by zero
    sp = np.sqrt(max_lat_accel / (curvature + 1e-6))
    
    # Clip the speed to not exceed the target_speed (the car's physical limit)
    sp = np.minimum(sp, target_speed)

    # --- 3. Terminal Velocity Handling ---
    # The robot should slow down as it approaches the end of the path
    # We linearly ramp down speed for the last 5 points
    if len(sp) > 5:
        for i in range(5):
            sp[-(i+1)] = sp[-(i+1)] * (i / 5.0)

    return cyaw.tolist(), sp.tolist()


def rotate_waypoints(xy, theta):
        """
        xy: (N,2) array of [x,y]
        theta: rotation angle in degree
        """
        theta=np.deg2rad(theta)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        return R @ xy

if __name__=="__main__":
    main()



