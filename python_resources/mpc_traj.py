import numpy as np
from hal.products.mats import SDCSRoadMap
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import CubicSpline

def main():

    traj=np.load(r"trajactories/test_spline.npy")
    print(f"the shape of traj si {traj.shape}")


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
    path = roadmap.generate_path(nodeSequence=nodeSequence)*0.957
    path=rotate_waypoints(path,-9)
    x,y=path[0,:],path[1,:]
    x,y,yaw,speed=generate_mpc_trajectory(x,y)
    



    print(path)
    plt.plot(x,y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("X vs Y")
    plt.grid(True)
    plt.show()




def generate_mpc_trajectory(x_points, y_points, ds=0.1):
    """
    Takes rough waypoints and returns a smooth trajectory (x, y, yaw, k)
    ds: distance between interpolated points [m]
    """
    
    # 1. Calculate distance along the path (s)
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Cumulative distance (s)
    s = np.zeros(len(x_points))
    s[1:] = np.cumsum(distances)
    
    # 2. Create Cubic Spline functions for x(s) and y(s)
    cs_x = CubicSpline(s, x_points)
    cs_y = CubicSpline(s, y_points)
    
    # 3. Interpolate densely
    s_new = np.arange(0, s[-1], ds)
    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    
    # 4. Calculate Yaw (heading) and Curvature (k)
    # First derivatives (dx/ds, dy/ds)
    dx_ds = cs_x(s_new, 1)
    dy_ds = cs_y(s_new, 1)
    
    # Second derivatives (d2x/ds2, d2y/ds2)
    ddx_ds = cs_x(s_new, 2)
    ddy_ds = cs_y(s_new, 2)
    
    # Yaw = arctan2(dy/ds, dx/ds)
    yaw = np.arctan2(dy_ds, dx_ds)
    
    # Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    # Since parameter is arc length s, x'^2 + y'^2 approx 1, simplifying calculation
    k = (dx_ds * ddy_ds - dy_ds * ddx_ds) / ((dx_ds**2 + dy_ds**2)**(1.5))
    
    return x_new, y_new, yaw, k




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



