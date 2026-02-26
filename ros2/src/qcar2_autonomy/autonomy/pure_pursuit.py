#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Header

import numpy as np
from hal.products.mats import SDCSRoadMap
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as R # Useful for Quaternion -> Euler
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from qcar2_interfaces.msg import MotorCommands
from scipy.spatial.transform import Rotation as Rot


class pure_pursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        self.max_reacquire = 20.
        self.wheelbase=0.257
        self.index=0



        self.motor_cmd_pub=self.create_publisher(Twist,'/nav_vel', 1)



        # ---- CONFIG ----
        useSmallMap = False
        leftHandTraffic = False

        nodeSequence = [10, 2, 4, 14 , 20 , 22 , 9 , 7 , 14 , 20 , 22 , 10]

        roadmap = SDCSRoadMap(
            leftHandTraffic=leftHandTraffic,
            useSmallMap=useSmallMap
        )
        

        # Path shape: 2 x N  (x, y)
        self.path_np = roadmap.generate_path(nodeSequence=nodeSequence)[:2, :] * 0.96      #0.975    #scaling the path

        self.path_np = self.rotate_waypoints(self.path_np,-8)                            # rotate by 9 degree

        # self.timer=self.create_timer(0.08,self.pp_controller)
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.pp_controller,
            10
        )
        self.get_logger().info("Published XY path to path_viz")

    


    def pp_controller(self,msg:Odometry):
     

        # 1. Position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            # 2. Orientation (Quaternion -> Euler)
            q = msg.pose.pose.orientation
            quat_list = [q.x, q.y, q.z, q.w]
            (roll, pitch, yaw) = self.euler_from_quaternion(quat_list)

            point_current=np.array([x,y])



            # nearest_p, nearest_dist, t, index = self.nearest_point(point=point_current, trajectory=(self.path_np).T)
            pose_lookahead=self._get_current_waypoint(lookahead_distance=1.0,position=point_current,theta=yaw)

            speed,steering=self.get_actuation(pose_theta=yaw,   lookahead_point=pose_lookahead,   position=point_current,   lookahead_distance=1.0,   wheelbase=self.wheelbase)
          
            QCarCommands = Twist()
            QCarCommands.linear.x=float(speed)
            QCarCommands.angular.z=float(steering)
            self.motor_cmd_pub.publish(QCarCommands)

            

      

####---------------------------------------------utility-------------------------------------------------------------------------####


    def euler_from_quaternion(self,q):
            # q = [x, y, z, w]
            r = Rot.from_quat(q)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            return roll, pitch, yaw


    def rotate_waypoints(self,xy, theta):
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
    


    def get_actuation(self, pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
        speed = 0.5
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/lookahead_distance**2)
        steering_angle = np.arctan(wheelbase/radius)
        return speed, steering_angle



    def nearest_point(self,point, trajectory):
        """
        Return the nearest point along the given piecewise linear trajectory.

        Args:
            point (numpy.ndarray, (2, )): (x, y) of current pose
            trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
                NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

        Returns:
            nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
            nearest_dist (float): distance to the nearest point
            t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
            i (int): index of nearest point in the array of trajectory waypoints
        """
        diffs = trajectory[1:,:] - trajectory[:-1,:]
        l2s   = diffs[:,0]**2 + diffs[:,1]**2
        dots = np.empty((trajectory.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
        projections = trajectory[:-1,:] + (t*diffs.T).T
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp*temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment
    
    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)
            theta (float): current vehicle heading

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """

        nearest_p, nearest_dist, t, i = self.nearest_point(position, self.path_np.T[max(0,self.index-10):self.index+20,:])

        i=self.index+i
     
        self.index=i
       
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = self.intersect_point(position,
                                                      lookahead_distance,
                                                      self.path_np.T,
                                                      i + t,
                                                      wrap=True)
            
            if i2 is None:
                return None
            current_waypoint = np.array([self.path_np.T[i2, 0], self.path_np.T[i2, 1], 5])
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.path_np.T[i, :]
        else:
            return None
        
    def intersect_point(self,point, radius, trajectory, t=0.0, wrap=False):
        """
        starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

        Assumes that the first segment passes within a single radius of the point

        http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
        """
        start_i = int(t)
        start_t = t % 1.0
        first_t = None
        first_i = None
        first_p = None
        trajectory = np.ascontiguousarray(trajectory)
        for i in range(start_i, trajectory.shape[0]-1):
            start = trajectory[i,:]
            end = trajectory[i+1,:]+1e-6
            V = np.ascontiguousarray(end - start)

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            #   print "NO INTERSECTION"
            # else:
            # if discriminant >= 0.0:
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if i == start_i:
                if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                    first_t = t1
                    first_i = i
                    first_p = start + t1 * V
                    break
                if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                    first_t = t2
                    first_i = i
                    first_p = start + t2 * V
                    break
            elif t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        # wrap around to the beginning of the trajectory if no intersection is found1
        if wrap and first_p is None:
            for i in range(-1, start_i):
                start = trajectory[i % trajectory.shape[0],:]
                end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
                V = end - start

                a = np.dot(V,V)
                b = 2.0*np.dot(V, start - point)
                c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
                discriminant = b*b-4*a*c

                if discriminant < 0:
                    continue
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2.0*a)
                t2 = (-b + discriminant) / (2.0*a)
                if t1 >= 0.0 and t1 <= 1.0:
                    first_t = t1
                    first_i = i
                    first_p = start + t1 * V
                    break
                elif t2 >= 0.0 and t2 <= 1.0:
                    first_t = t2
                    first_i = i
                    first_p = start + t2 * V
                    break

        return first_p, first_i, first_t
    

####--------------------------------------------------------------utility--------------------------------------------------------------####


def main(args=None):
    rclpy.init(args=args)
    node=pure_pursuit()
    try:
        rclpy.spin(node)
    except SystemExit:                 # <--- Catch the signal here
        rclpy.logging.get_logger("Quitter").info('Done')
    
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__=="__main__":
    main()