#! /usr/bin/env python3

# # Quanser specific packages
# from pit.YOLO.nets import YOLOv8
# from pit.YOLO.utils import QCar2DepthAligned


from ultralytics import YOLO
model=YOLO("best2.pt")

# Generic python packages
import time  # Time library
import numpy as np
import cv2

# ROS specific packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
'''
Description:

Node for detecting traffic light state and signs on the road. Provides flags
which define if a traffic signal has been detected and what action to take.
'''

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        # Additional parameters
        imageWidth  = 640
        imageHeight = 480

        self.stop_sign=0
        self.stop_sign_time=-100
        self.stop_buffer=0
        self.current_pose=np.array([0,0])
        self.drop_off=np.array([-0.910,0.800])
        self.pick_up=np.array([0.125,4.395])  
        self.pick_up_f,self.drop_off_f=False,False
        self.traffic_stop=0
        self.taxi_hub=np.array([-1.744,-0.198])
        self.qcar_state =1.0



        self.qcar_hardware_node= "qcar2_hardware"

        self.qcar_hardware_client = self.create_client(SetParameters, f'/{self.qcar_hardware_node}/set_parameters')

        while not self.qcar_hardware_client.wait_for_service(timeout_sec = 4.0):
            self.get_logger().info(f'waiting for {self.qcar_hardware_node} parameter service!.....')
        
        self.get_logger().info(f'connected to  {self.qcar_hardware_node} parameter service!.....')

        self.led_set_logic(qcar_state=5.0)





        self.model=YOLO("best2.pt")

        self.brake_pub = self.create_publisher(Twist, 'cmd_vel_brake', 10)

        self.stop_msg=Bool()

        # 1. Subscribe to the RAW topic for lowest latency
        self.subscription = self.create_subscription(
            Image,
            '/camera/csi_image',
            self.image_callback,
            10) # QoS profile (depth of 10)
        
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.pose_callback,
            10
        )
        
        self.bridge= CvBridge()
        self.led_set_logic(qcar_state=4.0)

    
    def pose_callback(self,msg):
         # 1. Position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            # 2. Orientation (Quaternion -> Euler)
            # q = msg.pose.pose.orientation
            # quat_list = [q.x, q.y, q.z, q.w]
            # (roll, pitch, yaw) = self.euler_from_quaternion(quat_list)

            self.current_pose=np.array([x,y])

            


    def image_callback(self, msg):
            try:

                print(f"traffic stop sign is {self.traffic_stop}")

                


                
                if self.stop_sign==1 or self.traffic_stop==1:
                    stop_msg = Twist()
        
                    # Explicitly set everything to 0.0
                    stop_msg.linear.x = 0.0
                    stop_msg.linear.y = 0.0
                    stop_msg.linear.z = 0.0
                    stop_msg.angular.x = 0.0
                    stop_msg.angular.y = 0.0
                    stop_msg.angular.z = 0.0
        
                    # Publish to the high-priority topic
                    self.brake_pub.publish(stop_msg)



    
                     

                if self.stop_sign==1 and abs(self.stop_sign_time-time.time()) < 3.0:
                     return
                
                elif self.stop_sign==1 and abs(self.stop_sign_time-time.time())>3  :
                     self.stop_buffer=1
                     self.stop_sign=0
                     self.led_set_logic(qcar_state=4.0)


                if abs(self.stop_sign_time-time.time())>7 :
                     self.stop_buffer=0


                pick_up_dist=np.linalg.norm(self.current_pose-self.pick_up)
                drop_off_dist=np.linalg.norm(self.current_pose-self.drop_off)

                taxi_hub_dist=np.linalg.norm(self.current_pose-self.taxi_hub)
                print(f"taxihub distance is {taxi_hub_dist}")


                if pick_up_dist < 0.15 and self.stop_sign==0 and self.stop_buffer==0:
                     if self.pick_up_f is False:
                          
                        self.stop_sign=1
                        self.stop_sign_time=time.time()
                        self.led_set_logic(qcar_state=2.0)
                    
                     self.pick_up_f=True

                if drop_off_dist< 0.25 and self.stop_sign==0 and self.stop_buffer==0:
                     if self.drop_off_f is False:
                        self.stop_sign=1
                        self.stop_sign_time=time.time()
                        self.led_set_logic(qcar_state=3.0)

                     self.drop_off_f=True

                
                if taxi_hub_dist < 0.20 :
                     
                          
                    self.stop_sign=1
                    self.led_set_logic(qcar_state=5.0)
                
                     

                        
                
                

                     
                # 2. Convert ROS Image message to OpenCV format (BGR)
                # This happens in RAM—no network compression/decompression needed
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # 3. YOLO Processing would happen here
                results = self.model(cv_image, verbose=False)

                for result in results:
                            boxes = result.boxes  # Boxes object containing all detections
                            res_array=[]
                            for box in boxes:
                                cls = int(box.cls[0]) 
                                res_array.append(cls)
                                x1, y1, x2, y2 = map(int, box.xyxy[0]) #top left and bottom right image coordinates
                                w,h=abs(x1-x2),abs(y1-y2)
                                

                                label = model.names[cls] 
                                res_array+=[label]




                                if (cls==5) and (self.stop_sign==0) and (self.stop_buffer==0):
                                    # x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                                   
                                    
                                 
                                    if abs(x1-x2)>20: 
                                        self.stop_sign=1
                                        self.led_set_logic(qcar_state=1.0)
                                        self.stop_sign_time=time.time()

                            

                #                 # changing the trafficlight flag
                                if cls==3 and  180<x1<250:  #red light 
                                    # self.get_logger().info(f"red light detected width id {h} at {x1 , x2 , y1 , y2}")                                         
                                    
                                    if  h>10:
                                        self.traffic_stop=1
                                        self.led_set_logic(qcar_state=1.0)

                                elif cls==6 and 180<x1<250 and h>10:  # yellow light

                                    # self.get_logger().info(f"yellow light detected width is {h} at {x1 , x2 , y1 , y2}") 
                                   
                                    if  h>10:
                                        self.traffic_stop=1
                                        self.led_set_logic(qcar_state=1.0)
                                       


                                elif cls==1 and 180<x1<250 and h>10:   # green light

                                    # self.get_logger().info(f"green light detected width is {h} at {x1 , x2 , y1 , y2}") 
                                   
                                    if  h>10 :
                                       self. traffic_stop=0
                                       self.led_set_logic(qcar_state=4.0)


                
                
                # # 4. Optional: Show the frame (disable this on the car to save CPU)
                cv2.imshow("YOLO Input Stream", cv_image)
                cv2.waitKey(1)
                
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')

    def led_set_logic(self,qcar_state):

        # This section is trying to emulate the cli call -> ros2 param set qcar2_hardware led_color_id <value>
        # LED Red (used for taxi hub stop and inbetween ride stops)

        if qcar_state == 1.0:
            self.send_request(param_name="led_color_id",
                              param_value= 0,
                              param_type= ParameterType.PARAMETER_INTEGER,
                              client= self.qcar_hardware_client)
        
        # Arrived at pickup
        elif qcar_state == 2.0:
            self.send_request(param_name="led_color_id",
                              param_value= 2,
                              param_type= ParameterType.PARAMETER_INTEGER,
                              client= self.qcar_hardware_client)
        
        # Drop off coordinate
        elif qcar_state == 3.0:
            self.send_request(param_name="led_color_id",
                              param_value= 3,
                              param_type= ParameterType.PARAMETER_INTEGER,
                              client= self.qcar_hardware_client)
        # Driving state
        elif qcar_state == 4.0:
            self.send_request(param_name="led_color_id",
                              param_value= 1,
                              param_type= ParameterType.PARAMETER_INTEGER,
                              client= self.qcar_hardware_client)
            
        elif qcar_state == 5.0:
            self.send_request(param_name="led_color_id",
                              param_value= 5,
                              param_type= ParameterType.PARAMETER_INTEGER,
                              client= self.qcar_hardware_client)


    # method used to end multiple parameters to multiple nodes
    def send_request(self, param_name, param_value, param_type,client):
            

        param = Parameter()
        param.name = param_name
        param.value.type = param_type

        if param_type == ParameterType.PARAMETER_INTEGER_ARRAY:
            param.value.integer_array_value = param_value

        elif param_type == ParameterType.PARAMETER_INTEGER:
            param.value.integer_value = param_value

        elif param_type == ParameterType.PARAMETER_BOOL_ARRAY:
            param.value.bool_array_value = param_value
        
        elif param_type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            param.value.double_array_value = param_value


        request = SetParameters.Request()
        request.parameters = [param]
        future = client.call_async(request)




        
       
def main():

  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = ObjectDetector()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      node.destroy_node()
      rclpy.shutdown()
    
      pass

  rclpy.shutdown()

if __name__ == '__main__':
  main()