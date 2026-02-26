# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
vehicle_control.py

Skills acivity code for vehicle control lab guide.
Students will implement a vehicle speed and steering controller.
Please review Lab Guide - vehicle control PDF
"""
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS,IS_PHYSICAL_QCAR,QCarCameras
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from pal.products.qcar import QCarRealSense
from ultralytics import YOLO

model=YOLO("best2.pt")
tf = 100
startDelay = 3
controllerUpdateRate = 300

global t0, count, signalFlag, v_ref, waypointCounter,pick_up
waypointCounter = 0
count = 0
signalFlag= 'Go'
pick_up=0
person_flag=0
traffic_light_direction=[300,620]   # look for traffic light with x between 300 and 620 i.e look straight
traffic_stop=0

v_ref = 0.9
K_p = 1
K_i = 1
K_d = 0.1


enableSteeringControl = True
K_stanley = 2
nodeSequence = [10 , 2 , 4, 14 , 20 , 22 , 9 , 13 , 19 , 17 , 20 , 22 , 10 ] 


#----region : Initial setup-----#

if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    print(waypointSequence)
else:
    initialPose = [0, 0, 0]

calibrate=False

calibrationPose = [0,0,-np.pi/2]

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD, done, nextSlope, previousSlopes
pick_up=0
nextSlope = 0.01
previousSlopes = 0
KILL_THREAD = False
done = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

#-----endregion-----#



    # ==============  SECTION A -  Speed Control  ====================


class SpeedController:                                       #for providing throttle input

    def __init__(self, kp=0, ki=0, kd = 0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ei = 0
        self.prev_error = v_ref

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        if dt ==0.0:
            dt = 0.001
        de_dt = (e - self.prev_error) / dt
        output  = self.kp * e + self.ki * self.ei + self.kd * de_dt
        self.prev_error = e

        return np.clip(
             output,
            -self.maxThrottle,
            self.maxThrottle
        )
    
        
        # ==============  SECTION B -  Steering Control  ====================

class SteeringController:                               

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints                                                
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.slopes = calculate_slope(waypointSequence)                     #slope sequence of WAYPOINTS wrt to world frame
        
        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0
        self.psi_prev = 0
        

    def update(self, p, th, speed, dt,k):
        self.k=k
        global done, nextSlope, previousSlopes, waypointCounter, traffic_light_direction
        if  self.N - self.wpi >2: 
            nextSlope = self.slopes[self.wpi+1]
        else:
            nextSlope = 0
        previousSlopes = self.slopes[:self.wpi]
        if self.wpi>1630:
            done = True

        waypointCounter = self.wpi  
        
        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]

        if self.wpi<3000 and self.wpi+90<self.N:                                    #tmatrix ransformation from world frame to car frame
            wp_3=self.wp[:,self.wpi+90]
            c, s = np.cos(th), np.sin(th)
            R_inv = np.array([[ c,  s],
                            [-s,  c]])
            wp_3_new=R_inv@wp_3
            wp_2_new=R_inv@wp_2

            slope=np.rad2deg(np.arctan((wp_3_new[1]-wp_2_new[1])/(wp_3_new[0]-wp_2_new[0])))
        else:
            slope=0
        
        if abs(slope)<10:
            traffic_light_direction=[300,620] # look for traffic light with x between 300 and 620 (i.e look straight since waypoint heading is staight)
        
        elif slope>15:
            traffic_light_direction=[0,300] # look for traffic light with x between 300 (i.e look left)

        elif slope< -15:
            traffic_light_direction=[620,820] # look for traffic light with x between 300 and 620  (i.e look right)

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)

        self.p_ref = ep
        self.th_ref = tangent

        if dt == 0.0:
            headingError = psi 
        else:
            headingError = psi +  (0.1 * ((psi- self.psi_prev)/dt))
        self.psi_prev = psi
        return np.clip(
            wrap_to_pi(headingError + np.arctan2(self.k*ect,speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
            
#----------------- An utility functions---------------------
def compare_elements(x, my_list, ele):

    if len(my_list) < ele:
        return False 
    
    last_elements = my_list[-ele:]

    return np.array_equal(last_elements, np.full(ele, x))
    
    
def calculate_slope(waypoints):
    slopes = []
 
    num_waypoints = waypoints.shape[1]
 
    # Iterate over the waypoints
    for i in range(num_waypoints - 1):
       
        current_point = waypoints[:,i]
        next_point = waypoints[:,i + 1]
        
        # Calculate the change in x and y
        delta_x = next_point[0] - current_point[0]
        delta_y = next_point[1] - current_point[1]
        
        # Check for vertical line (avoid division by zero)
        if delta_x == 0:
            slope = float('inf')  # Vertical line, slope is infinity
        else:
            slope = delta_y / delta_x
        
        slopes.append(slope)
    
    return slopes

     # ==============  SECTION c - Parallel Parking  ====================

class ParallelParking:
    def __init__(self):
        self.state = 'start'  # FSM: 'start' → 'reverse_right' → 'reverse_left' → 'align' → 'done'
        self.start_time = None
        self.done = False

    def update(self, x, y, theta):
        """Return control commands based on current state."""
        if self.done:
            return 0.0, 0.0  # stop vehicle

        if self.state == 'start':
            self.start_time = time.time()
            self.state = 'reverse_right'
            print("→ Reverse left to enter slot")

        t_elapsed = time.time() - self.start_time


        if self.state == 'reverse_right':
            if t_elapsed < 0.46:
                return -0.17, np.deg2rad(-50)  # reverse with right steer
            else:
                self.state = 'reverse_left'
                self.start_time = time.time()

        elif self.state == 'reverse_left':
            if t_elapsed < 0.38:
                return -0.17, np.deg2rad(50)  # reverse with left steer
            else:
                self.state = 'align'
                self.start_time = time.time()

        elif self.state == 'align':
            if t_elapsed < 0.48:
                return 0.18, np.deg2rad(0) # small forward to align
            else:
                self.state = 'done'
                self.done = True
                print("✓ Parking complete")

        return 0.0, 0.0
    
 # ==============  SECTION D -  Seperate thread for yolo based object detection  ====================
def yolo_detection():
    global signalFlag, myCam,stop_sign,stop_sign_time,traffic_stop,drop_off,person_flag
    global KILL_THREAD
    stop_sign=0
    traffic_stop=0
    drop_off=0
    person_flag=0
    my_flag=0

    with cameras:
        while (not KILL_THREAD):
            try:
                cameras.readAll()
                for i, c in enumerate(cameras.csi):
                    if c is not None:
                        img=c.imageData
                        results=model(img,verbose=False)

                                

                        for result in results:
                            boxes = result.boxes  # Boxes object containing all detections
                            res_array=[]
                            for box in boxes:
                                cls = int(box.cls[0]) 
                                res_array.append(cls)
                                x1, y1, x2, y2 = map(int, box.xyxy[0]) #top left and bottom right image coordinates
                                w,h=abs(x1-x2),abs(y1-y2)
                                

                               
                                if cls==2:                               #cheking for pedestrian detection
                                   res_array.insert(0,[x1,x2,y1,y2])  

                                if cls==5 and stop_sign==0:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                                   
                                    stop_sign_time=time.time()
                                    label = model.names[cls] 
                                    
                                                   
                                    if abs(x1-x2)>55:                          #Threshold width of bounding box for estimating distance
                                        stop_sign=1
                                

                                if (cls!=1 or cls!=3 or cls!=6) and (x2< traffic_light_direction[0]) or (x1>traffic_light_direction[1]):     #checking for traffic light in direction of waypoints
                                    traffic_stop=0
                                    continue



                                # changing the trafficlight flag
                                if cls==3 and traffic_stop==0 :  #red light                                          
                                    
                                    if  w>13:
                                        traffic_stop=1

                                elif cls==6 and traffic_stop==1 :  # yellow light
                                   
                                    if  w>13:
                                        traffic_stop=0
                                       


                                elif cls==1 and traffic_stop==1:   # green light
                                   
                                    if  w>13 :
                                        traffic_stop=0


                            #flag for pedestrian detection         
                            if 2 in res_array:
                                x1,x2,y1,y2=res_array[0]
                                w=abs(x1-x2)
                                
                                if  w>28:                                     # bounding box width threshold
                                    person_flag=1

                                else:
                                    person_flag=0
                            else:
                                person_flag=0
            except IndexError:
                pass

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, v_ref, waypointCounter,pick_up,drop_off,stop_sign,traffic_stop,park_flag,K_stanley,stop_sign_time
    K_stanley=1
    pick_up=0
    drop_off=0
    park_flag =0
    pick_up_time=0
    drop_off_time = 0
    traffic_stop=0
    stop_sign=0
    u = 0
    delta = 0
    LEDs=np.array([0,0,0,0,0, 0, 0, 0])
    countMax = controllerUpdateRate / 10
    count = 0


  #initializing classes for parallel parking ,speedcontrol and steering control

    parking = ParallelParking()

    speedController = SpeedController(
        kp=K_p,
        ki=K_i,
        kd= K_d
    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley
        )
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose,calibrate=calibrate)
    else:
        gps = memoryview(b'')



# for localization
    with qcar, gps:
        t0 = time.time()
        t=0
        while (t < tf+startDelay) and (not KILL_THREAD):
            tp = t
            t = time.time() - t0
            dt = t-tp

            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                th = ekf.x_hat[2,0]
                p = ( np.array([x, y])
                    + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
         

           #for estimating diatnce from pick up location, drop off location and taxi hub area
            pick_up_norm=np.linalg.norm(np.subtract([0.125, 4.395],[x,y]))
            drop_off_norm=np.linalg.norm(np.subtract([-0.960, 0.840],[x,y]))
            taxi_hub_norm = np.linalg.norm(np.subtract([-0.787, -0.935],[x,y]))

            if pick_up_norm< 0.25 and pick_up ==0:
                pick_up = 1
                pick_up_time=time.time()
            if drop_off_norm<0.25 and drop_off==0:
                drop_off=1
                drop_off_time=time.time()
            if taxi_hub_norm<0.25 and t > startDelay+10:
                park_flag=1
            
            if t < startDelay:
                u = 0
                delta = 0
            


            #flags for object detection


            if pick_up==1:
                if time.time()-pick_up_time<3:
                    u,delta=0,0
                    LEDs=np.array([1,1,1, 1, 1, 0, 0, 0])
                   
                else:
                    pick_up=2
                    LEDs=np.array([0,0,0,0,0, 0, 0, 0])
                   

            if drop_off==1:
                if (time.time()-drop_off_time)<3:
                    u,delta=0,0
                    LEDs=np.array([1,1,1, 1, 1, 0, 0, 0])
                   
                else:
                    drop_off=2
                    LEDs=np.array([0,0,0,0,0, 0, 0, 0])
                   

            if stop_sign==1:
                 if (time.time()-stop_sign_time)<3 :
                    u,delta=0,0
                 if (time.time()-stop_sign_time)>3.5:
                    stop_sign=2    
                    stop_sign_time=time.time()
            if (stop_sign==2) and ((time.time()-stop_sign_time)>4):
                stop_sign=0   


            if traffic_stop==1:
                u,delta=0,0


            if person_flag==1:
                u,delta=0,0


            if park_flag ==1:
                print("Reached taxihub back")
                u, delta = parking.update(x, y, th)                                                    # perform parking maneuver at taxi hub

            if not (pick_up==1 or drop_off==1 or stop_sign==1 or traffic_stop==1 or person_flag==1 or park_flag==1):

                # dyanmically changing the reference velocity and stanley gain based on waypoint trajectories
                if waypointCounter > 5 and nextSlope != previousSlopes[-1]:
                    v_ref,K_stanley = 0.4,1

                elif nextSlope == float('inf') and not compare_elements(nextSlope, previousSlopes, 30):
                    v_ref,K_stanley = 0.4, 1

                elif  waypointCounter > 500 and  nextSlope == -float('0.0') and not compare_elements(nextSlope, previousSlopes, 30):
                    v_ref,K_stanley = 0.4, 1

                else:
                    v_ref,K_stanley = 1.25, 0.5



                #region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                #endregion

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v, dt, K_stanley)
                else:
                    delta = 0
                #endregion

            qcar.write(u,delta,LEDs)   

            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref

                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])


                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion
            continue
        qcar.read_write_std(throttle= 0, steering= 0)


if __name__ == '__main__':

    #region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    if enableSteeringControl:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

        steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
        steeringScope.axes[0].attachSignal(name='x_meas')
        steeringScope.axes[0].attachSignal(name='x_ref')

        steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
        steeringScope.axes[1].attachSignal(name='y_meas')
        steeringScope.axes[1].attachSignal(name='y_ref')

        steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
        steeringScope.axes[2].attachSignal(name='th_meas')
        steeringScope.axes[2].attachSignal(name='th_ref')

        steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = 'Time [s]'

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

        im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

        steeringScope.axes[4].attachSignal(name='Estimated', width=2)

        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    
    myCam = QCarRealSense(mode='RGB')
    cameras = QCarCameras(
    enableBack=False,
    enableFront=False,
    enableLeft=True,
    enableRight=False,
    )
    controlThread = Thread(target=controlLoop)
    yolo_thread = Thread(target=yolo_detection)

    yolo_thread.start()
    controlThread.start()


    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        yolo_thread.join()

    input('Experiment complete. Press any key to exit...')

