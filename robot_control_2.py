import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
import time
import cv2 
import imutils
import pickle
import numpy as np
from aruko_pose_estimation import *

# robot params
R=0.05
ax=0.15 #coord of wheels from center
ay=0.33
#Robot velosity coeffs
vx_t=0.000
vy_t=0.000
w_t=0.0

start_time = time.perf_counter()
dt = 0
actual_time = 0
time_buf = 0
x_dist=0
y_dist=0
w_dist=0

f1_t, f2_t, f3_t, f4_t = 0, 0, 0, 0

# Aruco markers id
base_id = 0
target_id = [141,] 
# Нулевой маркер - верхний левый угол (0,0)
# Первый маркер - правый верхний угол (1,0)
# Второй меркаре - левый нижний угол (0,1)
global_markers_id = [138, 141, 137]
# Глобальные направления по оси Х и У в 2Д координатаз камеры
global_vec_X = []
global_vec_Y = []
# Глобальные координаты цели
target_2d_cors = []
# Координаты точки (0,0)
global_base_cors = []
# Общая скорость
total_speed = 0.1




# def callback(data):
#     global start_time
#     global time_buf
#     global actual_time
#     global x_dist
#     global y_dist
#     global w_dist
#     global f1_t, f2_t, f3_t, f4_t

#     actual_time = time.perf_counter() - start_time
    
#     f1=data.velocity[5]
#     f2=data.velocity[6]
#     f3=data.velocity[7]
#     f4=data.velocity[8]

#     vx = -R/4*(f1+f2-f3-f4)
#     vy = -R/4*(f1-f2+f3-f4)
#     w = -R/(4*(ax+ay))*(-f1-f2-f3-f4)

    
#     dt = actual_time - time_buf
#     x_dist = x_dist + vx*dt
#     y_dist = y_dist + vy*dt
#     w_dist = w_dist + w*dt

#     print ('time=', '{:.3f}'.format(actual_time), end='\n')
#     print ('task: ', end='')
#     print ('vx_t=', '{:.3f}'.format(vx_t), ' vy_t=', '{:.3f}'.format(vy_t), ' w_t=','{:.3f}'.format(w_t), end='| ')
#     print('f1_t=', '{:.3f}'.format(f1_t), end='| ')
#     print('f2_t=', '{:.3f}'.format(f2_t), end='| ')
#     print('f3_t=', '{:.3f}'.format(f3_t), end='| ')
#     print('f4_t=', '{:.3f}'.format(f4_t), end='|\n')
#     print('f1=', '{:.3f}'.format(f1), end='| ')
#     print('f2=', '{:.3f}'.format(f2), end='| ')
#     print('f3=', '{:.3f}'.format(f3), end='| ')
#     print('f4=', '{:.3f}'.format(f4), end='| ')
#     print('vx=', '{:.3f}'.format(vx), end='| ')
#     print('vy=', '{:.3f}'.format(vy), end='| ')
#     print('w=', '{:.3f}'.format(w), end='| ')
#     print('x_dist=', '{:.3f}'.format(x_dist), end='| ')
#     print('y_dist=', '{:.3f}'.format(y_dist), end='|')
#     print('w_dist=', '{:.3f}'.format(w_dist), end='|\n')

#     time_buf = time.perf_counter() - start_time


def talker():

    global msg
    global f1_t, f2_t, f3_t, f4_t
    global base_id
    global target_id
    global global_markers_id
    global global_vec_X
    global global_vec_Y
    global target_2d_cors
    global global_base_cors


    # Оценка глобальных координат
    cors_estimated = False
    while not cors_estimated:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
        frame = imutils.resize(frame, width=860)
        estim_frame, global_vec_X, global_vec_Y, global_base_cors, cors_estimated = global_cor_estim(frame, camera_param[0], camera_param[1], global_markers_id)
        cv2.imshow("aruko test", estim_frame)
        cv2.waitKey(1)
    print("Hit Esc to continue")
    cv2.waitKey(0) # Нажать Esc для продолжения работы программы

    # Запрос target
    print("Enter target Aruco marker X (0<=x<=1):")
    x = float(input())
    target_vec_x = x * np.array(global_vec_X) + np.array(global_base_cors)
    print("Enter target Aruco marker Y (0<=y<=1):")
    y = float(input())
    target_vec_y = y * np.array(global_vec_Y) + np.array(global_base_cors)
    target_2d_cors = target_vec_x + target_vec_y - np.array(global_base_cors)  
    print("global_base_cors: ", global_base_cors)
    print("target_cors: ", target_2d_cors)

    while not rospy.is_shutdown():

        # получение кадров
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # изменить размер окна
        frame = imutils.resize(frame, width=860)

        # ARUCO DETECTION
        # aruco_detection(frame)
        
        #POSE ESTIMATION     
        estim_frame, path_vector = pose_estimation(frame, camera_param[0], camera_param[1],
                                                   base_id, target_2d_cors)
        cv2.imshow("aruko test", estim_frame)

        k = cv2.waitKey(1)
        if k%256 == 114:
            # R pressed
            print("Target reload. Robot stopped")
            path_vector = [-1., 0., 0.] # change target code

        # выход
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        if len(path_vector)!=0:

            # if need to change target
            if path_vector[0] == -1.:
                msg.data = [0.,0.,0.,0.]
                pub.publish(msg) 
                rate.sleep()

                # Запрос target
                print("Enter target Aruco marker X (0<=x<=1):")
                x = float(input())
                target_vec_x = x * np.array(global_vec_X) + np.array(global_base_cors)
                print("Enter target Aruco marker Y (0<=y<=1):")
                y = float(input())
                target_vec_y = y * np.array(global_vec_Y) + np.array(global_base_cors)
                target_2d_cors = target_vec_x + target_vec_y - np.array(global_base_cors)  
                print("global_base_cors: ", global_base_cors)
                print("target_cors: ", target_2d_cors)
            else:
                # норма вектора
                norm_vec = pow(pow(path_vector[0],2)+pow(path_vector[1],2), 0.5)
                vx_t = total_speed*path_vector[0]/norm_vec   
                vy_t = total_speed*(-1)*path_vector[1]/norm_vec    
                f1_t=-(1/R)*(vx_t+vy_t-w_t*(ax+ay))
                f2_t=(1/R)*(-vx_t+vy_t+w_t*(ax+ay))
                f3_t=-(1/R)*(-vx_t+vy_t-w_t*(ax+ay))
                f4_t=(1/R)*(vx_t+vy_t+w_t*(ax+ay))
               
                msg.data = [f1_t,f2_t,f3_t,f4_t]
                pub.publish(msg) 
                rate.sleep()
     
        

    cam.release()
    cv2.destroyAllWindows()
    #keeps python from exiting until this node is stopped
    rospy.spin()

# срабатывает при остановке скрипта через ctrl+c
def myhook():
    print("shutdown time!")
    # stops robot
    rate.sleep()
    msg.data = [0.0, 0.0, 0.0, 0.0]
    pub.publish(msg) 
    print("SHUTDOWN")


if __name__ == '__main__':
    # захват видео потока
    cam = cv2.VideoCapture(0)

    try:

        # Создание ноды
        rospy.init_node('pubListener', anonymous=False)
        # Создание публиканта
        pub = rospy.Publisher('youbot_base/joints_vel_controller/command', Float64MultiArray, queue_size=10)
        # Создание подписчика
        # rospy.Subscriber("joint_states", JointState, callback)

        # Конфигурация глобальных параметров
        msg = Float64MultiArray()
        rate = rospy.Rate(10)
        rospy.on_shutdown(myhook)

        # Запуск ноды
        talker()
    except rospy.ROSInterruptException:
        pass