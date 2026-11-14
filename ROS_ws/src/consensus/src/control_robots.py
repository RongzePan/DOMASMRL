#! /usr/bin/env python


## version 2:
## 1, navigate the robot using a constant heading angle
## 2, add the ddpg neural network
## 3, 24 laser data and just heading
## 4, added potential collisions
## 版本 2:
## 1, 使用固定的航向角导航机器人
## 2, 添加 DDPG 神经网络
## 3, 使用 24 个激光数据和航向
## 4, 增加潜在碰撞的处理


## Command:
## 命令:
## roslaunch turtlebot_iros turtlebot_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot/turtlebot_iros/modified_world.world'
## source ~/iros_env/bin/activate
## rosrun turtlebot_iros ddpg_turtlebot.py

# 导入 MAPPO 算法和策略模块
from mappo.algorithms.algorithm import r_actor_critic, r_mappo
from mappo.runner.separated import base_runner
from mappo.envs import env_core
# 导入 DOMAS 环境
from consensus.src.MultipleTurtlebotsEnv import MultipleTurtlebotsEnv

import rospy
# ROS Python 客户端库
import rospkg
# ROS 包管理工具
import tf
# 用于处理坐标变换
from std_msgs.msg import String
# 导入标准消息类型
from geometry_msgs.msg import Twist, Point, Quaternion
# 导入几何消息类型
from math import radians, copysign, sqrt, pow, pi, atan2
# 导入数学函数
from tf.transformations import euler_from_quaternion
# 四元数转换工具

import threading
# 用于多线程操作
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
# 导入 Gazebo 模型消息
from gazebo_msgs.srv import SetModelState
# 用于设置 Gazebo 模型状态
from sensor_msgs.msg import LaserScan
# 激光扫描消息
# from kobuki_msgs.msg import BumperEvent
import time
# 时间操作库
# import gym
# 强化学习环境库
import numpy as np
# 数值计算库
import math
# 数学库
import random
# 随机数库

from std_srvs.srv import Empty
# 空服务，用于暂停和重置仿真

# from std_srvs.srv import Empty

class InfoGetter(object):
    # 信息获取器类，用于阻塞等待接收到的信息
    def __init__(self):
        # event that will block until the info is received
        # 初始化事件，用于阻塞直到信息接收到
        self._event = threading.Event()
        # attribute for storing the rx'd message
        # 存储接收到的消息
        self._msg = None

    def __call__(self, msg):
        # Uses __call__ so the object itself acts as the callback
        # 使用 __call__ 使对象本身成为回调函数
        # save the data, trigger the event
        # 触发事件
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        """阻塞直到接收到数据，可设置超时，返回接收到的消息"""
        self._event.wait(timeout)
        return self._msg
# rospy.init_node('test', anonymous=True)
# tf_listener = tf.TransformListener()

# pose_ig = InfoGetter()
# def get_odom(a,b):
#     time.sleep(1)
#     while True:
#         (trans, rot) = tf_listener.lookupTransform(a, b, rospy.Time(0))
#         rotation = euler_from_quaternion(rot)
#         if trans != None:
#             break
#
#     return (Point(*trans), rotation[2])
# (pos,ros) = get_odom('world','robot2_tf/base_link')
class PID:
    """
    Discrete PID control
    """
    """
    离散 PID 控制器
    """
    def __init__(self, P=100.0, I=0.0, D=100.0, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10):
        # 初始化 PID 参数
        self.Kp = P
        # 比例增益
        self.Ki = I
        # 积分增益
        self.Kd = D
        # 微分增益
        self.Derivator = Derivator
        # 微分器
        self.Integrator = Integrator
        # 积分器
        self.Integrator_max = Integrator_max
        # 积分器最大值
        self.Integrator_min = Integrator_min
        # 积分器最小值
        self.set_point = 0.0
        # 设置点
        self.error = 0.0
        # 当前误差

    def update(self, current_value):
        # 更新 PID 控制量
        self.error = self.set_point - current_value
        # 针对圆形情况的特殊设计
        if self.error > pi:  # specific design for circular situation
            self.error = self.error - 2*pi
        elif self.error < -pi:
            self.error = self.error + 2*pi
        # 计算各项控制量
        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error
        self.Integrator = self.Integrator + self.error
        # 积分限幅
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min
        self.I_value = self.Integrator * self.Ki
        PID = self.P_value + self.I_value + self.D_value
        return PID

    def setPoint(self, set_point):
        # 设置目标点
        self.set_point = set_point
        self.Derivator = 0
        self.Integrator = 0

    def setPID(self, set_P=100.0, set_I=0.0, set_D=100.0):
        # 设置 PID 参数
        self.Kp = set_P
        self.Ki = set_I
        self.Kd = set_D
class InfoGetter(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg


class Formation:
    # 机器人编队控制类
    def __init__(self):
        # ROS 节点必须先初始化，再创建 ServiceProxy / Publisher
        rospy.init_node('talker', anonymous=True)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # 初始化 DOMAS 环境
        self.domas_env = MultipleTurtlebotsEnv()
        # 初始化 MAPPO 环境
        self.mappo_env = env_core.EnvCore()

        # 配置 MAPPO 策略
        self.mappo_agent = r_mappo.RMAPPOAgent(env=self.mappo_env, config={
            'obs_dim': self.mappo_env.obs_dim,
            'action_dim': self.mappo_env.action_dim,
            'num_agents': self.mappo_env.agent_num
        })

        # MAPPO 代理的配置
        config = {
            "num_agents": 4,  # 机器人数量
            "action_space": self.domas_env.action_space,
            "observation_space": self.domas_env.observation_space,
        }

        # 初始化 MAPPO 代理
        self.agent = r_mappo.RMAPPOAgent(env=self.domas_env, config=config)

        # 初始化控制指令和 ROS 发布者
        self.move_cmd1 = Twist()
        self.move_cmd2 = Twist()
        self.move_cmd3 = Twist()
        self.move_cmd4 = Twist()
        self.pub_robot1 = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=1)
        self.pub_robot2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=1)
        self.pub_robot3 = rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=1)
        self.pub_robot4 = rospy.Publisher('/robot4/cmd_vel', Twist, queue_size=1)

        # 设置机器人坐标和目标点
        self.odom = 'world'
        self.odom1,self.odom2,self.odom3,self.odom4 = 'robot1_tf/base_link', 'robot2_tf/base_link', 'robot3_tf/base_link', 'robot4_tf/base_link'
        self.position = Point()
        self.tf_listener = tf.TransformListener()
        # 初始化控制指令
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.move_cmd1 = Twist()
        self.move_cmd1.linear.x = 0
        self.move_cmd1.angular.z = 0
        self.move_cmd2 = Twist()
        self.move_cmd2.linear.x = 0
        self.move_cmd2.angular.z = 0
        self.move_cmd3 = Twist()
        self.move_cmd3.linear.x = 0
        self.move_cmd3.angular.z = 0
        self.move_cmd4 = Twist()
        self.move_cmd4.linear.x = 0
        self.move_cmd4.angular.z = 0
        self.rate = rospy.Rate(10)
        self.talker()
        time.sleep(1)
        self.talker()
        self.pid_theta = PID(0, 0, 0)
        self.theta = 0
        self.target1 = [10,1]
        self.target2 = [5, 10]
        self.target3 = [10, 2]
        self.target4 = [3, 5]
        self.position_target_1 = Point()
        self.position_target_1.x = 1
        self.position_target_1.y = 0
        self.position_target_2 = Point()
        self.position_target_2.x = .5
        self.position_target_2.y = 1
        self.position_target_3 = Point()
        self.position_target_3.x = 1
        self.position_target_3.y = .2
        self.position_target_4 = Point()
        self.position_target_4.x = .3
        self.position_target_4.y = .5

        # self.position_target_1 = Point()
        # self.position_target_1.x = 2
        # self.position_target_1.y = 1.5
        # self.position_target_2 = Point()
        # self.position_target_2.x = 2
        # self.position_target_2.y = 1.5
        # self.position_target_3 = Point()
        # self.position_target_3.x = 2
        # self.position_target_3.y = 1.5
        # self.position_target_4 = Point()
        # self.position_target_4.x = 2
        # self.position_target_4.y = 1.5


        self.Lambda = np.zeros(8)

    def get_odom(self,a,b):
        # 获取两个坐标系间的位姿
        try:
            (trans, rot) = self.tf_listener.lookupTransform(a,b, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])
    def talker(self):
        # 初始化发布器并发送初始指令
        self.pub_robot1 = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=1)
        self.pub_robot2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=1)
        self.pub_robot3 = rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=1)
        self.pub_robot4 = rospy.Publisher('/robot4/cmd_vel', Twist, queue_size=1)
        rate = rospy.Rate(100) # 10hz
        # while not rospy.is_shutdown():
        self.pub_robot1.publish(self.move_cmd1)
        self.pub_robot2.publish(self.move_cmd2)
        self.pub_robot3.publish(self.move_cmd3)
        self.pub_robot4.publish(self.move_cmd4)

        # rate.sleep()

    def step(self):

        # 设定训练的总回合数
        num_episodes = 100  # 可以根据需求调整

        # 创建 ROS 频率控制
        rate = rospy.Rate(10)  # 10 Hz，可以根据需要调整

        for episode in range(num_episodes):
            rospy.loginfo(f"Starting episode {episode + 1}")
            # 重置环境以获取初始状态
            state = self.domas_env.reset()
            done = False

            while not done and not rospy.is_shutdown():
                # 通过 MAPPO 代理生成的策略获取动作
                action = self.agent.select_action(state)

                # 将每个动作应用于机器人，通过 ROS 发布控制命令
                for i, robot_action in enumerate(action):
                    move_cmd = Twist()
                    move_cmd.linear.x = robot_action[0]
                    move_cmd.angular.z = robot_action[1]

                    # 根据不同的机器人发布到相应的主题
                    if i == 0:
                        self.pub_robot1.publish(move_cmd)
                    elif i == 1:
                        self.pub_robot2.publish(move_cmd)
                    elif i == 2:
                        self.pub_robot3.publish(move_cmd)
                    elif i == 3:
                        self.pub_robot4.publish(move_cmd)

                # 在环境中执行动作，返回下一个状态、奖励、是否结束标志以及其他信息
                next_state, reward, done, info = self.domas_env.step(action)

                # 更新 MAPPO 代理的状态，以便强化学习
                self.agent.update(state, action, reward, next_state)

                # 输出调试信息
                rospy.loginfo(f"Reward: {reward}, Done: {done}")

                # 更新当前状态
                state = next_state

                # 控制执行频率
                rate.sleep()

            rospy.loginfo(f"Episode {episode + 1} ended.")

        # 执行编队移动的步进控制
        while not rospy.is_shutdown():
            try:
                self.tf_listener.waitForTransform(self.odom, self.odom1, rospy.Time(), rospy.Duration(1.0))
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                try:
                    self.tf_listener.waitForTransform(self.odom, self.odom1, rospy.Time(),
                                                      rospy.Duration(1.0))
                except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                    rospy.loginfo("Cannot find transform between odom and jackal_tf/base_link or robot1_tf/base_link")
                    rospy.signal_shutdown("tf Exception")
            (self.position_robot1, self.rotation_robot1) = self.get_odom(self.odom, self.odom1)

            try:
                self.tf_listener.waitForTransform(self.odom, self.odom2, rospy.Time(), rospy.Duration(1.0))
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                try:
                    self.tf_listener.waitForTransform(self.odom, self.odom2, rospy.Time(),
                                                      rospy.Duration(1.0))
                except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                    rospy.loginfo("Cannot find transform between odom and jackal_tf/base_link or robot1_tf/base_link")
                    rospy.signal_shutdown("tf Exception")
            (self.position_robot2, self.rotation_robot2) = self.get_odom(self.odom, self.odom2)

            try:
                self.tf_listener.waitForTransform(self.odom, self.odom3, rospy.Time(), rospy.Duration(1.0))
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                try:
                    self.tf_listener.waitForTransform(self.odom, self.odom3, rospy.Time(),
                                                      rospy.Duration(1.0))
                except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                    rospy.loginfo("Cannot find transform between odom and jackal_tf/base_link or robot1_tf/base_link")
                    rospy.signal_shutdown("tf Exception")
            (self.position_robot3, self.rotation_robot3) = self.get_odom(self.odom, self.odom3)

            try:
                self.tf_listener.waitForTransform(self.odom, self.odom4, rospy.Time(), rospy.Duration(1.0))
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                try:
                    self.tf_listener.waitForTransform(self.odom, self.odom4, rospy.Time(),
                                                      rospy.Duration(1.0))
                except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                    rospy.loginfo("Cannot find transform between odom and jackal_tf/base_link or robot1_tf/base_link")
                    rospy.signal_shutdown("tf Exception")
            (self.position_robot4, self.rotation_robot4) = self.get_odom(self.odom, self.odom4)

            print("Robot1:", self.position_robot1)

            print("Robot2:", self.position_robot2)

            print("Robot3:", self.position_robot3)

            print("Robot4:", self.position_robot4)
            # self.move_to_point()
            self.target_update()
            self.checkarrive()

            # self.move_cmd2.linear.x = 0
            # self.move_cmd2.angular.z = 0
            # self.talker()

    def target_update(self):
        # 移动机器人到目标点
        L = np.array([[2,-1,0,-1],[-1,2,-1,0],[0,-1,2,-1],[-1,0,-1,2]])
        dx1 = self.target1[0] - self.position_robot1.x
        dy1 = self.target1[1] - self.position_robot1.y
        dx2 = self.target2[0] - self.position_robot2.x
        dy2 = self.target2[1] - self.position_robot2.y
        dx3 = self.target3[0] - self.position_robot3.x
        dy3 = self.target3[1] - self.position_robot3.y
        dx4 = self.target4[0] - self.position_robot4.x
        dy4 = self.target4[1] - self.position_robot4.y

        Y = np.array([self.position_robot1.x,self.position_robot1.y,self.position_robot2.x,self.position_robot2.y,self.position_robot3.x,self.position_robot3.y,self.position_robot4.x,self.position_robot4.y])
        alpha = 0.1
        beta = 0.1
        delta = 0.1
        gradient = -1*np.array([2*dx1,2*dy1,2*dx2,2*dy2,2*dx3,2*dy3,2*dx4,2*dy4])
        Y_target = Y - beta * (np.matmul(np.kron(L, np.eye(2)),Y) + np.matmul(np.kron(L, np.eye(2)),self.Lambda) + alpha * gradient)
        self.Lambda += delta * np.matmul(np.kron(L, np.eye(2)) , Y_target)

        self.position_target_1.x = Y_target[0]
        self.position_target_1.y = Y_target[1]
        self.position_target_2.x = Y_target[2]
        self.position_target_2.y = Y_target[3]
        self.position_target_3.x = Y_target[4]
        self.position_target_3.y = Y_target[5]
        self.position_target_4.x = Y_target[6]
        self.position_target_4.y = Y_target[7]

    def checkarrive(self):
        # 检查机器人是否到达目标点
        dx1 = self.position_target_1.x - self.position_robot1.x
        dy1 = self.position_target_1.y - self.position_robot1.y
        dx2 = self.position_target_2.x - self.position_robot2.x
        dy2 = self.position_target_2.y - self.position_robot2.y
        dx3 = self.position_target_3.x - self.position_robot3.x
        dy3 = self.position_target_3.y - self.position_robot3.y
        dx4 = self.position_target_4.x - self.position_robot4.x
        dy4 = self.position_target_4.y - self.position_robot4.y
        # print("position of robot 1", dx1, dy1)

        # calculate for robot1

        if np.sqrt(dx1 ** 2 + dy1 ** 2) < 0.6:
            print("ROBOT2 has Arrived")
            self.move_cmd1.linear.x = 0
            self.move_cmd1.angular.z = -self.rotation_robot1
            # self.move_cmd1.angular.z = 0
            # self.talker()
        else:
            # print("Distance:", np.sqrt(dx1 ** 2 + dy1 ** 2))
            angle_turtlebot1 = self.rotation_robot1
            angle_turtlebot_target1 = atan2(dy1, dx1)
            if angle_turtlebot1 < 0:
                angle_turtlebot1 = angle_turtlebot1 + 2 * math.pi
            if angle_turtlebot_target1 < 0:
                angle_turtlebot_target1 = angle_turtlebot_target1 + 2 * math.pi
            angle_diff1 = angle_turtlebot_target1 - angle_turtlebot1
            if angle_diff1 < -math.pi:
                angle_diff1 = angle_diff1 + 2 * math.pi
            if angle_diff1 > math.pi:
                angle_diff1 = angle_diff1 - 2 * math.pi
            self.move_cmd1.angular.z = np.clip(angle_diff1, -0.5, 0.5)
            self.move_cmd1.linear.x = np.clip(np.sqrt(dx1 ** 2 + dy1 ** 2), 0, 0.05)

        # calculate for robot2
        if np.sqrt(dx2 ** 2 + dy2 ** 2) < 0.3:
            print("ROBOT2 has Arrived")
            self.move_cmd2.linear.x = 0
            # self.move_cmd2.angular.z = 0
            self.move_cmd2.angular.z = -self.rotation_robot2
            # self.talker()
        else:
            print("Distance:", np.sqrt(dx2 ** 2 + dy2 ** 2))
            angle_turtlebot2 = self.rotation_robot2
            angle_turtlebot_target2 = atan2(dy2, dx2)
            if angle_turtlebot2 < 0:
                angle_turtlebot2 = angle_turtlebot2 + 2 * math.pi
            if angle_turtlebot_target2 < 0:
                angle_turtlebot_target2 = angle_turtlebot_target2 + 2 * math.pi
            angle_diff2 = angle_turtlebot_target2 - angle_turtlebot2
            if angle_diff2 < -math.pi:
                angle_diff2 = angle_diff2 + 2 * math.pi
            if angle_diff2 > math.pi:
                angle_diff2 = angle_diff2 - 2 * math.pi
            self.move_cmd2.angular.z = np.clip(angle_diff2, -0.5, 0.5)
            self.move_cmd2.linear.x = np.clip(np.sqrt(dx2 ** 2 + dy2 ** 2), 0, 0.05)
            # self.talker()

        # calculate for robot3
        if np.sqrt(dx3 ** 2 + dy3 ** 2) < 0.2:
            print("ROBOT3 has Arrived")
            self.move_cmd3.linear.x = 0
            # self.move_cmd3.angular.z = 0
            self.move_cmd3.angular.z = -self.rotation_robot3
            # self.talker()
        else:
            print("ROBOT3 Distance to Target:", np.sqrt(dx3 ** 2 + dy3 ** 2))
            angle_turtlebot3 = self.rotation_robot3
            angle_turtlebot_target3 = atan2(dy3, dx3)
            if angle_turtlebot3 < 0:
                angle_turtlebot3 = angle_turtlebot3 + 2 * math.pi
            if angle_turtlebot_target3 < 0:
                angle_turtlebot_target3 = angle_turtlebot_target3 + 2 * math.pi
            angle_diff3 = angle_turtlebot_target3 - angle_turtlebot3
            if angle_diff3 < -math.pi:
                angle_diff3 = angle_diff3 + 2 * math.pi
            if angle_diff3 > math.pi:
                angle_diff3 = angle_diff3 - 2 * math.pi
            self.move_cmd3.angular.z = np.clip(angle_diff3, -0.5, 0.5)
            self.move_cmd3.linear.x = np.clip(np.sqrt(dx3 ** 2 + dy3 ** 2), 0, 0.05)
        #
        # # calculate for robot4
        if np.sqrt(dx4 ** 2 + dy4 ** 2) < 0.3:
            print("ROBOT4 has Arrived")
            self.move_cmd4.linear.x = 0
            self.move_cmd4.angular.z = -self.rotation_robot4
            # self.move_cmd4.angular.z = 0
            # self.talker()
        else:
            print("ROBOT4 Distance to Target:", np.sqrt(dx4 ** 2 + dy4 ** 2))
            angle_turtlebot4 = self.rotation_robot4
            angle_turtlebot_target4 = atan2(dy4, dx4)
            if angle_turtlebot4 < 0:
                angle_turtlebot4 = angle_turtlebot4 + 2 * math.pi
            if angle_turtlebot_target4 < 0:
                angle_turtlebot_target4 = angle_turtlebot_target4 + 2 * math.pi
            angle_diff4 = angle_turtlebot_target4 - angle_turtlebot4
            if angle_diff4 < -math.pi:
                angle_diff4 = angle_diff4 + 2 * math.pi
            if angle_diff4 > math.pi:
                angle_diff4 = angle_diff4 - 2 * math.pi
            self.move_cmd4.angular.z = np.clip(angle_diff4, -0.5, 0.5)
            self.move_cmd4.linear.x = np.clip(np.sqrt(dx4 ** 2 + dy4 ** 2), 0, 0.05)


        self.talker()
    def reset(self):
        # 重置机器人位置和控制指令
        state_msg = [ModelState() for _ in range(4)]
        positions = [(0, 0), (-2.5, 0.5), (3.4, 1.5), (10, 10)]
        for i in range(4):
            state_msg[i].model_name = f'Robot{i + 1}'
            state_msg[i].pose.position.x, state_msg[i].pose.position.y = positions[i]
            state_msg[i].pose.position.z = 0.0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        for msg in state_msg:
            set_state(msg)

if __name__ == '__main__':
    # 主执行入口，初始化编队并开始运行
    try:
        # 实例化编队控制
        formation = Formation()
        # 初始化环境
        formation.reset()
        # 开始执行编队控制
        while not rospy.is_shutdown():
            formation.step()

        consensus = Formation()
        consensus.reset()
        while True:
            consensus.step()

    except rospy.ROSInterruptException:
        pass
