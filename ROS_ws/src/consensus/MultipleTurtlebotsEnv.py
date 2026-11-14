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
# 导入 Gazebo 模型消息
from gazebo_msgs.msg import ModelState
# 导入 Gazebo 模型消息

from gazebo_msgs.srv import SetModelState
# 用于设置 Gazebo 模型状态
from sensor_msgs.msg import LaserScan
# 激光扫描消息
# from kobuki_msgs.msg import BumperEvent
import time
# 时间操作库

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
# class InfoGetter(object):
#     def __init__(self):
#         # event that will block until the info is received
#         self._event = threading.Event()
#         # attribute for storing the rx'd message
#         self._msg = None
#
#     def __call__(self, msg):
#         # Uses __call__ so the object itself acts as the callback
#         # save the data, trigger the event
#         self._msg = msg
#         self._event.set()
#
#     def get_msg(self, timeout=None):
#         """Blocks until the data is rx'd with optional timeout
#         Returns the received message
#         """
#         self._event.wait(timeout)
#         return self._msg


class Formation:
    # 机器人编队控制类
    def __init__(self):
        # ROS 节点必须先初始化，再创建 ServiceProxy / Publisher
        rospy.init_node('talker', anonymous=True)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # 设置机器人坐标和目标点
        self.odom1,self.odom2,self.odom3,self.odom4 = 'robot1_tf/base_link', 'robot2_tf/base_link', 'robot3_tf/base_link', 'robot4_tf/base_link'
        self.position = Point()
        self.position_target_2 = Point()
        self.position_target_2.y = -2
        self.position_target_3 = Point()
        self.position_target_3.x = -2
        self.position_target_4 = Point()
        self.position_target_4.y = 2
        self.tf_listener = tf.TransformListener()
        # 初始化控制指令
        self.move_cmd = Twist()
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
        self.talker()
        self.pid_theta = PID(0, 0, 0)
        self.theta = 0
        self.rate = rospy.Rate(10)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
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
        rate = rospy.Rate(10) # 10hz
        # while not rospy.is_shutdown():
        self.pub_robot1.publish(self.move_cmd1)
        self.pub_robot2.publish(self.move_cmd2)
        self.pub_robot3.publish(self.move_cmd3)
        self.pub_robot4.publish(self.move_cmd4)

        # rate.sleep()

    def step(self):
        # 执行编队移动的步进控制
        while not rospy.is_shutdown():
            position_rotation = self.get_odom(self.odom1, self.odom2)
            if position_rotation:
                self.position_robot2, self.rotation_robot2 = position_rotation
            else:
                rospy.logwarn("Failed to get odom for robot2")

            position_rotation = self.get_odom(self.odom1, self.odom3)
            if position_rotation:
                self.position_robot3, self.rotation_robot3 = position_rotation
            else:
                rospy.logwarn("Failed to get odom for robot3")

            position_rotation = self.get_odom(self.odom1, self.odom4)
            if position_rotation:
                self.position_robot4, self.rotation_robot4 = position_rotation
            else:
                rospy.logwarn("Failed to get odom for robot4")

            self.checkarrive()
            self.talker()

    def move_to_point(self):
        # 移动机器人到目标点
        # Compute orientation for angular vel and direction vector for linear vel
        # x = self.position_target_2.x - self.position_robot2.x
        # y = self.position_target_2.y - self.position_robot2.y
        diff_x = self.position_target_2.x - self.position_robot2.x
        diff_y = self.position_target_2.y - self.position_robot2.y
        direction_vector = np.array([diff_x, diff_y])
        direction_vector = direction_vector/sqrt(diff_x*diff_x + diff_y*diff_y)  # normalization
        theta = atan2(diff_y, diff_x)

        # We should adopt different parameters for different kinds of movement
        self.pid_theta.setPID(1, 0, 0)     # P control while steering
        self.pid_theta.setPoint(theta)
        rospy.logwarn("### PID: set target theta = " + str(theta) + " ###")

        # Adjust orientation first
        while not rospy.is_shutdown():
            angular = self.pid_theta.update(self.theta)
            # angular =
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2
            if abs(angular) < 0.01:
                break
            self.move_cmd2.linear.x = 0
            self.move_cmd2.angular.z = angular
            self.pub_robot2.publish(self.move_cmd2)
            self.rate.sleep()

        # Have a rest
        self.stop()
        self.pid_theta.setPoint(theta)
        #self.pid_theta.setPID(1, 0, 0)   # PI control while moving
        self.pid_theta.setPID(1, 0.02, 0.2)  # PID control while moving

        # Move to the target point
        while not rospy.is_shutdown():
            diff_x = self.position_target_2.x - self.position_robot2.x
            diff_y = self.position_target_2.y - self.position_robot2.y
            vector = np.array([diff_x, diff_y])
            linear = np.dot(vector, direction_vector) # projection
            if abs(linear) > 0.2:
                linear = linear/abs(linear)*0.2

            angular = self.pid_theta.update(self.theta)
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2

            if abs(linear) < 0.01 and abs(angular) < 0.01:
                break
            self.move_cmd2.linear.x = linear
            self.move_cmd2.angular.z = angular
            self.pub_robot2.publish(self.move_cmd2)
            self.rate.sleep()

    def stop(self):
        self.move_cmd2.linear.x = 0
        self.move_cmd2.angular.z = 0
        self.pub_robot2.publish(self.move_cmd2)

    def checkarrive(self):
        # 检查机器人是否到达目标点
        dx2 = self.position_target_2.x - self.position_robot2.x
        dy2 = self.position_target_2.y - self.position_robot2.y
        dx3 = self.position_target_3.x - self.position_robot3.x
        dy3 = self.position_target_3.y - self.position_robot3.y
        dx4 = self.position_target_4.x - self.position_robot4.x
        dy4 = self.position_target_4.y - self.position_robot4.y
        if np.sqrt(dx2 ** 2 + dy2 ** 2) < 0.05:
            print("ROBOT2 has Arrived")
            self.move_cmd2.linear.x = 0
            self.move_cmd2.angular.z = -self.rotation_robot2
            self.talker()
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
            self.move_cmd2.angular.z = np.clip(angle_diff2, -0.3, 0.3)
            self.move_cmd2.linear.x = np.clip(np.sqrt(dx2 ** 2 + dy2 ** 2), 0, 0.1)


        # calculate for robot3
        if np.sqrt(dx3 ** 2 + dy3 ** 2) < 0.05:
            print("ROBOT3 has Arrived")
            self.move_cmd3.linear.x = 0
            self.move_cmd3.angular.z = -self.rotation_robot3
            self.talker()
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
            self.move_cmd3.angular.z = np.clip(angle_diff3, -0.3, 0.3)
            self.move_cmd3.linear.x = np.clip(np.sqrt(dx3 ** 2 + dy3 ** 2), 0, 0.1)

        # calculate for robot4
        if np.sqrt(dx4 ** 2 + dy4 ** 2) < 0.05:
            print("ROBOT4 has Arrived")
            self.move_cmd4.linear.x = 0
            self.move_cmd4.angular.z = -self.rotation_robot4
            self.talker()
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
            self.move_cmd4.angular.z = np.clip(angle_diff4, -0.3, 0.3)
            self.move_cmd4.linear.x = np.clip(np.sqrt(dx4 ** 2 + dy4 ** 2), 0, 0.1)


            self.talker()

    def reset(self):
        # 重置机器人位置和控制指令
        # self.target_x = 3.15
        # self.target_y = 3.45
        # self.crash_indicator = 0
        state_msg1 = ModelState()
        state_msg1.model_name = 'Robot1'
        state_msg1.pose.position.x = 0.0
        state_msg1.pose.position.y = 0.0
        state_msg1.pose.position.z = 0.0
        state_msg1.pose.orientation.x = 0
        state_msg1.pose.orientation.y = 0
        state_msg1.pose.orientation.z = 0
        state_msg1.pose.orientation.w = 0
        state_msg2 = ModelState()
        state_msg2.model_name = 'Robot2'
        state_msg2.pose.position.x = 1.0
        state_msg2.pose.position.y = 0.0
        state_msg2.pose.position.z = 0.0
        state_msg2.pose.orientation.x = 0
        state_msg2.pose.orientation.y = 0
        state_msg2.pose.orientation.z = 0
        state_msg2.pose.orientation.w = 0
        state_msg3 = ModelState()
        state_msg3.model_name = 'Robot3'
        state_msg3.pose.position.x = 1.0
        state_msg3.pose.position.y = 1.0
        state_msg3.pose.position.z = 0.0
        state_msg3.pose.orientation.x = 0
        state_msg3.pose.orientation.y = 0
        state_msg3.pose.orientation.z = 0
        state_msg3.pose.orientation.w = 0
        state_msg4 = ModelState()
        state_msg4.model_name = 'Robot4'
        state_msg4.pose.position.x = 0.0
        state_msg4.pose.position.y = 1.0
        state_msg4.pose.position.z = 0.0
        state_msg4.pose.orientation.x = 0
        state_msg4.pose.orientation.y = 0
        state_msg4.pose.orientation.z = 0
        state_msg4.pose.orientation.w = 0

        rospy.wait_for_service('gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp1 = set_state(state_msg1)
            resp2 = set_state(state_msg2)
            resp3 = set_state(state_msg3)
            resp4 = set_state(state_msg4)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub_robot1.publish(self.move_cmd)
        self.pub_robot2.publish(self.move_cmd)
        self.pub_robot3.publish(self.move_cmd)
        self.pub_robot4.publish(self.move_cmd)
        time.sleep(1)
        self.pub_robot1.publish(self.move_cmd)
        self.pub_robot2.publish(self.move_cmd)
        self.pub_robot3.publish(self.move_cmd)
        self.pub_robot4.publish(self.move_cmd)
        self.rate.sleep()

class MultipleTurtlebotsEnv(object):
    """Gym-like DOMAS environment used by ``control_robots.py``.

    ``control_robots.py`` imported this class and called ``reset()`` /
    ``step()`` / ``action_space`` / ``observation_space`` on it, while the only
    class in this module was :class:`Formation` (a blocking ROS formation
    controller with an infinite ``step()`` loop and no RL interface).  This
    adapter supplies the missing contract on top of
    :class:`mappo.envs.env_core.EnvCore`.

    :returns from step: ``(next_state, reward, done, info)`` where ``done`` is a
        single boolean, as required by the ``while not done`` loop in
        ``Formation.step``.
    """

    def __init__(self, num_agents=4, obs_dim=14, action_dim=5, max_steps=200):
        # Imported lazily so that importing this ROS module does not require
        # gym / matplotlib / torch on the robot.
        from mappo.envs.env_core import EnvCore

        self.core = EnvCore(num_agents=num_agents, obs_dim=obs_dim,
                            action_dim=action_dim, max_steps=max_steps)
        self.num_agents = self.agent_num = self.core.agent_num
        self.obs_dim = self.core.obs_dim
        self.action_dim = self.core.action_dim
        self.observation_space = self.core.observation_space
        self.action_space = self.core.action_space
        self.share_observation_space = self.core.share_observation_space
        self.max_steps = max_steps

    def reset(self):
        """Return a stacked ``(num_agents, obs_dim)`` initial state."""
        return np.stack(self.core.reset()).astype(np.float32)

    def step(self, actions):
        """Apply one joint action and return ``(state, reward, done, info)``."""
        obs, rewards, dones, infos = self.core.step(actions)
        state = np.stack(obs).astype(np.float32)
        reward = float(np.mean(np.asarray(rewards, dtype=np.float32)))
        done = bool(all(dones))
        info = {
            "per_agent_rewards": np.asarray(rewards, dtype=np.float32).reshape(-1),
            "per_agent_dones": np.asarray(dones),
            "step": self.core.current_step,
            "infos": infos,
        }
        return state, reward, done, info

    def seed(self, seed=None):
        return self.core.seed(seed)

    def close(self):
        return self.core.close()

    def render(self, mode="human"):
        return self.core.render(mode=mode)


if __name__ == '__main__':
    # 主执行入口，初始化编队并开始运行
    try:
        consensus = Formation()
        while True:
            # consensus.step()
            consensus.reset()
    except rospy.ROSInterruptException:
        pass