import numpy as np
from math import pi, atan2, sqrt
from pathlib import Path
from mappo.envs.spaces_compat import spaces

try:  # plotting is optional; the environment must stay usable headless
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on the host environment
    plt = None

"""
1. Visualize the robot's path (e.g., each step)
2. Visualize the learning curve
3. Visualize the reward data to ensure that reinforcement learning capabilities are being used and to avoid getting trapped in local optima
4. Avoid pseudo-learning by setting lower bounds on rewards, upper bounds on steps, or otherwise
"""

# Class definition and initialization
# Define a virtual Turtlebot

class VirtualTurtlebot:
    """Virtual TurtleBot agent"""

    def __init__(self, robot_id, init_pose, target_pose, render_mode=False):
        self.id = robot_id
        # Unique identification of robots
        # self.pose = np.array(init_pose, dtype=np.float32)
        self.pose = init_pose
        # [x, y, theta] Current pose [x coordinate, y coordinate, orientation in radians]
        # self.target = np.array(target_pose, dtype=np.float32)
        self.target = target_pose
        # Target position [target x-coordinate, target y-coordinate]
        self.linear_vel = 0.0
        # Linear velocity (m/s)
        self.angular_vel = 0.0
        # Angular velocity (rad/s)
        self.laser_scan = np.full(24, 3.0)
        # Simulated 24-beam LiDAR data(The initial value is 3.0, indicating no occlusion)
        # Trace record property
        # self.trajectory = [init_pose[:2].copy()]
        self.trajectory = []
        # Trajectory storage
        self.reward_components = {
            'navigation': [],
            'time_penalty': [],
            'energy': [],
            'collision': [],
            'success': []
        }
        self.render_mode = render_mode
        self.path_x = []
        # Store X coordinates of the path
        self.path_y = []
        # Store Y coordinates of the path

    def update_trajectory(self):
        self.trajectory.append([self.pose[0], self.pose[1]])
        # Make sure only x, y are saved

    # Pose update method
    # Update robot pose based on current speed
    def update_pose(self, dt=0.1):
        """Update robot pose using kinematic model"""
        # Simplified kinematic model
        if self.angular_vel != 0:
            delta_theta = self.angular_vel * dt
            # Time step (default 0.1 seconds)
            self.pose[2] = (self.pose[2] + delta_theta) % (2 * pi)
            self.pose[0] += self.linear_vel * np.cos(self.pose[2]) * dt
            self.pose[1] += self.linear_vel * np.sin(self.pose[2]) * dt
        else:
            self.pose[0] += self.linear_vel * np.cos(self.pose[2]) * dt
            self.pose[1] += self.linear_vel * np.sin(self.pose[2]) * dt

        # Track record, only x and y coordinates
        self.trajectory.append([self.pose[0], self.pose[1]])
        # strictly two-dimensional

# Calculate the change in Angle delta_theta modulo ensure the Angle is in [0, 2π)
# update x and y coordinates based on linear velocity and Angle (use trigonometric functions to calculate displacement components)

# Multi-robot Environment Core
# initializer
# define action and observation Spaces in Gym format

class EnvCore(object):
    """
    Multi-turtlebot virtual Environment core class
    Reinforcement learning environment core class, compatible with env_core interface and extended functionality
    """
# Observation space: 28 dimensions per robot (24 lasers + 4 normalizations)
# Action space: 2 dimensions [-1, 1] for each robot, mapped to actual velocity range

    def __init__(self, num_agents=4, render_mode=False):
        # Add constructor arguments
        self.agent_num = num_agents
        # Initialize the number of robots with parameter
        self.obs_dim = 28
        # Observation dimension: 24 LiDAR + 4 normalized values
        self.action_dim = 2
        # Action dimension: [linear velocity, angular velocity]
        self.max_speed = 0.2
        # Maximum linear velocity (m/s)
        self.max_rotate = 0.4
        # Maximum angular velocity (rad/s)
        self.success_threshold = 0.2
        # Distance threshold for reaching target (m)
        self.collision_distance = 0.3
        # Collision detection distance (m)
        self.max_steps = 100
        # Maximum steps per episode
        self.current_step = 0
        # Initialize space definition (changed to dynamic calculation)
        self._init_spaces()
        # Initializer
        self._init_components()
        # Define the action and observation Spaces
        self.action_space = [spaces.Box(-1, 1, (2,))
                             for _ in range(self.agent_num)]
        self.observation_space = [spaces.Box(-np.inf, np.inf, (self.obs_dim,))
                                  for _ in range(self.agent_num)]
        # Shared observation space definition (for centralized training)
        self.share_observation_space = [
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.obs_dim * num_agents,))
            for _ in range(num_agents)
        ]
        # Verification space Definition
        print(f"A single observation space dimension: {self.observation_space[0].shape}")
        print(f"Shared observation space dimensions: {self.share_observation_space[0].shape}")
        # Visualize the initialization element (headless-safe)
        self.scatters = []
        self.lines = []
        self.fig = self.ax = None
        if plt is not None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self._plot_init()
        # self.writer = SummaryWriter()
        # TensorBoard writer
        self.num_agents = num_agents
        self.robots = []
        self.trajectory_lengths = [[] for _ in range(self.num_agents)]
        # Initialize the length record
        self._init_components()
        self.render_mode = render_mode
        self.path_x = []
        # Store the x-coordinate of the path
        self.path_y = []
        # Store the y-coordinate of the path
        self.max_initial_distance = 5.0
        # Max initial distance based on environment bounds
        self.dangerous_distance = 0.5
        # Collision warning threshold
        self.early_stop_ratio = 0.8
        # Early stopping step ratio
        self.initial_poses = []


    def _plot_init(self):
        """Initialize visual elements (strict type validation version)"""
        # Draw obstacles
        for obs in self.obstacles:
            # Coordinate data validation
            try:
                # Convert to indexable sequence
                raw_pos = list(obs['pos'])
                if len(raw_pos) != 2:
                    raise ValueError(
                        f"Obstacle coordinate dimension error - expected 2D coordinates, got: {len(raw_pos)}")

                # Explicit coordinate conversion
                x = float(raw_pos[0])
                y = float(raw_pos[1])
                pos_tuple = (x, y)
                # Verified 2D float tuple

                # Radius type validation
                radius = float(obs['radius'])
            except (TypeError, ValueError, IndexError) as e:
                raise RuntimeError(f"Invalid obstacle data format: {obs}") from e

            # Create graphic object (strict type matching)
            circle = plt.Circle(
                xy=pos_tuple,
                # Using double-verified coordinates
                radius=radius,
                # Strict float type
                color='gray',
                alpha=0.7,
                linewidth=0
                # Remove border lines to prevent warnings
            )
            self.ax.add_patch(circle)

        # Initialize robot trajectory lines
        for _ in range(self.agent_num):
            line, = self.ax.plot([], [], lw=2)
            self.lines.append(line)
            scatter = self.ax.scatter([], [], s=100)
            self.scatters.append(scatter)

    def _init_components(self):
        """Initialize robot and obstacle environment components"""
        self.robots = []
        self._init_obstacles()
        self._init_robots()
        # Initialize robots

        # List of robot instances

    def _init_obstacles(self):
        """Initialize obstacles"""
        # Obstacle configuration
        self.obstacles = [
            {'pos': [2.5, 0.0], 'radius': 0.5},
            {'pos': [-2.0, 1.5], 'radius': 0.3}
        ]

    # Space initialization method
    def _init_spaces(self):
        """Dynamically initialize space definitions"""
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
            for _ in range(self.agent_num)
        ]
        self.action_space = [
            spaces.Box(low=-1, high=1, shape=(self.action_dim,))
            for _ in range(self.agent_num)
        ]

# Robot initialization
# Create robot instances with predefined poses

    def _init_robots(self):
        """Initialize robot configurations"""

        initial_poses = [
            # Initial pose configuration
            # Four robots distributed around origin with different orientations
            [0.0, 0.0, 0.0],
            # Robot 0: Origin (0,0) facing 0°
            [1.0, 0.0, pi / 2],
            # Robot 1: (1,0) facing 90°
            [0.0, 1.0, pi],
            # Robot 2: (0,1) facing 180°
            [1.0, 1.0, -pi / 2]
            # Robot 3: (1,1) facing -90°
        ]
        target_poses = [
            # Target position configuration
            # Four robots moving to different quadrants
            [2.0, 0.0],
            # Robot 0 target: (2,0)
            [0.0, 2.0],
            # Robot 1 target: (0,2)
            [-2.0, 0.0],
            # Robot 2 target: (-2,0)
            [0.0, -2.0]
            # Robot 3 target: (0,-2)
        ]
        # Cycle the four canonical poses when more than four agents are used,
        # otherwise EnvCore(num_agents > 4) raised IndexError at construction.
        self.robots = [
            VirtualTurtlebot(i,
                             list(initial_poses[i % len(initial_poses)]),
                             list(target_poses[i % len(target_poses)]))
            for i in range(self.agent_num)
        ]

        # Environment reset
        # Reset all robots to initial state

    def reset(self):
        """Reset environment state"""
        self._init_robots()
        self.current_step = 0
        self.trajectory_lengths = [[] for _ in range(self.agent_num)]

        # Reset paths
        self.path_x = []
        self.path_y = []
        # self.initial_poses = [0.0, 0.0]
        # Reference replacement if initial position variables exist
        self.initial_poses = [robot.pose[:2] for robot in self.robots]

        return self._get_observations()

    # Returns a list containing initial observations for all robots

    # Observation generation
    def _get_observations(self):
        """Generate normalized observation data"""
        obs_list = []
        for robot in self.robots:
            # Update LiDAR data
            robot.laser_scan = self._simulate_laser(robot)

            # Construct observation vector
            target_dx = robot.target[0] - robot.pose[0]
            target_dy = robot.target[1] - robot.pose[1]

            obs = np.concatenate([
                robot.laser_scan / 3.0,
                # 24-dimensional LiDAR data: detection range 0.1-3.0m, normalized to [0,1]
                [robot.pose[0] / 5.0, robot.pose[1] / 5.0,
                    # Position normalized to ±5m range
                    target_dx / 5.0, target_dy / 5.0]
                # Relative target position normalization
                # 4-dimensional state info: normalized self-position and relative target position
            ])
            obs_list.append(obs)
        return obs_list

# LiDAR Simulation

    def _simulate_laser(self, robot):
        """Simulate high-precision LiDAR scanning"""
        ranges = []
        angles = np.linspace(-pi / 2, pi / 2, 24)
        # Scanning angle range: -90° to 90°
        for angle in angles:
            min_dist = 3.0
            # Maximum detection range: 3m
            current_angle = robot.pose[2] + angle

            # Obstacle detection
            for obs in self.obstacles:
                dx = obs['pos'][0] - robot.pose[0]
                dy = obs['pos'][1] - robot.pose[1]
                dist = sqrt(dx ** 2 + dy ** 2) - obs['radius']
                angle_diff = atan2(dy, dx) - current_angle
                if abs(angle_diff) < pi / 24:
                    # Detection precision: 7.5°
                    min_dist = min(min_dist, dist)

            # Inter-robot detection
            for other in self.robots:
                if other.id == robot.id: continue
                dx = other.pose[0] - robot.pose[0]
                dy = other.pose[1] - robot.pose[1]
                dist = sqrt(dx ** 2 + dy ** 2) - 0.2
                # Robot radius: 0.2 meters
                angle_diff = atan2(dy, dx) - current_angle
                if abs(angle_diff) < pi / 24:
                    min_dist = min(min_dist, dist)

            ranges.append(max(0.1, min_dist))
            # Minimum detection distance: 0.1 meters
        return np.array(ranges)

    def step(self, actions):
        """Execute one environment timestep"""
        if actions is None:
            raise ValueError("Actions must be provided (got None).")
        # Verify action length matches number of robots
        if len(actions) != self.agent_num:
            raise ValueError(f"Expected {self.agent_num} actions, got {len(actions)}")

        self.current_step += 1

        # 1. Execute actions
        for i, robot in enumerate(self.robots):
            action = actions[i]
            # Clip action values to valid ranges
            robot.linear_vel = np.clip(action[0], -self.max_speed, self.max_speed)
            robot.angular_vel = np.clip(action[1], -self.max_rotate, self.max_rotate)
            # Update robot pose
            robot.update_pose()

        # Track trajectory length by index.  Trajectory points themselves are
        # appended by ``update_trajectory`` below as strict 2-D [x, y] pairs;
        # appending the raw 3-element pose here used to mix 2-D and 3-D points
        # and broke every trajectory plot.
        for idx, robot in enumerate(self.robots):
            if len(self.trajectory_lengths[idx]) < self.max_steps:
                self.trajectory_lengths[idx].append(len(robot.trajectory))
            else:
                print(f"Robot {idx} terminated early, no more trajectory data!")

        # 2. Get observations
        obs = self._get_observations()

        # Initialize reward and termination lists
        rewards = [[0.0] for _ in range(self.agent_num)]
        dones = [False] * self.agent_num

        # Calculate rewards and termination conditions
        for i, robot in enumerate(self.robots):
            dx = robot.target[0] - robot.pose[0]
            dy = robot.target[1] - robot.pose[1]
            distance = np.linalg.norm([dx, dy])  # Simplified using vector norm

            # ----------------- Reward Calculation -----------------
            # 1. Base navigation reward (higher when closer to target)
            navigation_reward = self.max_initial_distance - distance

            # 2. Time penalty per step
            time_penalty = -0.05

            # 3. Collision detection and penalty
            collision_level = self._check_collision(robot)
            collision_penalty = 0.0
            if collision_level == 1:  # Actual collision
                collision_penalty = -10.0
                robot.linear_vel = 0.0  # Stop movement
                robot.angular_vel = 0.0
            elif collision_level == 2:  # Warning distance
                collision_penalty = -5.0

            robot.update_trajectory()  # Record trajectory

            # 4. Energy consumption penalty
            energy = -0.1 * (abs(robot.linear_vel) + abs(robot.angular_vel))

            # 5. Success bonus for reaching target
            success_bonus = 20.0 if distance < self.success_threshold else 0.0

            # Combine total reward
            total_reward = (
                    navigation_reward
                    + time_penalty
                    + collision_penalty
                    + energy
                    + success_bonus
            )
            rewards[i][0] = total_reward  # Update reward for correct robot index

            # ----------------- Termination Conditions -----------------
            # Timeout condition
            if self.current_step >= self.max_steps:
                dones = [True] * self.agent_num

            # Early stopping if over 80% steps and not near target
            early_stop = (self.current_step > self.max_steps * self.early_stop_ratio)
            early_stop = early_stop and (distance > self.success_threshold)

            done = False
            if collision_level == 1 or distance < self.success_threshold or early_stop:
                done = True

            # Additional success bonus
            if distance < self.success_threshold:
                total_reward += 10.0
                success_bonus = 10.0
            else:
                success_bonus = 0.0

            # Update termination status
            # if collision_level == 1:
            #     dones.append(True)
            # elif distance < self.success_threshold:
            #     dones.append(True)
            # else:
            #     dones.append(False)

            # Anti-pseudo-learning mechanism
            if all(r < -5 for r in sum(rewards, [])):
                dones = [True] * self.agent_num
                print("Triggered anti-pseudo-learning protection!")

            # Update rewards and done status by index
            rewards[i][0] = total_reward
            dones[i] = done

            # Record reward components for visualization/tuning
            if not hasattr(robot, 'reward_components'):
                robot.reward_components = {
                    'navigation': [],
                    'time_penalty': [],
                    'energy': [],
                    'collision': [],
                    'success': []
                }
            robot.reward_components['navigation'].append(navigation_reward)
            robot.reward_components['time_penalty'].append(time_penalty)
            robot.reward_components['energy'].append(energy)
            robot.reward_components['collision'].append(collision_penalty)
            robot.reward_components['success'].append(success_bonus)

            # Path recording
            # self.path_x.append(self.initial_poses[0])
            # self.path_y.append(self.initial_poses[1])
            for robot in self.robots:
                self.path_x.append(robot.pose[0])
                self.path_y.append(robot.pose[1])

            if all(dones):
                print("All robots terminated, stopping trajectory updates!")

        print(f"Current step: {self.current_step}, rewards length: {len(rewards)}, robot count: {self.agent_num}")
        print(f"Trajectory lengths: {[len(t) for t in self.trajectory_lengths]}")
        return obs, rewards, dones, [{}] * self.agent_num

    def _check_collision(self, robot):
        """Comprehensive collision detection"""
        """Returns collision level:
                0 - No collision
                1 - Actual collision
                2 - Danger zone warning"""
        # Obstacle collision detection
        for obs in self.obstacles:
            dx = obs['pos'][0] - robot.pose[0]
            dy = obs['pos'][1] - robot.pose[1]

            if np.linalg.norm([dx, dy]) < (obs['radius'] + 0.2):
                return 1  # Collision with obstacle

        # Inter-robot collision detection
        for other in self.robots:
            if other.id == robot.id:
                continue

            dx = other.pose[0] - robot.pose[0]
            dy = other.pose[1] - robot.pose[1]
            dist = np.linalg.norm([dx, dy])

            if dist < 0.4:
                # Actual collision threshold (sum of robot radii)
                return 1
            elif dist < 0.8:
                # Danger zone warning threshold
                return 2

        return 0  # No collision detected

# -------- Visualization Section --------

    def render(self, mode="human"):
        """Render the current fleet state.

        Artists are rebuilt on every call.  The old implementation cleared the
        axes and then called ``set_data`` on ``Line2D`` objects that ``clear()``
        had already discarded, and opened a second blocking ``plt.show()``
        window on each step.
        """
        if plt is None or not self.render_mode:
            return None
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        ax = self.ax
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        for obs in self.obstacles:
            center = tuple(float(v) for v in obs['pos'])
            ax.add_patch(plt.Circle(center, obs['radius'], color='gray', alpha=0.7))

        for robot in self.robots:
            points = [p[:2] for p in robot.trajectory if len(p) >= 2]
            if points:
                traj = np.asarray(points, dtype=float)
                ax.plot(traj[:, 0], traj[:, 1], lw=2, label=f'Robot {robot.id}')
            ax.scatter(robot.pose[0], robot.pose[1], s=100, label=f'Robot {robot.id} position')
            ax.scatter(robot.target[0], robot.target[1], marker='*', s=120,
                       label=f'Target {robot.id}')

        ax.set_title("Robot Trajectories")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize="small")
        self.fig.canvas.draw_idle()
        if mode == "rgb_array":
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            return buf[:, :, :3].copy()
        plt.pause(0.01)  # non-blocking refresh
        return None

    def visualize_learning(self, output_path=None, show=None):
        """Generate and save the 2x2 learning-curve figure.

        :param output_path: where to write the PNG.  Defaults to the fixed
            workspace location ``ROS_ws/results/figures/learning_curve.png``.
            The old implementation used the bare relative name
            ``'learning_curve.png'``, so the file landed in whatever directory
            the process happened to be started from -- which is why it was hard
            to find.
        :param show: open an interactive window.  ``None`` auto-detects a
            non-interactive (Agg) backend and skips it, so headless runs never
            block on ``plt.show()``.
        :returns: absolute path of the saved PNG, or ``None`` when matplotlib
            is unavailable.
        """
        if plt is None:
            print("matplotlib is not installed; skipping learning-curve plot.")
            print("Install it with: python -m pip install matplotlib")
            return None
        if output_path is None:
            output_path = (Path(__file__).resolve().parents[3]
                           / "results" / "figures" / "learning_curve.png")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if show is None:
            show = plt.get_backend().lower() not in {
                "agg", "pdf", "svg", "ps", "cairo", "template"
            }
        plt.figure(figsize=(12, 8))

        # 1. Total reward plot (using different colors)
        plt.subplot(2, 2, 1)
        colors = ['red', 'green', 'blue', 'orange']  # Distinct colors for each robot
        for i, robot in enumerate(self.robots):
            # Calculate total rewards by summing all components
            total_rewards = [
                sum(r) for r in zip(
                    robot.reward_components['navigation'],
                    robot.reward_components['time_penalty'],
                    robot.reward_components['energy'],
                    robot.reward_components['collision'],
                    robot.reward_components['success']
                ) if len(r) > 0]

            # Plot only if data exists
            if len(total_rewards) > 0:
                plt.plot(total_rewards, color=colors[i], label=f'Robot {i}', linewidth=2)
            else:
                print(f"Warning: Robot {i} has no total_rewards data!")

        plt.title("Total Reward Over Time")
        plt.legend()

        # 2. Reward component breakdown (bar chart)
        plt.subplot(2, 2, 2)
        components = ['navigation', 'time_penalty', 'energy', 'collision', 'success']
        for comp in components:
            # Calculate average for each component across all robots
            avg_values = []
            for robot in self.robots:
                data = robot.reward_components[comp]
                avg_values.append(np.mean(data) if len(data) > 0 else 0.0)  # Default to 0 if empty

            # Calculate global average (ignoring NaN values)
            valid_avgs = [v for v in avg_values if not np.isnan(v)]
            global_avg = np.mean(valid_avgs) if len(valid_avgs) > 0 else 0.0

            plt.bar(comp, global_avg)
        plt.title("Average Reward Components")
        plt.ylabel("Average Reward")

        # 3. Trajectory length statistics.  All agents advance one trajectory
        # point per step, so their curves are identical; drawing four
        # overlapping lines with a four-entry legend is misleading.
        plt.subplot(2, 2, 3)
        steps = range(self.current_step)
        series = [self.trajectory_lengths[idx] for idx in range(self.agent_num)
                  if len(self.trajectory_lengths[idx]) == len(steps)]
        if series:
            all_identical = all(s == series[0] for s in series[1:])
            if all_identical and len(series) > 1:
                plt.plot(steps, series[0], color='tab:red',
                         label=f'All {len(series)} robots (identical)')
            else:
                for idx, values in enumerate(series):
                    plt.plot(steps, values,
                             label=f'Robot {idx}',
                             linestyle='--' if idx % 2 == 0 else '-')
        plt.xlabel("Time Steps")
        plt.ylabel("Trajectory Length (steps)")
        plt.title("Trajectory Length Progression")
        plt.legend()

        # 4. Complete trajectory visualization
        plt.subplot(2, 2, 4)
        for robot in self.robots:
            if len(robot.trajectory) == 0:
                continue
            # Convert to 2D array ensuring proper format
            traj = np.array([[p[0], p[1]] for p in robot.trajectory if len(p) == 2])
            plt.plot(traj[:, 0], traj[:, 1], 'o-',
                     markersize=3,
                     linewidth=1,
                     label=f'Robot {robot.id}')
        plt.title("Complete Trajectories")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend()

        # Finalize and save to the deterministic workspace location.
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120)
        print(f"[EnvCore] learning curve saved to: {output_path}")
        if show:
            plt.show()
        else:
            plt.close('all')
        return str(output_path)

    def close(self):
        """ Clean up resources in the environment """
        print("Closing environment and releasing resources...")
        if plt is not None:
            plt.close('all')
        # Close all matplotlib Windows to prevent too many from taking up memory

if __name__ == '__main__':
    env = EnvCore(num_agents=4)
    obs = env.reset()

    print("===== Environment Initialization Completed =====")
    print(f"Robot count: {env.agent_num}")
    print(f"Obstacle configuration: {[o['pos'] for o in env.obstacles]}")
    print("Initial robot states:")
    for i, robot in enumerate(env.robots):
        print(f"  Robot {i}: Position({robot.pose[0]:.2f}, {robot.pose[1]:.2f}) "
              f"Heading: {robot.pose[2] / pi * 180:.1f}° Target({robot.target[0]:.1f}, {robot.target[1]:.1f})")
    print("=" * 50 + "\n")

    for _ in range(env.max_steps):
        actions = [np.random.uniform(-1, 1, 2) for _ in range(env.agent_num)]
        # Example policy:
        # actions = [
        #     [1.0, 0.0] if robot.pose[0] < robot.target[0] else [-1.0, 0.0]
        #     for robot in env.robots
        # ]

        obs, rewards, dones, _ = env.step(actions)

        # env.render()  # Enable real-time rendering

        # Print step header
        print(f"\n===== Step {env.current_step} =====")
        print(f"Remaining steps: {env.max_steps - env.current_step}")

        # Detailed status output
        for robot_id, robot in enumerate(env.robots):
            # Calculate key metrics
            dx, dy = np.array(robot.target) - np.array(robot.pose[:2])
            distance = sqrt(dx ** 2 + dy ** 2)
            min_laser = np.min(robot.laser_scan)

            # Action analysis
            action_str = f"Linear: {robot.linear_vel:.2f} m/s, Angular: {robot.angular_vel:.2f} rad/s"

            # Status indicators
            status = []
            if distance < env.success_threshold:
                status.append("Target reached")
            if env._check_collision(robot):
                status.append("Collision detected")
            status = status or ["Operating"]

            # Build output information
            print(f"Robot {robot_id}:")
            print(f"  Current position: ({robot.pose[0]:.2f}, {robot.pose[1]:.2f}) Heading: {robot.pose[2] / pi * 180:.1f}°")
            print(f"  Target distance: {distance:.2f}m  Nearest obstacle: {min_laser:.2f}m")
            print(f"  Current action: {action_str}")
            print(f"  Status: {' | '.join(status)}")
            print(f"  Reward value: {rewards[robot_id][0]:.2f} (Navigation: {1 / (1 + distance):.2f} "
                  f"Energy penalty: {-0.1 * (abs(robot.linear_vel) + abs(robot.angular_vel)):.2f})")
            print("-" * 40)

            # Termination condition check
            if all(dones):
                print("\n!!! All robots terminated !!!")
                print("Termination reasons:")
                for r_id, r in enumerate(env.robots):
                    if np.linalg.norm(np.array(r.target) - np.array(r.pose[:2])) < env.success_threshold:
                        print(f"  Robot {r_id}: Successfully reached target")
                    elif env._check_collision(r):
                        print(f"  Robot {r_id}: Collision occurred")
                break
        else:
            print(f"\n!!! Maximum steps {env.max_steps} reached without completing all tasks !!!")

        # Final status summary
        print("\n===== Training Summary =====")
        print(f"Total steps executed: {env.current_step}")
        for robot_id, robot in enumerate(env.robots):
            final_dist = np.linalg.norm(np.array(robot.target) - np.array(robot.pose[:2]))
            # Correct calculation
            collision = "Yes" if env._check_collision(robot) else "No"
            print(f"Robot {robot_id}: Final distance {final_dist:.2f}m, Collision: {collision}")

    env.visualize_learning()
    # Generate learning curves
    plt.show()
    # Display all plots
