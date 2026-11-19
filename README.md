# DOMASMRL — 交付验收手册

**DOMAS 多机器人强化学习（MAPPO）ROS / Gazebo 工程**

| 项目 | 内容 |
|---|---|
| 公开仓库 | https://github.com/RongzePan/DOMASMRL |
| 默认分支 | `main` |
| 文档版本 | 2026-11-19 |
| 基线提交 | `4f6179a` — *Initial commit: complete DOMAS-MAPPO ROS workspace* |
| 验收结论 | 离线构建验证 **38/38 PASS，0 失败**；真实 CMake configure + build + install 三个 catkin 包全部通过 |

> 本手册面向交付验收，覆盖环境配置、快速构建、启动/终止、报告位置、测试覆盖、实测结果、源码目录、构建系统、Windows 主方案、macOS 备选方案与故障排查。所有命令均以**仓库根目录**为当前目录。

---

## 1. 项目概述

DOMASMRL 是一个基于 **ROS Melodic + Gazebo + TurtleBot3** 的四机器人一致性/编队控制与 **MAPPO（Multi-Agent PPO）** 强化学习工作区。仓库包含三个一等 catkin 包：

| 包 | 职责 |
|---|---|
| `mappo` | MAPPO 训练栈：actor/critic 网络、PPO trainer、双 replay buffer、shared/separated runner、虚拟 TurtleBot 环境、训练与渲染入口 |
| `consensus` | ROS 编队控制节点、四机器人 gym-like 环境、launch/RViz 配置 |
| `sim_world` | Gazebo 世界（11 个 `.world`）与模型（parking block / AprilTag dock / target / wall） |

技术栈：Python 3、NumPy、PyTorch（训练期）、catkin/CMake、Gazebo SDF、RViz YAML、Bash。

### 1.1 交付范围

- 三个一等包的全部源码、包清单、CMake 安装规则与 `setup.py`
- 工作区顶层 `src/CMakeLists.txt`、`ROS_ws/models/` 资源、`ROS_ws/results/figures/` 实验图
- 离线构建验证链：`tools/verify_build.py`、`tools/catkin_shim/`、`tools/python_preflight.sh`、`tools/plot_learning_curve.py`
- 本手册

### 1.2 明确不在本次交付范围（验收边界）

- **真实 ROS/Gazebo 端到端仿真**：需要目标机 Ubuntu 18.04 + ROS Melodic + GPU/PyTorch，见 §2.4
- **vendored `turtlebot3*` 包**：按仓库既有 `.gitignore` 约定排除（两个目录各带嵌套 `.git`，合计约 137 MB），克隆后需按 §3.3 补装
- **vendored C++ 目标编译**：依赖真实 ROS 头文件/库，离线验证仅做元数据与资产校验
- **PyTorch 前向/反向训练**：离线验证使用依赖桩完成 49 个模块的导入冒烟，不执行数值训练

---

## 2. 环境配置

### 2.1 组件矩阵

| 组件 | 离线验证 | 真实训练 | 真实仿真 | 实测版本 |
|---|:---:|:---:|:---:|---|
| Python | 必需 | 必需 | 必需 | 3.15.2 / 3.13.14 |
| NumPy | 必需 | 必需 | 必需 | 2.5.4 |
| PyYAML | 可选（缺失则降级为结构检查） | 可选 | 可选 | 6.0.3 |
| setuptools | 可选 | 必需 | 必需 | 84.2.1 |
| cmake | 可选（缺失则 SKIP 真实构建） | 不需要 | 不需要 | 4.4.4 |
| Bash | 可选（缺失则 SKIP shell 检查） | 必需（launcher） | 不需要 | 5.3.15 (Git for Windows) |
| matplotlib | 不需要 | 不需要 | 不需要 | 3.11.2（绘图用） |
| PyTorch | 不需要（依赖桩） | 必需 | 必需 | 未安装，需 `pip install torch` |
| ROS Melodic / Gazebo | 不需要 | 不需要 | 必需 | 目标机 Ubuntu 18.04 |

> 验证脚本对可选依赖采用 **SKIP 而非 FAIL** 策略；缺依赖时结果仍为 PASS，但会明确列出跳过项并给出安装命令。

### 2.2 Windows 主方案（推荐，已实测）

这是本项目的开发、验证与交付基线环境。

**前置组件**

1. Python 3.13 或更高（实测 3.15.2 与 3.13.14 均可）
2. Git for Windows（提供 `bash`；Git Bash / PortableGit 均可）
3. 无需 Visual Studio：`project(<pkg> LANGUAGES NONE)` 使三个纯 Python/data 包不要求 C/C++ 编译器；若 IDE 精简环境导致默认生成器不可用，验证脚本自动回退到 VS BuildTools 自带的独立 `ninja.exe`
4. 无需 ROS、Gazebo、PyTorch 即可完成离线验证

**安装**

```bash
git clone https://github.com/RongzePan/DOMASMRL.git
cd DOMASMRL
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r ROS_ws\tools\requirements-verify.txt
```

`requirements-verify.txt` 内容：`cmake>=3.20`、`PyYAML>=6.0`、`setuptools>=60`、`numpy>=1.24`。

可选扩展：

```bash
.venv\Scripts\python.exe -m pip install matplotlib      # 生成学习曲线图
.venv\Scripts\python.exe -m pip install torch numpy     # 真实 MAPPO 训练
```

**网络注意**：若本机代理仅放行 `api.github.com`，对 `github.com` 的 CONNECT 会返回 502（表现为 `schannel: server closed abruptly`）。本仓库已固化直连配置：

```bash
git config --local http.proxy ""
git config --local http.version HTTP/1.1
```

新克隆的机器如遇同类问题，执行上面两行即可。

### 2.3 macOS 备选方案

离线验证链在 macOS 上可完整通过：`/bin/bash`、Homebrew CMake、Unix Makefiles/Ninja 生成器、`LANGUAGES NONE` 均无需 Windows 组件。

```bash
brew install git cmake python@3.11
git clone https://github.com/RongzePan/DOMASMRL.git && cd DOMASMRL
python3 -m venv .venv
.venv/bin/python -m pip install -r ROS_ws/tools/requirements-verify.txt
.venv/bin/python ROS_ws/tools/verify_build.py
```

**ROS 1 Melodic 官方仅支持 Ubuntu 18.04**，macOS 无法原生运行 `roscore` / `catkin_make` / Gazebo。真实仿真需借助 Linux 容器或虚拟机：

```bash
# 在 macOS 上启动 Ubuntu 18.04 容器（挂载本仓库），随后在容器内按 ROS 官方流程安装 Melodic
docker run --rm -it -v "$PWD":/workspace ubuntu:18.04 bash
# 容器内：参考 https://wiki.ros.org/melodic/Installation/Ubuntu
# 安装 ros-melodic-ros-base、gazebo_ros、turtlebot3 相关包后执行 §3.2
```

Gazebo 图形界面需 XQuartz / X11 转发；无图形需求可用 `headless:=true` 的 launch 参数。此路径属于**环境准备工作**，不计入离线构建验证结果。

### 2.4 目标机：Ubuntu 18.04 + ROS Melodic（真实仿真）

```bash
sudo apt install ros-melodic-ros-base ros-melodic-gazebo-ros \
                 ros-melodic-turtlebot3-description ros-melodic-xacro \
                 ros-melodic-robot-state-publisher ros-melodic-rviz python3-pip
source /opt/ros/melodic/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/ROS_ws/src/sim_world/model
```

依赖声明见各包 `package.xml`：`consensus` 依赖 `rospy`、`tf`、`std_msgs`、`geometry_msgs`、`gazebo_msgs`、`std_srvs`、`gazebo_ros`、`robot_state_publisher`、`rviz`、`xacro`、`turtlebot3_description`、`mappo`、`sim_world`；`sim_world` 依赖 `gazebo_ros`。

---

## 3. 快速构建

### 3.1 离线验证（Windows / macOS 通用，推荐验收入口）

```bash
# Windows
.venv\Scripts\python.exe ROS_ws\tools\verify_build.py

# macOS / Linux
.venv/bin/python ROS_ws/tools/verify_build.py
```

退出码 `0` 表示全部检查通过；报告写入 `ROS_ws/build/verify/verification_report.json`。该命令会：

1. 字节编译 55 个 Python 源文件
2. 校验 162 个 XML/launch/SDF/world/config、36 个 YAML/RViz、2 个 shell 脚本
3. 校验 13 个 `package.xml`（含 vendored 包）与两个 `setup.py`
4. 用 `tools/catkin_shim/cmake/catkinConfig.cmake` 对 `mappo`、`consensus`、`sim_world` 执行**真实 CMake configure → build → install**
5. 校验安装树 12 项清单（100 个安装文件）
6. 带 ROS/torch 依赖桩导入 49 个模块，并跑 10 项功能性自测

### 3.2 真实 catkin 构建（Ubuntu 18.04）

```bash
source /opt/ros/melodic/setup.bash
cd ROS_ws
catkin_make
source devel/setup.bash
```

工作区顶层 `src/CMakeLists.txt` 已在 configure 期解析 `/opt/ros/melodic/share/catkin/cmake/toplevel.cmake`，避免 Git-on-Windows 符号链接残留导致 `catkin_make` 失败。

### 3.3 第三方 ROS 依赖补装（克隆后必需）

`ROS_ws/src/turtlebot3*` 未随本仓库分发，完整 catkin 工作区需补装：

```bash
cd ROS_ws/src
git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ../.. && rosdep install --from-paths src --ignore-src -r -y
```

> 分支以目标 ROS 发行版为准；本工程基线为 `melodic-devel`。仅做离线验证时无需此步骤。

---

## 4. 启动与终止

| 场景 | 启动命令 | 终止方式 |
|---|---|---|
| 离线构建验证 | `python ROS_ws/tools/verify_build.py` | 一次性任务；`Ctrl+C` 可中断 |
| 运行时预检（不启动训练） | `bash ROS_ws/src/mappo/train_mpe.sh --check` | 自动退出 |
| 生成学习曲线图 | `python ROS_ws/tools/plot_learning_curve.py --steps 100 --seed 0` | 自动退出 |
| MAPPO 训练（需 torch） | `bash ROS_ws/src/mappo/train_mpe.sh` | `Ctrl+C`；必要时 `pkill -f mappo.train` |
| 多 seed 训练（legacy） | `bash ROS_ws/src/mappo/scripts/train_mpe.sh`（`SEED_MAX=N`） | `Ctrl+C` |
| 四机器人 Gazebo 仿真 | `roslaunch consensus 4turtlebot.launch` | 见下方终止流程 |
| 仿真 + 控制 + 训练合启 | `roslaunch consensus combined_launch_file.launch` | 见下方终止流程 |

**训练参数**：`NUM_AGENTS`（默认 4）、`OBS_DIM`（28）、`ACT_DIM`（2）、`CONFIG_PATH`（`src/mappo/config/train_config.yaml`），也可向 launcher 追加 `train.py` 原生参数。配置覆盖规则：命令行显式参数优先于 YAML 默认值。

**启动前检查**：`train_mpe.sh --check` 会清理无效的 `PYTHONHOME`、探测解释器健康度、按解释器 `os.pathsep` 构造 `PYTHONPATH`，并报告缺失依赖。缺 torch 时明确提示 `python -m pip install numpy torch`，不会出现裸 traceback。

**ROS/Gazebo 终止流程**（Ubuntu）：

```bash
# 1) 前台 launch：Ctrl+C
rosnode kill -a                          # 2) 停止所有节点
killall -9 roscore gzserver gzclient    # 3) 清理残留主进程与仿真进程
pkill -f mappo.train                     # 4) 清理训练子进程
```

---

## 5. 报告与产物位置

| 产物 | 路径 | 说明 |
|---|---|---|
| **构建验证报告（主交付物）** | `ROS_ws/build/verify/verification_report.json` | 机器可读；字段见下表 |
| 控制台报告 | stdout | 每阶段 `[PASS]/[SKIP]/[FAIL]` 明细与末尾汇总 |
| CMake 安装树 | `ROS_ws/build/verify/install/` | 三个包安装结果，共 100 个文件 |
| Python 字节码缓存 | `ROS_ws/build/verify/pyc/` | 验证中间产物 |
| **实验学习曲线图** | `ROS_ws/results/figures/learning_curve.png` | 2×2 四宫格，约 118 KB，固定输出位置 |
| 历史同名图 | `ROS_ws/src/mappo/envs/learning_curve.png` | 仓库原有历史产物 |
| 训练产物 | `ROS_ws/results/<env>/<scenario>/<algo>/<exp>/runN/` | `logs/`（TensorBoard）、`models/`（`.pt`）、`gifs/render.gif`；需真实训练后生成 |

`verification_report.json` 顶层字段：

| 字段 | 含义 |
|---|---|
| `workspace` / `timestamp` / `python` / `executable` | 运行上下文，便于复现 |
| `total_checks` / `passed_checks` / `failed_checks` / `skipped_checks` | 检查计数 |
| `result` | `PASS`（`failed_checks == 0`）或 `FAIL` |
| `checks[]` | 每项含 `stage`、`name`、`ok`、`status`（PASS/FAIL/SKIP）、`detail` |

---

## 6. 测试覆盖

共 **38 项检查 / 8 个阶段**（`stage` 名称与报告一致）：

| 阶段 | 项数 | 覆盖内容 |
|---|---:|---|
| `1-python-compile` | 1 | 55 个 Python 源文件 `py_compile` |
| `1b-python-preflight` | 5 | 坏 `PYTHONHOME` 复现、清理后恢复 `mappo` 导入、3 个 launcher 回归 |
| `2-assets` | 3 | 162 个 XML/launch/SDF/world/config、36 个 YAML/RViz、2 个 shell (`bash -n`) |
| `3-catkin-metadata` | 5 | 3 个一等包、顶层 CMake、13 个 manifest、安装目标引用、两个 `setup.py` 实执行 |
| `4-references` | 2 | 12 个 `$(find ...)`、12 个 `model://` URI 可解析性 |
| `5-cmake-build` | 11 | cmake 可用性 + 3×(configure / build / install) + 12 项安装清单 |
| `6-import-smoke` | 1 | 49 个模块导入（ROS/torch 依赖桩） |
| `7-functional` | 10 | 配置解析与 YAML 覆盖、双 EnvCore、向量化包装、RMAPPOAgent、环境适配器、数值工具、双 replay buffer |

**未覆盖（明确边界）**：真实 ROS 节点运行时、Gazebo 物理仿真、vendored C++ 目标编译、PyTorch 前向/反向与收敛性、多机分布式通信。这些依赖目标机环境，见 §1.2。

---

## 7. 实测验证结果

### 7.1 验收基线（2026-11-19，Windows 11 / Python 3.15.2 venv）

| 指标 | 实测值 |
|---|---|
| 验证命令 | `.venv\Scripts\python.exe ROS_ws\tools\verify_build.py` |
| 结果 | **PASS — 38 passed / 0 skipped / 0 failed，退出码 0** |
| Python / NumPy / cmake / PyYAML / setuptools | 3.15.2 / 2.5.4 / 4.4.4 / 6.0.3 / 84.2.1 |
| Python 源码编译 | 55 / 55 |
| XML / YAML / shell 资产 | 162 / 36 / 2，全部有效 |
| catkin manifest | 13 个，名称与依赖一致 |
| CMake 构建 | mappo / consensus / sim_world 三个包 configure + build + install 全通过 |
| 安装树 | 100 个文件，12 项清单核对通过 |
| 模块导入 / 功能自测 | 49 / 10，全部通过 |
| 报告 | `ROS_ws/build/verify/verification_report.json` |

### 7.2 兼容性回归（同日）

| 场景 | 结果 | 说明 |
|---|---|---|
| A. 项目 venv Python 3.15.2（依赖齐全） | **PASS** 38/38，0 SKIP | 主验收场景 |
| B. Python 3.13（无 numpy/PyYAML/setuptools/cmake） | **PASS** 26 passed / 4 skipped / 0 failed | 缺依赖时优雅降级，明确列出 SKIP 与安装命令，不崩溃 |
| C. IDE 精简 PATH + Python 3.15.2 | **PASS** 38/38，0 SKIP | 无 Git Bash / VS 环境变量时，自动定位 bash 与 cmake，CMake 回退 Ninja 生成器 |

### 7.3 客户验收清单

- [x] 仓库可公开克隆，目录与文档一致（初始提交 171 个文件 + 本手册，分支 `main`）
- [x] 全量 Python 源码可字节编译，无语法错误
- [x] 所有 XML / YAML / shell 配置资产语法有效
- [x] catkin 元数据完整，安装目标引用的文件均存在
- [x] launch 与 Gazebo 模型 URI 引用全部可解析
- [x] 真实 CMake configure + build + install 三个一等包成功
- [x] 49 个模块可导入，10 项功能性自测通过
- [x] 验证报告为 `PASS`，`failed_checks = 0`
- [x] 实验图固定输出到 `ROS_ws/results/figures/`，可复现生成
- [x] Windows 主方案与 macOS 备选方案均有可执行命令
- [ ] 真实 ROS/Gazebo 端到端仿真与 MAPPO 收敛性 —— 需按 §2.4 / §3.3 在目标机执行，不在离线验证范围

---

## 8. 源码目录

```
DOMASMRL/
├── README.md                          # 本手册
├── .gitignore                         # 排除 venv / build / IDE / 训练产物 / vendored turtlebot3*
└── ROS_ws/                            # catkin 工作区
    ├── .catkin_workspace
    ├── src/
    │   ├── CMakeLists.txt             # 工作区顶层（configure 期解析 catkin toplevel.cmake）
    │   ├── mappo/                     # MAPPO 训练栈
    │   │   ├── package.xml / CMakeLists.txt / setup.py / config.py
    │   │   ├── config/train_config.yaml
    │   │   ├── algorithms/
    │   │   │   ├── algorithm/        # r_actor_critic, rMAPPOPolicy, r_mappo, rmappo_agent
    │   │   │   └── utils/            # act, cnn, mlp, rnn, distributions, popart, util
    │   │   ├── envs/                 # VirtualTurtlebot, env_core, env_continuous/discrete,
    │   │   │   ├── mpe/              #   env_wrappers, spaces_compat, learning_curve.png
    │   │   │   └── MPE_env.py
    │   │   ├── runner/{shared,separated}/   # base_runner, env_runner
    │   │   ├── utils/                # separated_buffer, shared_buffer, util, valuenorm, tensorboard
    │   │   ├── train/                # train.py, __main__.py  (python -m mappo.train)
    │   │   ├── scripts/render/       # render_mpe.py
    │   │   └── train_mpe.sh          # 训练 launcher
    │   ├── consensus/                # ROS 编队控制
    │   │   ├── MultipleTurtlebotsEnv.py / __init__.py
    │   │   ├── src/                  # control_robots.py, 兼容层
    │   │   ├── launch/               # 4turtlebot / combined_launch_file / one_turtle
    │   │   └── config/               # RViz 配置
    │   └── sim_world/                # Gazebo 世界与模型
    │       ├── world/                # 11 个 .world
    │       └── model/                # backup / dock / parkingblock / tag1 / target / wall
    ├── models/target/                # 工作区级 Gazebo target 模型
    ├── tools/
    │   ├── verify_build.py           # 构建验证主程序
    │   ├── catkin_shim/cmake/catkinConfig.cmake  # 离线 catkin CMake 桩
    │   ├── python_preflight.sh       # Python 运行环境预检
    │   ├── plot_learning_curve.py   # 可复现绘图
    │   └── requirements-verify.txt  # 验证依赖清单
    ├── build/verify/                 # 验证产物（不入库）
    │   ├── verification_report.json
    │   ├── cmake/<pkg>/             # CMake 构建目录
    │   └── install/                 # CMake 安装树
    └── results/                      # 实验产物（figures 入库，训练产物不入库）
        └── figures/learning_curve.png
```

---

## 9. 构建系统

工程采用**三层构建体系**，目标机与开发机互不干扰：

| 层 | 组件 | 用途 |
|---|---|---|
| 目标机（Ubuntu 18.04） | `/opt/ros/melodic/.../toplevel.cmake` + `catkin_make` + 各包 `CMakeLists.txt` + `setup.py` | 真实 catkin 构建、安装 Python 包与 launch/world/model |
| 开发机离线验证 | `tools/verify_build.py` + `tools/catkin_shim/cmake/catkinConfig.cmake` + CMake 4.4.4 | 无 ROS 时等价验证 configure / build / install |
| Python 包层 | `setup.py`（`package_dir={"<pkg>": "."}`）+ `catkin_python_setup()` | 包位于 catkin 包根目录的非标准布局 |

关键设计：

- `project(<pkg> LANGUAGES NONE)`：三个包均为纯 Python/data，不启用 C/C++ 语言， IDE 精简环境亦可配置。真实 catkin 顶层已启用语言，子目录此写法无副作用。
- `catkin_shim` 仅实现一等包用到的 `catkin_package` / `catkin_python_setup` / `catkin_install_python` 等宏，并校验被引用脚本与目录确实存在；**不替代目标机上的真实 catkin**。
- 安装规则含 `catkin_install_python(PROGRAMS ...)`、`install(PROGRAMS ...)`、`install(DIRECTORY ...)`；两个同名 `train_mpe.sh` 通过 `RENAME train_mpe_legacy.sh` 避免互相覆盖。
- 验证脚本自动定位 bash / cmake / ninja，默认生成器失败时回退 Ninja，缺工具时按 SKIP 记录而非崩溃。

---

## 10. 故障排查

| 现象 | 根因 | 处置 |
|---|---|---|
| `Fatal Python error: Failed to import encodings module` | `PYTHONHOME` 指向不完整 prefix，或 venv 基础解释器已删除 | Git Bash `unset PYTHONHOME`；PowerShell `Remove-Item Env:PYTHONHOME`；cmd `set PYTHONHOME=`；或退出/重建 venv。用 `bash tools/python_preflight.sh` 诊断 |
| 验证输出 `stage crashed` / `bash: command not found` | IDE 进程 PATH 不含 Git Bash | 安装 Git for Windows；脚本会自动搜索 PortableGit/Git 常见路径，仍找不到则 SKIP shell 检查 |
| `[SKIP] Full yaml.safe_load validation` | 该解释器无 PyYAML | `python -m pip install -r tools/requirements-verify.txt`（已降级为结构检查，不影响 PASS） |
| `[SKIP] Import smoke test / Functional smoke tests` | 无 numpy | 同上；numpy 已加入依赖清单 |
| `[SKIP] Real CMake configure+build+install` | 无 cmake 可执行文件 | 同上；脚本也会跨解释器搜索已安装的 cmake wheel |
| `nmake -? ... no such file` / VS 生成器失败 | 精简环境缺少 MSVC 环境变量 | 脚本自动回退 `-G Ninja` + VS BuildTools 自带 `ninja.exe` |
| `No module named 'mappo'` | `PYTHONPATH` 用 `:` 拼接（Windows Python 需要 `;`）或 POSIX 路径未转换 | 使用 launcher（`train_mpe.sh --check`）而非手工设置；launcher 按 `os.pathsep` 拼接并用 `cygpath -w` 转换 |
| `Missing training dependencies: torch` | 未安装 PyTorch（正常提示，非错误） | `python -m pip install numpy torch` |
| PowerShell 执行 `python3` 无输出 | `python3` 是 WindowsApps 占位程序 | 改用 `python`，或用 venv 解释器绝对路径 |
| `git push` 报 `schannel: server closed abruptly` / `CONNECT 502` | 代理仅放行 `api.github.com` | `git config --local http.proxy ""` 与 `git config --local http.version HTTP/1.1`（本仓库已固化） |
| `catkin_make` 找不到 `toplevel.cmake` | 顶层 CMakeLists 曾是 Git-on-Windows 符号链接残留 | 已改为 configure 期 `include(...)`；并确认已 `source /opt/ros/melodic/setup.bash` |
| launch 报找不到 world / model | 路径指向不存在的 `consensus/worlds/` 或 `model://dock` 缺失 | 已修正为 `$(find sim_world)/world/...`，并补齐 `sim_world/model/dock/`；设置 `GAZEBO_MODEL_PATH` |
| `TURTLEBOT3_MODEL` 未设置导致 xacro 失败 | `4turtlebot.launch` 读取该环境变量 | `export TURTLEBOT3_MODEL=burger` |
| 学习曲线图缺失 / `matplotlib` 报错 | 未安装 matplotlib 或旧版写到当前目录 | `pip install matplotlib`；图固定输出到 `ROS_ws/results/figures/learning_curve.png` |

---

## 11. 已知限制

1. **ROS 1 Melodic 仅官方支持 Ubuntu 18.04**：Windows 与 macOS 只能做离线验证；真实仿真须用 Linux 容器/虚拟机。
2. **vendored `turtlebot3*` 未随仓库分发**：完整 catkin 工作区需按 §3.3 补装。
3. **真实训练需 PyTorch**：离线验证以依赖桩覆盖导入路径，不验证训练收敛性。
4. **TensorBoard / GIF / 模型权重**：仅在真实训练后于 `ROS_ws/results/.../runN/` 下生成，当前仓库只含学习曲线图。
5. **仓库暂无 LICENSE 文件**：如需开源许可，请在验收前补充说明。

---

## 12. 命令速查

```bash
# 环境（Windows 主方案）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r ROS_ws\tools\requirements-verify.txt

# 验收（主入口）
.venv\Scripts\python.exe ROS_ws\tools\verify_build.py
# 报告
#   ROS_ws/build/verify/verification_report.json

# 运行时预检
bash ROS_ws/src/mappo/train_mpe.sh --check

# 生成实验图
.venv\Scripts\python.exe ROS_ws/tools/plot_learning_curve.py --steps 100 --seed 0
#   ROS_ws/results/figures/learning_curve.png

# 真实训练（需 torch）
bash ROS_ws/src/mappo/train_mpe.sh

# 目标机 ROS（Ubuntu 18.04）
source /opt/ros/melodic/setup.bash
export TURTLEBOT3_MODEL=burger
cd ROS_ws && catkin_make && source devel/setup.bash
roslaunch consensus combined_launch_file.launch
```

---

## 13. 版本记录

| 日期 | 变更 |
|---|---|
| 2026-11-14 | 工程补全与修复（编译错误、缺失模块、导入路径、catkin 元数据）；建立离线构建验证链；创建公开仓库并完成初始提交 `4f6179a` |
| 2026-11-14 | 修 `PYTHONHOME` 启动崩溃、Windows 路径分隔与 IDE 精简 PATH 兼容；统一实验产物到 `ROS_ws/results/` |
| 2026-11-19 | 补齐交付验收文档；`requirements-verify.txt` 增加 numpy；验证脚本在缺 numpy 时优雅降级；三场景实测全部 `0 failed` |
