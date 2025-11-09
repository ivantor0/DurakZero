# DurakZero

DurakZero 将原始 DouZero 强化学习框架迁移到两人 36 张牌的杜拉克（Durak）游戏。本仓库实现了杜拉克的完整规则、自我博弈训练与简单的自博弈评估工具。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 训练

```bash
python train.py
```

常用参数：

* `--gpu_devices`：可见的 GPU ID。
* `--num_actor_devices`：用于自博弈模拟的设备数量。
* `--num_actors`：每个设备创建的 actor 数量。
* `--training_device`：学习器所在的设备（或 `cpu`）。
* `--actor_device_cpu`：强制 actor 运行在 CPU 上。

更多参数请查看 `douzero/dmc/arguments.py`。

### Notebook 工作流

仓库提供了 `notebooks/durakzero_training.ipynb`，用于交互式地配置训练参数、启动自博弈以及评估生成的模型，方便在本地或如 H100 等 GPU 服务器上运行完整流程。

若希望在真实在线对局中获得模型建议，可以使用：

* `tools/durakzero_live_helper.sh`：终端脚本，可选择通过 `pyshark` 或日志文件捕获数据流，并在命令行中输出 DurakZero 的实时推荐及对局日志。
* `tools/durak_live_helper.py`：命令行版本的实时助手（运行 `python tools/durak_live_helper.py --help` 查看参数）。

## 评估

```bash
python evaluate.py --checkpoint path/to/model.tar --num_games 1000
```

评估脚本会输出双方的胜场和胜率。

## 目录结构

* `douzero/env/env.py`：杜拉克规则与特征编码。
* `douzero/dmc/`：深度蒙特卡洛训练组件。
* `douzero/evaluation/`：模型加载与自博弈评估。
* `train.py`：训练入口。
* `evaluate.py`：评估入口。

项目沿用了 DouZero 的 Apache 2.0 许可证。
