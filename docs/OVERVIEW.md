# LIBERO-PRO 技术总览（OVERVIEW）

本文面向新成员与开发者，目标是让你在最短时间内理解 LIBERO-PRO 的工程架构、运行方式与二次开发路径。

---

## 第一部分：项目架构分析

### 1. 项目整体架构

#### 1.1 项目目录树（精简版）

<pre><code>LIBERO-PRO/
├─ README.md
├─ requirements.txt
├─ setup.py
├─ evaluation_config.yaml
├─ perturbation.py
├─ benchmark_scripts/
│  ├─ download_libero_datasets.py
│  ├─ check_task_suites.py
│  └─ render_single_task.py
├─ libero/
│  ├─ configs/
│  │  ├─ config.yaml
│  │  ├─ data/default.yaml
│  │  ├─ train/default.yaml
│  │  ├─ eval/default.yaml
│  │  ├─ lifelong/*.yaml
│  │  └─ policy/*.yaml
│  ├─ libero/
│  │  ├─ __init__.py
│  │  ├─ benchmark/
│  │  │  ├─ __init__.py
│  │  │  └─ libero_suite_task_map.py
│  │  ├─ bddl_files/
│  │  ├─ init_files/
│  │  ├─ assets/
│  │  └─ envs/
│  └─ lifelong/
│     ├─ main.py
│     ├─ evaluate.py
│     ├─ datasets.py
│     ├─ metric.py
│     ├─ utils.py
│     ├─ algos/
│     └─ models/
├─ libero_ood/
│  ├─ ood_environment.yaml
│  ├─ ood_object.yaml
│  ├─ ood_language.yaml
│  ├─ ood_spatial_relation.yaml
│  └─ ood_task.yaml
├─ scripts/
│  ├─ check_dataset_integrity.py
│  ├─ collect_demonstration.py
│  ├─ create_libero_task_example.py
│  ├─ get_dataset_info.py
│  ├─ config_copy.py
│  └─ create_template.py
└─ docs/
   ├─ CONTEXT.md
   └─ OVERVIEW.md
</code></pre>

#### 1.2 核心模块与职责

1. 配置与入口层
- [setup.py](setup.py): 注册命令行入口（lifelong.main、lifelong.eval 等）。
- [libero/configs/config.yaml](libero/configs/config.yaml): 全局 Hydra 主配置。
- [evaluation_config.yaml](evaluation_config.yaml): 扰动评测配置（单扰动或组合扰动开关）。

2. 基准任务与路径层
- [libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py): 基准注册、任务索引、bddl/init/demo 路径解析。
- [libero/libero/libero_suite_task_map.py](libero/libero/libero_suite_task_map.py): 各 suite 的任务清单映射。
- [libero/libero/__init__.py](libero/libero/__init__.py): 默认路径配置与 ~/.libero/config.yaml 初始化。

3. 训练与评估层
- [libero/lifelong/main.py](libero/lifelong/main.py): 终身学习训练主流程。
- [libero/lifelong/evaluate.py](libero/lifelong/evaluate.py): 加载 checkpoint 并做并行环境评估。
- [libero/lifelong/datasets.py](libero/lifelong/datasets.py): HDF5 轨迹读取、任务分组封装。
- [libero/lifelong/metric.py](libero/lifelong/metric.py): loss 与 success 指标计算。

4. 算法与策略层
- [libero/lifelong/algos](libero/lifelong/algos): Sequential、ER、AGEM、EWC、PackNet、Multitask 等。
- [libero/lifelong/models](libero/lifelong/models): BCRNNPolicy、BCTransformerPolicy、BCViLTPolicy。

5. OOD 扰动层
- [perturbation.py](perturbation.py): BDDL 解析与多类扰动器（environment/swap/object/language/task）。
- [libero_ood](libero_ood): 各类 OOD 扰动规则 YAML。

6. 运维与工具层
- [benchmark_scripts](benchmark_scripts): 数据下载、suite 可用性检查、单任务渲染。
- [scripts](scripts): 数据集完整性检查、数据采集、任务模板生成等。

#### 1.3 Mermaid 图：模块依赖关系

<pre class="mermaid">
flowchart LR
    A[Hydra Configs
    libero/configs] --> B[Training Entry
    libero/lifelong/main.py]
    A --> C[Eval Entry
    libero/lifelong/evaluate.py]

    B --> D[Benchmark Registry
    libero/libero/benchmark]
    C --> D

    D --> E[BDDL Files
    libero/libero/bddl_files]
    D --> F[Init States
    libero/libero/init_files]
    D --> G[Demo Datasets
    datasets path]

    B --> H[Datasets Wrapper
    lifelong/datasets.py]
    B --> I[Algo Layer
    lifelong/algos]
    I --> J[Policy Layer
    lifelong/models]

    K[Perturbation Config
    evaluation_config.yaml] --> L[Perturbation Engine
    perturbation.py]
    L --> E
    L --> F

    M[OOD Rules
    libero_ood/*.yaml] --> L
</pre>

#### 1.4 Mermaid 图：训练与扰动评测数据流

<pre class="mermaid">
flowchart LR
    A[Task Suite Name] --> B[Benchmark.get_task]
    B --> C[Resolve bddl/init/demo paths]
    C --> D[Load hdf5 demonstrations]
    D --> E[SequenceVLDataset / GroupedTaskDataset]
    E --> F[Algo.learn_one_task or learn_all_tasks]
    F --> G[Checkpoint + result.pt]

    H[evaluation_config.yaml] --> I[perturbation.create_env]
    I --> J[Generate *_temp bddl/init]
    J --> K[evaluate.py OffScreenRenderEnv rollout]
    K --> L[success_rate + stats file]
</pre>

#### 1.5 每模块技术栈与关键依赖

| 模块 | 技术栈 | 关键依赖 |
|---|---|---|
| 配置管理 | Hydra + OmegaConf + YAML | hydra-core, pyyaml |
| 训练与评估 | PyTorch + NumPy | torch, numpy |
| 模型文本嵌入 | HuggingFace Transformers | transformers |
| 机器人仿真 | robosuite + bddl + gym | robosuite, bddl, gym |
| 行为克隆数据管线 | robomimic | robomimic, h5py |
| 实验日志 | Weights & Biases | wandb |
| 视觉与工具 | OpenCV + matplotlib | opencv-python, matplotlib |

💡 提示
- 当前 requirements 中未固定 torch 版本，README 建议手动安装 torch==1.11.0+cu113，请优先按 README 安装。

⚠️ 警告
- 路径系统依赖 ~/.libero/config.yaml；如果该文件路径不正确，会导致 bddl/init/dataset 解析全部失败。

#### Checklist
- [ ] 已理解项目分为 配置层 / 基准层 / 训练评估层 / 扰动层 / 工具层
- [ ] 已明确训练依赖 demos、评估依赖 bddl+init+checkpoint
- [ ] 已明确 perturbation.py 只改评测环境，不改训练算法

---

### 2. 模块详细说明

#### 2.1 训练主模块（libero/lifelong/main.py）

功能描述
- 使用 Hydra 配置启动训练。
- 根据 benchmark_name 和 task_order_index 构造任务序列。
- 从 demonstrations 构建任务数据集。
- 生成任务语言嵌入，驱动策略学习。
- 输出实验目录与结果矩阵。

模块接口与调用关系
1. 读取配置并设置种子。
2. 通过 benchmark 获取任务路径。
3. 通过 get_dataset 加载每个任务 hdf5。
4. 通过 get_algo_class 构造算法实例。
5. 调用 learn_one_task 或 learn_all_tasks。
6. 调用 evaluate_loss / evaluate_success 计算结果。

关键函数
- main(hydra_cfg): 训练总控流程。
- get_task_embs(cfg, descriptions)（定义于 [libero/lifelong/utils.py](libero/lifelong/utils.py) 并在 main 中使用）: 任务语言向量生成。
- create_experiment_dir(cfg)（[libero/lifelong/utils.py](libero/lifelong/utils.py)）: 运行目录自动编号。

配置文件作用
- [libero/configs/config.yaml](libero/configs/config.yaml): 总开关与组件拼装。
- [libero/configs/data/default.yaml](libero/configs/data/default.yaml): 观测模态、序列长度、图像尺寸。
- [libero/configs/train/default.yaml](libero/configs/train/default.yaml): epoch/batch/optimizer/scheduler。
- [libero/configs/lifelong/*.yaml](libero/configs/lifelong): 算法选择与超参数。
- [libero/configs/policy/*.yaml](libero/configs/policy): 策略结构超参数。

#### 2.2 评估主模块（libero/lifelong/evaluate.py）

功能描述
- 根据命令行参数定位实验目录与 checkpoint。
- 创建并行仿真环境（SubprocVectorEnv + OffScreenRenderEnv）。
- 计算任务成功率并保存 stats。

关键命令参数
- --benchmark: libero_10/libero_spatial/libero_object/libero_goal
- --task_id: 任务索引
- --algo: base/er/ewc/packnet/multitask
- --policy: bc_rnn_policy/bc_transformer_policy/bc_vilt_policy
- --seed: 训练种子（用于定位目录）
- --device_id: GPU ID
- --ep 或 --load_task: multitask 与非 multitask 模式的 checkpoint 选择

#### 2.3 扰动模块（perturbation.py）

功能描述
- 读取 BDDL 文件并解析 obj_of_interest 与 init 状态。
- 执行五种扰动：
  - SwapPerturbator（位置交换）
  - ObjectReplacePerturbator（对象替换）
  - LanguagePerturbator（语言重写）
  - TaskPerturbator（任务语义与目标联合替换）
  - EnvironmentReplacePerturbator（环境替换）
- 使用 BDDLCombinedPerturbator 根据开关顺序组合执行。
- 使用 EvalEnvCreator 调用 generate_init_states.py 生成匹配 init 文件。

组合规则（内置约束）
1. use_swap 为 true 时必须最先执行。
2. use_task 为 true 时必须与其他扰动互斥。
3. 若配置缺失或路径不存在，会跳过对应扰动并打印提示。

#### 2.4 基准注册模块（libero/libero/benchmark）

功能描述
- 注册所有 benchmark class（大小写不敏感映射）。
- 将 task 名称映射到 bddl/init/demo 文件。
- 提供 task_order_index 置换机制（标准 10 任务套件）。

核心接口
- get_benchmark(name): 获取 benchmark 类。
- Benchmark.get_task_demonstration(i): 获取 demo 相对路径。
- Benchmark.get_task_bddl_file_path(i): 获取 bddl 绝对路径。

#### 2.5 数据集与封装模块（libero/lifelong/datasets.py）

功能描述
- 从 hdf5 构建 SequenceDataset。
- 注入 task_emb，形成视觉-语言数据样本。
- 通过 GroupedTaskDataset 支持任务分组训练。

关键点
- seq_len 用于时序建模；frame_stack/pad 逻辑继承自 robomimic。
- get_dataset 支持 obs_modality 定制。

#### 2.6 配置文件参数说明（核心）

1. 全局配置 [libero/configs/config.yaml](libero/configs/config.yaml)
- seed: 随机种子（默认 10000）
- benchmark_name: 任务套件（默认 LIBERO_SPATIAL）
- task_embedding_format: bert/one-hot/gpt2/clip/roberta
- use_wandb: 是否上传实验日志

2. 训练配置 [libero/configs/train/default.yaml](libero/configs/train/default.yaml)
- n_epochs: 默认 50
- batch_size: 默认 32
- grad_clip: 默认 100.0
- use_augmentation: 默认 true

3. 评估配置 [libero/configs/eval/default.yaml](libero/configs/eval/default.yaml)
- n_eval: 默认 20
- eval_every: 默认 5
- max_steps: 默认 600
- num_procs: 默认 20

4. 数据配置 [libero/configs/data/default.yaml](libero/configs/data/default.yaml)
- seq_len: 默认 10
- img_h/img_w: 默认 128/128
- obs.modality.rgb: agentview_rgb, eye_in_hand_rgb
- obs.modality.low_dim: gripper_states, joint_states

5. 扰动配置 [evaluation_config.yaml](evaluation_config.yaml)
- use_environment/use_swap/use_object/use_language/use_task
- ood_task_configs: 五类扰动 YAML 文件路径
- perturbation_mapping: 后缀映射（env/swap/object/lan/task）

⚠️ 警告
- EnvironmentReplacePerturbator 当前实现中 new_env 被写死为 living_room_table，这与“随机替换”注释不一致，实际行为请以代码为准。

#### Checklist
- [ ] 已理解训练、评估、扰动三条主链路
- [ ] 已掌握 benchmark_name 与 task suite 的对应关系
- [ ] 已掌握 evaluation_config.yaml 与 libero_ood 的联动关系

---

### 3. 代码组织逻辑

#### 3.1 命名规范与文件组织原则

1. 配置即组合
- 通过 Hydra defaults 在 config.yaml 中组合 data/policy/train/eval/lifelong 子配置。

2. benchmark 即路由
- 所有任务以 suite + task_id 形式路由到 bddl/init/demo 三类资源。

3. 算法与模型解耦
- algo 负责训练策略与记忆机制。
- policy 负责网络结构与动作头。

4. 扰动与训练解耦
- 扰动逻辑集中在 perturbation.py 与 libero_ood，不侵入训练算法。

#### 3.2 设计模式（工程实践层）

1. 注册表模式
- benchmark 注册：register_benchmark。
- algo/policy 通过映射函数动态加载。

2. 策略模式
- 不同终身学习算法共享统一训练流程接口。

3. 配置驱动模式
- 通过 YAML 配置切换算法、策略与超参数，避免硬编码。

#### 3.3 数据流与控制流

训练控制流（简述）
1. Hydra 解析配置。
2. benchmark 构建任务序列。
3. 构建数据集 + 任务嵌入。
4. 算法按任务增量训练。
5. 周期性评估并写出结果。

扰动评测控制流（简述）
1. 读取 evaluation_config.yaml。
2. 对目标 suite 的 bddl 执行扰动并输出 *_temp。
3. 生成匹配 init_states。
4. 外部评测脚本或 evaluate.py 加载新 suite 进行 rollout。

#### Checklist
- [ ] 已理解“配置驱动 + 注册表 + 解耦分层”三大组织原则
- [ ] 已理解训练与扰动评测互不耦合

---

## 第二部分：快速上手指南

### 1. 环境准备

#### 1.1 系统要求

- OS: Linux/Windows/macOS（推荐 Linux + CUDA）
- Python: 3.8.x（README 示例为 3.8.13）
- GPU: CUDA 11.3 对应 torch 1.11.0+cu113（推荐）
- Mujoco/robosuite 依赖: [需补充]（依据你的服务器环境）

#### 1.2 依赖安装步骤（可直接执行）

<pre><code class="language-bash">conda create -n libero_pro python=3.8.13 -y
conda activate libero_pro

git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git
cd LIBERO-PRO

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
</code></pre>

#### 1.3 数据与资源准备

1. 下载官方任务数据（demo）
<pre><code class="language-bash">python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface
</code></pre>

2. 下载并放置 LIBERO-Pro 的 bddl/init 增补文件
<pre><code class="language-bash"># 示例，按实际下载目录调整
# 目标目录必须与项目结构一致
# bddl -> libero/libero/bddl_files/
# init  -> libero/libero/init_files/
</code></pre>

3. 校验数据完整性
<pre><code class="language-bash">python scripts/check_dataset_integrity.py
python benchmark_scripts/check_task_suites.py
</code></pre>

💡 提示
- 首次 import libero 时会在用户目录生成 ~/.libero/config.yaml，请确认其中 datasets/bddl_files/init_states 指向真实路径。

⚠️ 警告
- 如果路径配置错误，训练与评估会报“文件不存在”，常被误判为代码问题。

#### 1.4 常见问题与解决方案

1. 问题：找不到 bddl/init/dataset
- 解决：检查 ~/.libero/config.yaml 路径映射是否存在且可读。

2. 问题：evaluate.py 找不到 checkpoint
- 解决：确认 benchmark/algo/policy/seed 与训练一致，并检查 run_xxx 自动编号目录。

3. 问题：扰动后评估失败
- 解决：检查 evaluation_config.yaml 中开关组合与 ood_task_configs 路径；确保 generate_init_states.py 可执行。

4. 问题：环境替换效果异常
- 解决：当前环境替换实现存在固定目标环境逻辑，建议先用 object/language/swap 做稳定评估。

#### Checklist
- [ ] 已完成 conda 环境与依赖安装
- [ ] 已准备 datasets + bddl_files + init_files
- [ ] 已完成 check_dataset_integrity 与 check_task_suites

---

### 2. 项目启动流程

#### 2.1 训练启动命令序列

基础训练（默认配置）
<pre><code class="language-bash">lifelong.main
</code></pre>

指定 benchmark、算法、策略
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=er policy=bc_transformer_policy seed=10000
</code></pre>

多任务训练示例
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_10 lifelong=multitask policy=bc_vilt_policy seed=10000
</code></pre>

#### 2.2 参数含义与可选值（核心）

| 参数 | 含义 | 常见可选值 |
|---|---|---|
| benchmark_name | 基准任务集 | LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10 |
| lifelong | 算法配置组 | base, er, ewc, agem, packnet, multitask |
| policy | 策略网络 | bc_rnn_policy, bc_transformer_policy, bc_vilt_policy |
| seed | 随机种子 | 任意整数 |
| data.seq_len | 时序长度 | 常见 10, 20 [需补充] |
| train.n_epochs | 训练轮数 | 默认 50 |
| eval.n_eval | 每次评估 episode 数 | 默认 20 |

#### 2.3 评估启动命令序列

非 multitask checkpoint 评估
<pre><code class="language-bash">lifelong.eval --benchmark libero_spatial --task_id 0 --algo er --policy bc_transformer_policy --seed 10000 --load_task 9 --device_id 0
</code></pre>

multitask checkpoint 评估
<pre><code class="language-bash">lifelong.eval --benchmark libero_10 --task_id 3 --algo multitask --policy bc_vilt_policy --seed 10000 --ep 50 --device_id 0
</code></pre>

#### 2.4 多场景启动配置实例

场景 A：基线 Sequential + RNN
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_GOAL lifelong=base policy=bc_rnn_policy seed=1
</code></pre>

场景 B：ER 增量学习 + Transformer
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=er lifelong.n_memories=1000 policy=bc_transformer_policy seed=2
</code></pre>

场景 C：PackNet 稀疏化约束
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_OBJECT lifelong=packnet lifelong.prune_perc=0.75 lifelong.post_prune_epochs=50 policy=bc_vilt_policy seed=3
</code></pre>

💡 提示
- Hydra 支持命令行覆盖任意配置项，建议将常用实验命令记录为 shell 脚本。

⚠️ 警告
- benchmark_name 使用大写类名风格（如 LIBERO_SPATIAL），evaluate.py 的 --benchmark 使用小写风格（如 libero_spatial），两者不要混淆。

#### Checklist
- [ ] 能独立启动至少 1 个训练命令
- [ ] 能独立完成 1 次 checkpoint 评估
- [ ] 已掌握大写 benchmark_name 与小写 --benchmark 的区别

---

### 3. 训练参数配置

#### 3.1 参数列表（默认值/范围/影响）

| 参数 | 默认值 | 建议范围 | 影响 |
|---|---|---|---|
| train.n_epochs | 50 | 20-200 [需补充] | 决定收敛程度与训练时长 |
| train.batch_size | 32 | 16-128 [需补充] | 影响显存占用与梯度稳定性 |
| train.grad_clip | 100.0 | 1-100 [需补充] | 避免梯度爆炸 |
| eval.n_eval | 20 | 10-100 | 影响评估方差 |
| eval.max_steps | 600 | 200-1000 [需补充] | 影响任务完成容错 |
| data.seq_len | 10 | 5-30 [需补充] | 影响时序建模能力 |
| data.img_h/img_w | 128/128 | 84-224 [需补充] | 影响视觉质量与算力开销 |
| lifelong.n_memories (ER/AGEM) | 1000 | 500-5000 [需补充] | 影响回放抗遗忘能力 |
| lifelong.e_lambda (EWC) | 50000 | 1e3-1e6 [需补充] | 影响旧任务约束强度 |
| lifelong.prune_perc (PackNet) | 0.75 | 0.3-0.9 [需补充] | 影响容量释放与性能 |

#### 3.2 常用参数模板

模板 1：快速冒烟训练
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=base policy=bc_rnn_policy train.n_epochs=5 eval.eval_every=1 eval.n_eval=5
</code></pre>

模板 2：稳定复现实验
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_10 lifelong=er policy=bc_transformer_policy seed=10000 train.n_epochs=50 eval.n_eval=20
</code></pre>

模板 3：重视抗遗忘
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_OBJECT lifelong=ewc lifelong.e_lambda=50000 lifelong.gamma=0.9 policy=bc_vilt_policy
</code></pre>

#### 3.3 调优建议与最佳实践

1. 先固定数据与路径
- 先保证 dataset/bddl/init 可读，再调超参数。

2. 再固定 policy，最后切 algo
- 同时改 policy 与 algo 难以定位收益来源。

3. 评估配置建议
- eval.n_eval 太小会导致方差过高；建议至少 20。

4. 资源约束策略
- 显存不足先降 batch_size，再降分辨率，再降 seq_len。

#### Checklist
- [ ] 已掌握最少 3 个核心参数对性能的影响
- [ ] 已准备可复现实验命令模板
- [ ] 已建立“先路径后参数”的调参顺序

---

## 第三部分：深入学习路线

### 1. 代码阅读顺序

#### 1.1 推荐阅读路径

阶段 1：总入口与配置拼装
1. [setup.py](setup.py)
2. [libero/configs/config.yaml](libero/configs/config.yaml)
3. [libero/libero/__init__.py](libero/libero/__init__.py)

阶段 2：训练与评估主干
1. [libero/lifelong/main.py](libero/lifelong/main.py)
2. [libero/lifelong/evaluate.py](libero/lifelong/evaluate.py)
3. [libero/lifelong/utils.py](libero/lifelong/utils.py)

阶段 3：基准和数据
1. [libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py)
2. [libero/libero/libero_suite_task_map.py](libero/libero/libero_suite_task_map.py)
3. [libero/lifelong/datasets.py](libero/lifelong/datasets.py)

阶段 4：算法与策略
1. [libero/lifelong/algos/base.py](libero/lifelong/algos/base.py)
2. [libero/lifelong/algos/er.py](libero/lifelong/algos/er.py)
3. [libero/lifelong/models/bc_transformer_policy.py](libero/lifelong/models/bc_transformer_policy.py)

阶段 5：OOD 扰动
1. [perturbation.py](perturbation.py)
2. [evaluation_config.yaml](evaluation_config.yaml)
3. [libero_ood/ood_task.yaml](libero_ood/ood_task.yaml)（及其他 ood_*.yaml）

#### 1.2 必读与可选文件

必读
- 训练主干：[libero/lifelong/main.py](libero/lifelong/main.py)
- 基准映射：[libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py)
- 扰动引擎：[perturbation.py](perturbation.py)

可选
- 数据采集：[scripts/collect_demonstration.py](scripts/collect_demonstration.py)
- 任务模板：[scripts/create_libero_task_example.py](scripts/create_libero_task_example.py)

#### 1.3 每阶段学习目标

1. 能解释任务如何从 suite+id 定位到 bddl/init/demo。
2. 能解释一次训练运行目录如何生成、如何恢复评估。
3. 能解释一个扰动开关如何改变评测任务集。

#### Checklist
- [ ] 已按 5 阶段完成阅读
- [ ] 能独立追踪一个任务从配置到 rollout 的全链路

---

### 2. 核心概念理解

#### 2.1 关键术语

- Benchmark Suite: 一组任务集合（如 LIBERO_SPATIAL）。
- Task Order: 10 任务套件上的任务排列顺序索引。
- Task Embedding: 从语言指令生成的文本向量。
- Lifelong Learning: 按任务序列持续学习并关注遗忘。
- OOD Perturbation: 在评测阶段修改环境/语义以测泛化。

#### 2.2 核心算法/业务逻辑

1. 增量学习逻辑
- 对于 Sequential/ER/EWC/PackNet：按任务逐个学习并在已见任务上评估。

2. 多任务学习逻辑
- Multitask 一次性联合训练全部任务，再统一评估。

3. 扰动生成逻辑
- 从原始 bddl 出发，按 flags 与顺序规则执行扰动，落地为 *_temp 套件。

#### 2.3 理论基础与参考资料

- LIBERO 原始 benchmark 论文与官方仓库。
- Continual Learning 经典方法：ER、EWC、AGEM、PackNet。
- 行为克隆与视觉语言动作建模（VLA）基础。
- 参考文献：[需补充]

#### Checklist
- [ ] 能区分 Sequential 与 Multitask 的训练差异
- [ ] 能解释 ER/EWC/PackNet 在抗遗忘上的不同机制
- [ ] 能解释扰动评测为何不等价于训练增强

---

### 3. 二次开发指南

#### 3.1 可扩展点与自定义方法

1. 新增算法
- 位置：[libero/lifelong/algos](libero/lifelong/algos)
- 要点：实现统一训练接口，并在 [libero/lifelong/algos/__init__.py](libero/lifelong/algos/__init__.py) 注册。

2. 新增策略网络
- 位置：[libero/lifelong/models](libero/lifelong/models)
- 要点：实现策略 forward/get_action，并在 [libero/lifelong/models/__init__.py](libero/lifelong/models/__init__.py) 注册。

3. 新增扰动算子
- 位置：[perturbation.py](perturbation.py)
- 要点：实现 perturb 接口，并纳入 BDDLCombinedPerturbator 执行顺序。

4. 新增任务套件
- 更新 [libero/libero/libero_suite_task_map.py](libero/libero/libero_suite_task_map.py) 与 benchmark 注册类。

#### 3.2 贡献规范与开发流程（建议）

1. 分支规范
- feat/xxx、fix/xxx、docs/xxx。

2. 变更最小化
- 配置改动与算法改动分开提交。

3. 文档同步
- 新增配置项必须同步更新 docs 与示例命令。

4. 兼容性
- 保持已有命令默认行为不变。

#### 3.3 调试技巧与测试方法

1. 路径调试
- 打印 get_libero_path 的关键路径并检查存在性。

2. 数据调试
- 使用 [scripts/get_dataset_info.py](scripts/get_dataset_info.py) 验证动作范围与轨迹长度。

3. 环境调试
- 使用 [benchmark_scripts/render_single_task.py](benchmark_scripts/render_single_task.py) 先验证单任务可渲染。

4. 扰动调试
- 对单个 bddl 先跑 process_bddl_file_mixed，再检查输出 *_temp 文件。

#### 3.4 二开示例（2-3 个）

示例 1：新增一种语义扰动规则

目标
- 为某个 task suite 增加新的语言改写候选。

步骤
1. 编辑 [libero_ood/ood_language.yaml](libero_ood/ood_language.yaml)，为目标 task 增加候选文本。
2. 在 [evaluation_config.yaml](evaluation_config.yaml) 中确保 use_language: true。
3. 执行评测前环境生成流程。

命令
<pre><code class="language-bash"># 由外部评测脚本或自定义脚本触发 perturbation.create_env
# 然后使用目标框架进行 rollout 评测
</code></pre>

示例 2：新增一个终身学习算法配置

目标
- 在现有算法上复制出一个新配置组用于对比实验。

步骤
1. 新建 [libero/configs/lifelong/my_algo.yaml](libero/configs/lifelong/my_algo.yaml) [需补充]。
2. 若为新算法实现，新增 [libero/lifelong/algos/my_algo.py](libero/lifelong/algos/my_algo.py) [需补充]。
3. 在 [libero/lifelong/algos/__init__.py](libero/lifelong/algos/__init__.py) 注册。

命令
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_10 lifelong=my_algo policy=bc_transformer_policy seed=42
</code></pre>

示例 3：新增一个 benchmark 套件

目标
- 将新任务集合并入 benchmark 统一路由。

步骤
1. 在 [libero/libero/libero_suite_task_map.py](libero/libero/libero_suite_task_map.py) 增加 suite->tasks 列表。
2. 在 [libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py) 增加 suite 名与注册类。
3. 准备对应目录的 bddl/init/demo。

命令
<pre><code class="language-bash">lifelong.main benchmark_name=LIBERO_MY_SUITE lifelong=base policy=bc_rnn_policy
</code></pre>

💡 提示
- 对二开优先采用“配置扩展”，其次才是“代码扩展”，可显著降低维护成本。

⚠️ 警告
- 新增 suite 但未同步准备 init/demo，会在训练或评估时出现路径断裂错误。

#### Checklist
- [ ] 能独立新增一种配置级扩展（不改代码）
- [ ] 能独立新增一个算法或策略并注册
- [ ] 能独立新增 suite 并完成路径闭环

---

## 附录：关键命令速查

<pre><code class="language-bash"># 安装
pip install -r requirements.txt
pip install -e .

# 下载数据
python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface

# 数据与任务可用性检查
python scripts/check_dataset_integrity.py
python benchmark_scripts/check_task_suites.py

# 训练
lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=er policy=bc_transformer_policy

# 评估
lifelong.eval --benchmark libero_spatial --task_id 0 --algo er --policy bc_transformer_policy --seed 10000 --load_task 9 --device_id 0
</code></pre>

---

## 文档状态

- 版本: v1.0
- 更新时间: 2026-04-20
- 不确定项标注: [需补充]
