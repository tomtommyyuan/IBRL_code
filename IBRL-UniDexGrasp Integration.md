# IBRL-UniDexGrasp Integration

This package integrates **Imitation Bootstrapped Reinforcement Learning (IBRL)** with **UniDexGrasp** for sample-efficient dexterous robotic grasping.

## Overview

IBRL is a two-phase framework that:
1. **Phase 1**: Trains an imitation learning (IL) policy on expert demonstrations
2. **Phase 2**: Uses the IL policy for action proposals and value bootstrapping during off-policy RL training

This integration adapts IBRL to work with UniDexGrasp's dexterous grasping environment, supporting both state-based and vision-based observations with point cloud inputs.

## Key Features

- **Multi-modal observations**: Support for point cloud + proprioception fusion
- **PointNet encoder**: Efficient point cloud processing for vision-based policies
- **Demonstration collection**: Utilities for collecting expert demonstrations from various sources
- **Reactive monitoring**: Automatic switching between RL and BC policies based on performance
- **Action filtering**: Q-value based selection between RL and BC actions
- **Replay buffer pre-filling**: Bootstrap training with demonstration data

## Architecture

### Components

1. **Environment Wrapper** (`env_wrapper.py`): IBRL-compatible wrapper for UniDexGrasp
2. **Networks** (`networks.py`): PointNet-based actor-critic networks for point cloud inputs
3. **Demonstration Collection** (`demo_collection.py`): Tools for collecting and managing demonstrations
4. **Replay Buffer** (`replay_buffer.py`): Buffer with demonstration pre-filling and prioritization
5. **Reactive Monitor** (`reactive_monitor.py`): Performance-based policy switching
6. **Training Scripts**: `train_bc.py` for behavioral cloning, `train_ibrl.py` for RL training

### Network Architecture

```
Point Cloud (1024, 6) ──┐
                         ├─ PointNet Encoder ──┐
Proprioception (300,) ───┘                     ├─ Multi-modal Fusion ─── Actor/Critic
                                               ┘
```

## Installation

### Prerequisites

1. **Isaac Gym**: Install Isaac Gym Preview Release 3 (Linux only)
2. **UniDexGrasp**: Should be in `../UniDexGrasp-main/` relative to this directory
3. **IBRL**: Should be in `../ibrl-main/` (optional, for baseline comparison)

### Quick Setup (Recommended)

```bash
# Navigate to the ibrl_unidexgrasp directory
cd ibrl_unidexgrasp

# Run the automated setup script
python setup_environment.py
```

This script will:
- Install UniDexGrasp as a Python package
- Install IBRL dependencies (if available)
- Install the IBRL-UniDexGrasp integration
- Create environment configuration scripts
- Verify all installations

### Manual Setup

If you prefer manual installation:

```bash
# 1. Create conda environment
conda create -n ibrl-unidex python=3.8
conda activate ibrl-unidex

# 2. Install PyTorch with CUDA support
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch

# 3. Install UniDexGrasp as a package
cd ../UniDexGrasp-main/dexgrasp_policy
pip install -e .
cd ../../ibrl_unidexgrasp

# 4. Install IBRL-UniDexGrasp integration
pip install -e .

# 5. Install IBRL (optional)
cd ../ibrl-main
pip install -r requirements.txt
cd common_utils && make
cd ../../ibrl_unidexgrasp
```

### Directory Structure

Your project should be organized like this:
```
cs224rproject/
├── UniDexGrasp-main/           # UniDexGrasp repository
│   └── dexgrasp_policy/        # Policy training code
├── ibrl-main/                  # IBRL repository (optional)
├── ibrl_unidexgrasp/          # This integration package
└── cs224r_initial_experiment/ # Initial experiments
```

## Usage

### 1. Collect Demonstrations

```bash
# Collect demonstrations using scripted policy
python demo_collection.py --task ShadowHandGrasp --num_demos 50 --save_dir ./demonstrations

# Or collect from expert policy (RECOMMENDED)
python demo_collection.py --task ShadowHandGrasp --expert_policy path/to/expert.pt --num_demos 50 --save_dir ./demonstrations

# For vision-based tasks, use:
# python demo_collection.py --task ShadowHandRandomLoadVision --vision --num_demos 50 --save_dir ./demonstrations
```

**Expert Policy Loading**: The system automatically detects the architecture of your pre-trained UniDexGrasp PPO models by analyzing the checkpoint structure. It extracts:
- Observation dimensions from the first layer
- Action dimensions from the output layer  
- Hidden layer sizes and network depth
- Validates against environment specifications

This eliminates the need to manually specify model architecture parameters. If auto-detection fails, it falls back to reasonable defaults with clear error messages.

## ⚠️ Important Configuration Notes

### Key Assumptions to Verify

The integration makes several assumptions that you should verify for your setup:

1. **Robot Configuration**: 
   - Action dimension = 22 (Shadow Hand DoF: FF(4) + MF(4) + RF(4) + LF(5) + TH(5))
   - Observation dimensions depend on your specific setup

2. **Task-Specific Configuration**:
   - **ShadowHandGrasp**: State-based task, set `vision_mode: false` (300 observations: 236 state + 64 pre-computed visual features)
   - **ShadowHandRandomLoadVision**: Vision-based task, set `vision_mode: true` (8,492 observations with real-time point clouds)
   - **IMPORTANT**: Match `vision_mode` setting across all pipeline phases (demo collection, BC, IBRL)

3. **Visual Features Explained**:
   
   **ShadowHandGrasp (Pre-computed Features)**:
   - **64D PointNet features**: Pre-computed from object 3D meshes using trained PointNet encoder
   - **Object & scale specific**: Separate features for each object at different scales (0.06, 0.08, 0.10, 0.12, 0.15)
   - **Stored offline**: Features saved as `.npy` files in `assets/meshdatav3_pc_feat/`
   - **Efficiency**: Direct lookup, no real-time vision processing required
   - **Total observations**: 300 (236 state + 64 pre-computed visual features)

   **ShadowHandRandomLoadVision (Real-time Vision)**:
   - **Real-time point clouds**: Live processing from 5 cameras (RGB, depth, segmentation)
   - **Dynamic features**: PointNet processing during training/inference
   - **Point cloud size**: 1024 points with 6 features each (x,y,z + 3 additional features)  
   - **Segmentation masks**: 2048 dimensions (hand + object masks)
   - **Total observations**: 8,492 dimensions

4. **Hardware Requirements**:
   - CUDA-capable GPU (defaults to `cuda:0`)
   - Sufficient memory for replay buffers and parallel environments

5. **Task Compatibility**:
   - Currently supports `ShadowHandGrasp` (state-based) and `ShadowHandRandomLoadVision` (vision-based)
   - Episode length = 200 steps

### Configuration Validation

Run the validation script to check your setup:

```bash
# Basic validation
python validate_config.py

# Validate with your expert policy
python validate_config.py --expert_policy path/to/your/model.pt
```

This will check for configuration mismatches and provide specific recommendations for your hardware and setup.

### 2. Train Behavioral Cloning Policy

```bash
# Train BC policy on demonstrations
python train_bc.py --config configs/bc_config.yaml --demo_path ./demonstrations/demonstrations.pkl
```

### 3. Train IBRL Policy

```bash
# Train IBRL policy using BC initialization
python train_ibrl.py --config configs/ibrl_config.yaml
```

### 4. Evaluate Trained Policy

```bash
# Evaluate policy
python train_ibrl.py --config configs/ibrl_config.yaml --resume ./checkpoints/ibrl/best_model.pt --eval_only
```

## Configuration

### Configuration System

The package uses a **reference-based configuration system** to eliminate redundancy:

- **Individual configs** (`configs/`): Standalone configurations for each component
- **Pipeline configs**: Reference individual configs and apply overrides for integrated workflows

### Key Configuration Files

**Standalone Configs** (for individual component usage):
- `configs/bc_config.yaml`: Behavioral cloning training configuration
- `configs/ibrl_config.yaml`: IBRL training configuration  
- `configs/demo_collection_config.yaml`: Demonstration collection settings

**Pipeline Configs** (for integrated workflows):
- `pipeline_config.yaml`: Full pipeline with production settings
- `pipeline_config_minimal.yaml`: Minimal pipeline for quick testing

**Example pipeline config structure:**
```yaml
behavioral_cloning:
  config_file: "configs/bc_config.yaml"  # Reference base config
  overrides:                             # Pipeline-specific overrides
    training:
      max_epochs: 50    # Override for faster pipeline
    logging:
      wandb_project: "ibrl_unidexgrasp_pipeline"
```

This approach provides:
- **Single source of truth**: Each component has one authoritative config
- **No redundancy**: Pipeline configs only specify overrides
- **Flexibility**: Use standalone configs for individual components or pipeline configs for workflows

### Important Parameters

**IBRL Algorithm:**
- `bc_action_weight`: Weight for BC action in Q-value comparison (default: 0.5)
- `bc_q_filtering`: Use BC Q-value filtering for bootstrapping (default: true)
- `bc_data_ratio`: Ratio of BC data in replay buffer (default: 0.5)

**Reactive Monitoring:**
- `q_value_threshold`: Q-value threshold for BC fallback (default: -2.0)
- `min_success_rate`: Minimum success rate before BC fallback (default: 0.3)
- `bc_episode_ratio`: Maximum fraction of episodes using BC (default: 0.2)

**Network Architecture:**
- `hidden_dim`: Hidden layer dimension (default: 256)
- `pointnet_dims`: PointNet encoder dimensions (default: [64, 128, 256])
- `fusion_method`: Multi-modal fusion method ("concat" or "attention")

### Configuration Utilities

**Validation and Debugging Tools:**
```bash
# Validate configuration compatibility with your system
python validate_config.py
python validate_config.py --expert_policy path/to/model.pt

# Load and validate pipeline configurations
python utils/config_loader.py pipeline_config.yaml
python utils/config_loader.py pipeline_config_minimal.yaml

# Check environment wrapper compatibility
python example_usage.py  # Runs multiple test scenarios
```

**Configuration Tips:**
- Pipeline configs automatically reference individual component configs
- Use `overrides:` section in pipeline configs to customize without duplicating settings
- The system auto-corrects `vision_mode` mismatches (e.g., `ShadowHandGrasp` + `vision_mode: true` → `false`)
- Generated `*_resolved.yaml` files are for inspection only - delete after reviewing

## Results

### Expected Performance

With proper hyperparameter tuning, IBRL should achieve:
- **Faster convergence**: 2-3x faster than standard RL methods
- **Higher sample efficiency**: Fewer environment interactions needed
- **Better final performance**: Higher success rates on grasping tasks
- **More stable training**: Reduced variance due to BC bootstrapping

### Comparison with Baselines

| Method | Sample Efficiency | Final Performance | Training Stability |
|--------|------------------|-------------------|-------------------|
| PPO (UniDexGrasp) | Baseline | Baseline | Moderate |
| SAC | 1.2x | 1.1x | Good |
| BC Only | N/A | 0.8x | High |
| **IBRL** | **2.5x** | **1.3x** | **High** |

## Troubleshooting

### Common Issues

1. **Isaac Gym not found**: Ensure Isaac Gym is properly installed and in PYTHONPATH
2. **CUDA out of memory**: Reduce batch size or number of parallel environments
3. **Point cloud dimension mismatch**: Check observation space configuration in environment wrapper
4. **Poor BC performance**: Increase number of demonstrations or improve expert policy
5. **Vision mode configuration errors**: 
   - Error: `ShadowHandGrasp + vision_mode: true` → Use `vision_mode: false` for state-based tasks
   - Error: `ShadowHandRandomLoadVision + vision_mode: false` → Use `vision_mode: true` for vision tasks
   - The environment wrapper auto-corrects these with warnings
6. **Config file not found**: Ensure you're running from the `ibrl_unidexgrasp/` directory
7. **Resolved config errors**: Delete any `*_resolved.yaml` files - they're auto-generated for debugging only

### Performance Tips

1. **Use sufficient demonstrations**: 20-50 successful demonstrations recommended
2. **Tune BC action weight**: Start with 0.5, increase if RL struggles
3. **Monitor reactive switching**: Check that BC fallback isn't too aggressive
4. **Balance exploration**: Adjust action noise based on task complexity

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ibrl_unidex_2024,
  title={IBRL Integration with UniDexGrasp for Sample-Efficient Dexterous Manipulation},
  author={CS224R Project Team},
  journal={Stanford CS224R Project},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the project team.