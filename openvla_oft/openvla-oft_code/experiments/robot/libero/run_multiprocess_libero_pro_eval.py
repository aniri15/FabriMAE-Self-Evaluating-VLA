"""
run_multiprocess_libero_pro_eval.py — LIBERO-PRO evaluation for OpenVLA-OFT.

Note (MAE naming): user-facing metric MAE-D (self_eval_mode=mae-d) uses internal mad
computation (top-k visual entropy). MAE-C is PI-only.
"""

import json
import logging
import debugpy
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

import draccus
from experiments.robot.libero.repo_paths import (
    default_evaluation_config_swap,
    libero_pro_libero_root,
    setup_libero_config,
)
import numpy as np
import torch
import tqdm
import yaml
from libero.libero import benchmark

import wandb
import scipy

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    resolve_self_eval_output_dir,
)
from experiments.robot.libero.mae_metric_naming import DEFAULT_MAE_D_OUTPUT, MAE_D_MODE
from experiments.robot.libero.attention_mad_utils import (
    append_online_mad_jsonl,
    build_episode_mad_record,
    compute_visual_entropy_lh_from_attentions,
    rollout_requests_attentions,
)
from experiments.robot.libero.self_eval_utils import (
    append_consistency_record,
    apply_action_noise,
    configure_self_eval,
    consistency_num_repeats,
    self_eval_json_filename,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from experiments.robot.libero.LIBERO_PRO import perturbation



# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 520,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 520,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 520,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_GOAL_TEMP: 300,
    TaskSuite.LIBERO_SPATIAL_TEMP: 220,
    TaskSuite.LIBERO_10_TEMP: 520,
    TaskSuite.LIBERO_OBJECT_TEMP: 280,
    TaskSuite.LIBERO_GOAL_LAN: 300,
    TaskSuite.LIBERO_SPATIAL_LAN: 220,
    TaskSuite.LIBERO_10_LAN: 520,
    TaskSuite.LIBERO_OBJECT_LAN: 280,
    TaskSuite.LIBERO_GOAL_OBJECT: 300,
    TaskSuite.LIBERO_SPATIAL_OBJECT: 220,
    TaskSuite.LIBERO_10_OBJECT: 520,
    TaskSuite.LIBERO_OBJECT_OBJECT: 280,
    TaskSuite.LIBERO_GOAL_SWAP: 300,
    TaskSuite.LIBERO_SPATIAL_SWAP: 220,
    TaskSuite.LIBERO_10_SWAP: 520,
    TaskSuite.LIBERO_OBJECT_SWAP: 280,
    TaskSuite.LIBERO_GOAL_TASK: 300,
    TaskSuite.LIBERO_SPATIAL_TASK: 220,
    TaskSuite.LIBERO_10_TASK: 520,
    TaskSuite.LIBERO_OBJECT_TASK: 280,
    TaskSuite.LIBERO_GOAL_ENV: 300,
    TaskSuite.LIBERO_SPATIAL_ENV: 220,
    TaskSuite.LIBERO_10_ENV: 520,
    TaskSuite.LIBERO_OBJECT_ENV: 280,
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    evaluation_config_path: Optional[str] = None     # Path to evaluation config YAML file (for LIBERO-PRO perturbations)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    track_uncertainty: bool = False                  # Whether to compute Shannon entropy per action query

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    save_attentions: bool = False                    # Whether to save attention matrices
    save_attn_dir: str = ""                          # Directory to save attention data
    save_attn_format: str = "json"                   # Format to save attention: "json" or "pt"

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Task slicing / parallel submission (to align with submit scripts)
    max_parallel_tasks: int = 1                      # Kept for CLI compatibility (tasks run sequentially here)
    task_start: int = -1                             # Start task ID (inclusive, -1 => auto)
    task_end: int = -1                               # End task ID (exclusive, -1 => all tasks)

    enable_self_eval: bool = False                   # Enable self-consistency scoring during rollout
    self_eval_mode: str = "consistency"              # consistency | mae-d
    consistency_repeats_per_init: int = 3            # Same fixed init scene repeats (consistency mode)
    action_noise_std: float = 0.0                    # Stddev of Gaussian noise on first 6 action dims
    self_eval_output_dir: str = ""                   # Optional override for self-eval JSON output dir
    attention_eval_method: str = "mae-d"             # mae-d only (MAE-C is PI-only)
    attention_eval_ratios: str = "0.01"              # Comma-separated top-k ratios for online MAE-D
    attention_eval_output_name: str = DEFAULT_MAE_D_OUTPUT
    libero_benchmark: str = "libero_pro"             # libero | libero_pro
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name
    #unnorm_key = cfg.unnorm_key

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig, task_start: Optional[int] = None, task_end: Optional[int] = None):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id_parts = [f"EVAL-{cfg.task_suite_name}"]

    # Check evaluation config to add perturbation info to filename
    if cfg.evaluation_config_path is not None and os.path.exists(cfg.evaluation_config_path):
        try:
            with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
                eval_cfg = yaml.safe_load(f)
                
            if eval_cfg.get("use_swap", False):
                run_id_parts.append("use-swap")
            if eval_cfg.get("use_environment", False):
                run_id_parts.append("use-env")
            if eval_cfg.get("use_object", False):
                run_id_parts.append("use-obj")
            if eval_cfg.get("use_language", False):
                run_id_parts.append("use-lang")
            if eval_cfg.get("use_task", False):
                run_id_parts.append("use-task")
        except Exception as e:
            print(f"[WARN] Failed to read evaluation config for logging setup: {e}")

    run_id_parts.append(f"{cfg.model_family}-{DATE_TIME}")
    
    # Add task range to run_id if provided to avoid filename collisions
    if task_start is not None and task_end is not None:
        run_id_parts.append(f"t{task_start}-{task_end}")
    
    run_id = "-".join(run_id_parts)

    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )
    #print(f"Logging to local log file: {local_log_filepath}")
    #print(f"Run ID: {run_id}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


# def save_single_attention_query(
#     cfg: Any,
#     attentions: Any,
#     task_id: int,
#     episode_idx: int,
#     query_idx: int,
#     task_description: str,
#     success: bool,
#     episode_number: int,
#     log_file=None,
#     num_patches: Optional[int] = None,
#     num_prompt_tokens: Optional[int] = None,
#     ):
#     """
#     Optimized version: Vectorized operations to remove inner loops.
#     """
#     if not cfg.save_attn_dir or attentions is None:
#         return

#     try:
#         os.makedirs(cfg.save_attn_dir, exist_ok=True)
#         if not os.access(cfg.save_attn_dir, os.W_OK):
#             # Error handling...
#             return

#         # --- Constants & Ranges ---
#         try:
#             from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
#         except ImportError:
#             ACTION_DIM = 7
#             NUM_ACTIONS_CHUNK = 8
        
#         NUM_ACTION_TOKENS = ACTION_DIM * NUM_ACTIONS_CHUNK # Typically 56
        
#         # Validate inputs
#         if num_patches is None or num_prompt_tokens is None:
#             return

#         visual_start, visual_end = 0, num_patches
#         text_start, text_end = num_patches, num_patches + num_prompt_tokens
#         action_start = text_end
#         action_end = action_start + NUM_ACTION_TOKENS

#         attn2save = []

#         # Helper for vectorized entropy: p * log(p)
#         def batch_entropy(probs, axis=-1, epsilon=1e-12):
#             """Compute entropy along axis for a numpy array."""
#             # probs shape: (56, seq_length of visual/text/action tokens)
#             # 1. Sum of probabilities --> (56, 1)
#             row_sums = np.sum(probs, axis=axis, keepdims=True)
    
#             # 2. 归一化 (Normalize) -> p_i / sum(p)
#             # 加上 epsilon 防止除以 0 (如果某一行对该模态完全没有注意力，sum 为 0)
#             # 如果 sum 为 0，probs_norm 也会是 0 (或接近 0)，最终 entropy 为 0，这是合理的
#             probs_norm = probs / (row_sums + epsilon)
            
#             # 3. Clip to avoid log(0)
#             probs_norm = np.clip(probs_norm, epsilon, 1.0)
            
#             # 4.Compute Entropy
#             # probs_norm shape: (56, seq_length of visual/text/action tokens)
#             # -np.sum(probs_norm * np.log(probs_norm), axis=axis) --> (56, 1)
#             entropy = -np.sum(probs_norm * np.log(probs_norm), axis=axis)
#             return entropy

#         if isinstance(attentions, tuple):
#             for l, layer_attn in enumerate(attentions):
#                 if layer_attn is None:
#                     continue
                
#                 # --- OPTIMIZATION 1: Move entire layer to CPU once ---
#                 # layer_attn shape: (1, num_heads, seq_len, seq_len)
#                 if hasattr(layer_attn, 'cpu'):
#                     layer_attn = layer_attn.float().cpu().numpy() # Convert to float32 numpy here
#                 else:
#                     layer_attn = layer_attn.astype(np.float32)

#                 batch_size, num_heads, seq_len, _ = layer_attn.shape
                
#                 if action_end > seq_len:
#                     print(f"Error: action_end {action_end} > seq_len {seq_len}")
#                     continue

#                 temp2save_layer = []

#                 for h in range(num_heads):
#                     # --- OPTIMIZATION 2: Extract Block of Interest ---
#                     # We only care about rows corresponding to Action Tokens generating output
#                     # Shape: (NUM_ACTION_TOKENS, seq_len) -> (56, seq_len)
#                     # This matrix contains attention FROM all action tokens TO everywhere
#                     head_action_attn_block = layer_attn[0, h, action_start:action_end, :]

#                     # --- OPTIMIZATION 3: Vectorized Sums ---
#                     # Shape: (56,)
#                     vis_sum = head_action_attn_block[:, visual_start:visual_end].sum(axis=1)
#                     txt_sum = head_action_attn_block[:, text_start:text_end].sum(axis=1)
#                     act_sum = head_action_attn_block[:, action_start:action_end].sum(axis=1)

#                     # --- OPTIMIZATION 4: Vectorized Entropy ---
#                     # Shape: (56,)
#                     vis_ent = batch_entropy(head_action_attn_block[:, visual_start:visual_end])
#                     txt_ent = batch_entropy(head_action_attn_block[:, text_start:text_end])
#                     act_ent = batch_entropy(head_action_attn_block[:, action_start:action_end])

#                     # Reshape to (NUM_ACTIONS_CHUNK, ACTION_DIM) -> (8, 7)
#                     # This maps the flat 56 tokens back to the grid structure
#                     # We ensure data is contiguous for reshaping
#                     grid_shape = (NUM_ACTIONS_CHUNK, ACTION_DIM)
                    
#                     head_mats = {
#                         "visual_attn_sum": vis_sum.reshape(grid_shape),
#                         "text_attn_sum": txt_sum.reshape(grid_shape),
#                         "action_attn_sum": act_sum.reshape(grid_shape),
#                         "visual_attn_entropy": vis_ent.reshape(grid_shape),
#                         "text_attn_entropy": txt_ent.reshape(grid_shape),
#                         "action_attn_entropy": act_ent.reshape(grid_shape),
#                     }

#                     # --- OPTIMIZATION 5: Vectorized Action Chunks Sums ---
#                     # We need to calculate how much the 56 tokens attend to specific action chunks.
#                     # We iterate 8 times (for each chunk target), but operations are vectorized over the 56 query tokens.
#                     for i in range(NUM_ACTIONS_CHUNK):
#                         c_start = action_start + i * ACTION_DIM
#                         c_end = c_start + ACTION_DIM
#                         # Sum attention to this specific chunk
#                         # head_action_attn_block is (56, seq_len), slicing columns gets (56, 7), sum axis 1 gets (56,)
#                         chunk_sum = head_action_attn_block[:, c_start:c_end].sum(axis=1)
#                         # Store as (8, 7) grid
#                         head_mats[f"action_{i+1}_attn_sum"] = chunk_sum.reshape(grid_shape)

#                     # --- OPTIMIZATION 6: Token-to-Token Attention (The detailed breakdown) ---
#                     # The original code stored "action_token{N}_attn_value" which is attention FROM current token TO token N.
#                     # This is just the square attention matrix of actions attending to actions.
#                     # Shape: (56, 56) -> rows are queries (current tokens), cols are targets (token 1..56)
#                     action_to_action_matrix = head_action_attn_block[:, action_start:action_end]
                    
#                     # We need to save this such that `head_mats[f"action_token{tok}_attn_value"]` 
#                     # is an (8, 7) grid representing how much EACH token (in the grid) attended to `tok`.
#                     # So we iterate over the COLUMNS of the action_to_action_matrix.
#                     for tok_idx in range(NUM_ACTION_TOKENS):
#                         # Attention TO token `tok_idx+1` FROM all 56 tokens
#                         # Shape: (56,) -> Reshape to (8, 7)
#                         col_val = action_to_action_matrix[:, tok_idx]
#                         head_mats[f"action_token{tok_idx+1}_attn_value"] = col_val.reshape(grid_shape)

#                     temp2save_layer.append(head_mats)
                
#                 attn2save.append(temp2save_layer)
#                 # Cleanup
#                 del layer_attn, head_action_attn_block

#         # --- Saving Logic (Unchanged but ensuring numpy types) ---
#         if attn2save:
#             episode_id = f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}"
            
#             data2save = {
#                 "episode_id": episode_id,
#                 "task_id": task_id,
#                 "episode_idx": episode_idx,
#                 "query_idx": query_idx,
#                 "task_description": task_description,
#                 "success": success,
#                 "attn": attn2save,
#                 "num_patches": num_patches,
#                 "num_prompt_tokens": num_prompt_tokens,
#             }
            
#             save_format = getattr(cfg, 'save_attn_format', 'json').lower()
#             if save_format not in ["json", "pt"]: save_format = "json"
            
#             save_path = os.path.join(cfg.save_attn_dir, f"{episode_id}.{save_format}")
            
#             # Retry loop for saving...
#             # (Keep your original saving logic here)
#             # Just showing a simple save for brevity:
#             if save_format == "pt":
#                 torch.save(data2save, save_path)
#             else:
#                 # Need a custom encoder or convert numpy to lists
#                 def np_converter(obj):
#                     if isinstance(obj, np.ndarray): return obj.tolist()
#                     if isinstance(obj, np.float32): return float(obj)
#                     raise TypeError(f"Unserializable type: {type(obj)}")
                
#                 with open(save_path, 'w') as f:
#                     json.dump(data2save, f, default=np_converter)

#     except Exception as e:
#         print(f"Error saving attention: {e}")
#         import traceback
#         traceback.print_exc()

def save_single_attention_weight(
    cfg: Any,
    attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    img: Any,
    success: bool,
    episode_number: int,
    log_file=None,
    num_patches: Optional[int] = None,
    num_prompt_tokens: Optional[int] = None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not cfg.save_attn_dir or attentions is None:
        return

    try:
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        ACTION_DIM = 7
        NUM_ACTIONS_CHUNK = 8
        NUM_ACTION_TOKENS = 56 # 7 * 8
        GRID_SHAPE = (NUM_ACTIONS_CHUNK, ACTION_DIM) # (8, 7)

        # Validate inputs
        if num_patches is None or num_prompt_tokens is None:
            return

        # Indices
        visual_start, visual_end = 0, num_patches
        text_start, text_end = num_patches, num_patches + num_prompt_tokens
        action_start = text_end
        action_end = action_start + NUM_ACTION_TOKENS

        # Containers for stacking
        layers_summary = [] # List of (Head, 6, 8, 7)
        layers_chunks = []  # List of (Head, 8, 8, 7)
        layers_tokens = []  # List of (Head, 56, 8, 7)
        layers_visual = []
        valid_layer_indices = []

        # Helper: Entropy
        def batch_entropy(probs, axis=-1, epsilon=1e-12):
            row_sums = np.sum(probs, axis=axis, keepdims=True)
            probs_norm = probs / (row_sums + epsilon)
            probs_norm = np.clip(probs_norm, epsilon, 1.0)
            return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

        # --- Processing Loop ---
        if isinstance(attentions, tuple):
            for l, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                
                # Move to CPU & Numpy
                if hasattr(layer_attn, 'cpu'):
                    layer_attn = layer_attn.float().cpu().numpy()
                else:
                    layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                batch_size, num_heads, seq_len, _ = layer_attn.shape
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 6, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Chunks: 8 chunks
                head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)

                head_visual_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, num_patches), dtype=np.float32)
                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    head_block = layer_attn[0, h, action_start:action_end, :]

                    head_visual_stack[h] = head_block[:, visual_start:visual_end]

                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_sum = head_block[:, visual_start:visual_end].sum(axis=1).reshape(GRID_SHAPE)
                    txt_sum = head_block[:, text_start:text_end].sum(axis=1).reshape(GRID_SHAPE)
                    act_sum = head_block[:, action_start:action_end].sum(axis=1).reshape(GRID_SHAPE)
                    
                    # Entropies: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_ent = batch_entropy(head_block[:, visual_start:visual_end]).reshape(GRID_SHAPE)
                    txt_ent = batch_entropy(head_block[:, text_start:text_end]).reshape(GRID_SHAPE)
                    act_ent = batch_entropy(head_block[:, action_start:action_end]).reshape(GRID_SHAPE)

                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = act_sum
                    head_summary_stack[h, 3] = vis_ent
                    head_summary_stack[h, 4] = txt_ent
                    head_summary_stack[h, 5] = act_ent

                    # 2. Chunk Sums (Vectorized)
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    # (56, seq_len) -> (56, 56)
                    act_to_act = head_block[:, action_start:action_end] # (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index
                    for i in range(NUM_ACTIONS_CHUNK):
                        c_start = i * ACTION_DIM
                        c_end = c_start + ACTION_DIM
                        # Sum cols (attention TO this chunk)
                        # (56,) -> (8, 7)
                        head_chunk_stack[h, i] = act_to_act[:, c_start:c_end].sum(axis=1).reshape(GRID_SHAPE)

                    # 3. Token Values (Vectorized)
                    # act_to_act is (56 rows queries, 56 cols targets)
                    # We want to save columns. column j = attention TO token j FROM all 56 queries.
                    # We need to reshape each column to (8, 7)
                    # act_to_act.T is (56 targets, 56 queries)
                    # Reshape to (56 targets, 8, 7)
                    head_token_stack[h] = act_to_act.T.reshape(NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM)

                # Store layer result
                layers_summary.append(head_summary_stack)
                layers_chunks.append(head_chunk_stack)
                layers_tokens.append(head_token_stack)
                layers_visual.append(head_visual_stack)
                valid_layer_indices.append(l)

                del layer_attn, head_block, head_summary_stack, head_chunk_stack, head_token_stack, head_visual_stack  

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary)) # (L, H, 6, 8, 7)
            tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)
            tensor_visuals = torch.from_numpy(np.stack(layers_visual))

            # 2. Prepare Metadata Tuple (Fixed Order)
            metadata = (
                task_id,            # 0
                episode_idx,        # 1
                query_idx,          # 2
                episode_number,     # 3
                success,            # 4
                task_description,   # 5
                valid_layer_indices,# 6 (List of ints)
                num_patches,        # 7
                num_prompt_tokens   # 8
            )

            # 3. Save as Tuple
            # file extension .pt
            save_path = os.path.join(cfg.save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            torch.save((metadata, tensor_summary, tensor_visuals, img), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

def save_single_attention_query(
    cfg: Any,
    attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    success: bool,
    episode_number: int,
    log_file=None,
    num_patches: Optional[int] = None,
    num_prompt_tokens: Optional[int] = None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not cfg.save_attn_dir or attentions is None:
        return

    try:
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        # 这里的常量硬编码是为了确保 reshape 逻辑稳定，你可以根据需要从 cfg 获取
        ACTION_DIM = 7
        NUM_ACTIONS_CHUNK = 8
        NUM_ACTION_TOKENS = 56 # 7 * 8
        GRID_SHAPE = (NUM_ACTIONS_CHUNK, ACTION_DIM) # (8, 7)

        # Validate inputs
        if num_patches is None or num_prompt_tokens is None:
            return

        # Indices
        visual_start, visual_end = 0, num_patches
        text_start, text_end = num_patches, num_patches + num_prompt_tokens
        action_start = text_end
        action_end = action_start + NUM_ACTION_TOKENS

        # Containers for stacking
        # 我们将收集所有层的 Tensor，最后一次性 Stack
        layers_summary = [] # List of (Head, 6, 8, 7)
        layers_chunks = []  # List of (Head, 8, 8, 7)
        layers_tokens = []  # List of (Head, 56, 8, 7)
        valid_layer_indices = []

        # Helper: Entropy
        def batch_entropy(probs, axis=-1, epsilon=1e-12):
            row_sums = np.sum(probs, axis=axis, keepdims=True)
            probs_norm = probs / (row_sums + epsilon)
            probs_norm = np.clip(probs_norm, epsilon, 1.0)
            return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

        # --- Processing Loop ---
        if isinstance(attentions, tuple):
            for l, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                
                # Move to CPU & Numpy
                if hasattr(layer_attn, 'cpu'):
                    layer_attn = layer_attn.float().cpu().numpy()
                else:
                    layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                batch_size, num_heads, seq_len, _ = layer_attn.shape
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 6, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Chunks: 8 chunks
                head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)

                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    head_block = layer_attn[0, h, action_start:action_end, :]

                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56,) -> Reshape (8, 7)
                    vis_sum = head_block[:, visual_start:visual_end].sum(axis=1).reshape(GRID_SHAPE)
                    txt_sum = head_block[:, text_start:text_end].sum(axis=1).reshape(GRID_SHAPE)
                    act_sum = head_block[:, action_start:action_end].sum(axis=1).reshape(GRID_SHAPE)
                    
                    # Entropies: (56,) -> Reshape (8, 7)
                    vis_ent = batch_entropy(head_block[:, visual_start:visual_end]).reshape(GRID_SHAPE)
                    txt_ent = batch_entropy(head_block[:, text_start:text_end]).reshape(GRID_SHAPE)
                    act_ent = batch_entropy(head_block[:, action_start:action_end]).reshape(GRID_SHAPE)

                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = act_sum
                    head_summary_stack[h, 3] = vis_ent
                    head_summary_stack[h, 4] = txt_ent
                    head_summary_stack[h, 5] = act_ent

                    # 2. Chunk Sums (Vectorized)
                    # We want to know how much the 56 tokens attended to each of the 8 chunks
                    # Target: action_start to action_end, sliced by 7
                    # head_block slice: (56, 56) -> reshape to (56, 8, 7) -> sum(axis=2) -> (56, 8)
                    # Wait, original logic was: sum attention TO chunk i.
                    # chunk_i range: [start, end]
                    # head_block[:, start:end].sum(axis=1) -> (56,) -> reshape (8, 7)
                    
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    act_to_act = head_block[:, action_start:action_end] # (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index
                    # So we need to transpose properly.
                    # Let's stick to original loop logic but vectorized:
                    for i in range(NUM_ACTIONS_CHUNK):
                        c_start = i * ACTION_DIM
                        c_end = c_start + ACTION_DIM
                        # Sum cols (attention TO this chunk)
                        # (56,) -> (8, 7)
                        head_chunk_stack[h, i] = act_to_act[:, c_start:c_end].sum(axis=1).reshape(GRID_SHAPE)

                    # 3. Token Values (Vectorized)
                    # act_to_act is (56 rows queries, 56 cols targets)
                    # We want to save columns. column j = attention TO token j FROM all 56 queries.
                    # We need to reshape each column to (8, 7)
                    # act_to_act.T is (56 targets, 56 queries)
                    # Reshape to (56 targets, 8, 7)
                    head_token_stack[h] = act_to_act.T.reshape(NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM)

                # Store layer result
                layers_summary.append(head_summary_stack)
                layers_chunks.append(head_chunk_stack)
                layers_tokens.append(head_token_stack)
                valid_layer_indices.append(l)

                del layer_attn, head_block, head_summary_stack, head_chunk_stack, head_token_stack  

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary)) # (L, H, 6, 8, 7)
            tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)

            # 2. Prepare Metadata Tuple (Fixed Order)
            metadata = (
                task_id,            # 0
                episode_idx,        # 1
                query_idx,          # 2
                episode_number,     # 3
                success,            # 4
                task_description,   # 5
                valid_layer_indices,# 6 (List of ints)
                num_patches,        # 7
                num_prompt_tokens   # 8
            )

            # 3. Save as Tuple
            # file extension .pt
            save_path = os.path.join(cfg.save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            # 这种保存方式极快，且加载时不需要 pickle 解析大量字典对象
            torch.save((metadata, tensor_summary, tensor_chunks, tensor_tokens), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

def save_attention_data(
    cfg: GenerateConfig,
    episode_attentions: List[Any],
    task_id: int,
    episode_idx: int,
    task_description: str,
    success: bool,
    episode_number: int,
):
    """Save attention matrices for an episode (legacy function for batch saving)."""
    if not cfg.save_attn_dir or not episode_attentions:
        return

    # Create directory if it doesn't exist
    os.makedirs(cfg.save_attn_dir, exist_ok=True)

    # Process and save each attention query
    for query_idx, attentions in enumerate(episode_attentions):
        if attentions is None:
            continue
        save_single_attention_query(
            cfg, attentions, task_id, episode_idx, query_idx,
            task_description, success, episode_number,
            num_patches=None, num_prompt_tokens=None  # Legacy function doesn't have this info
        )
    
    # Clear GPU cache after processing all attentions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def convert_numpy_to_json_serializable(obj):
    """Recursively convert numpy arrays and types to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size, 
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    task_id: int = -1,
    episode_idx: int = -1,
    episode_number: int = -1,
    self_eval_json_path: str = "",
    repeat_idx: int = 0,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    query_idx = 0  # Counter for attention queries (for immediate saving)
    episode_visual_entropy_queries: List[np.ndarray] = []

    # [FIX] Handle dynamic task suite names (e.g., libero_spatial_temp_swap)
    # If the exact name is not in TASK_MAX_STEPS, try to fall back to the base name
    if cfg.task_suite_name in TASK_MAX_STEPS:
        max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    else:
        # Try to find a base suite name that matches the prefix
        base_suite = None
        for suite in TASK_MAX_STEPS.keys():
            if cfg.task_suite_name.startswith(suite) and suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
                 base_suite = suite
                 break
        
        if base_suite:
            max_steps = TASK_MAX_STEPS[base_suite]
            #print(f"[INFO] Using max_steps from base suite '{base_suite}' for '{cfg.task_suite_name}'")
        else:
            # Fallback default
            max_steps = 300 
            print(f"[WARN] Could not determine max_steps for {cfg.task_suite_name}, using default {max_steps}")

    # Run episode
    success = False
    uncertainty_history: List[float] = []
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                if rollout_requests_attentions(cfg):
                    if cfg.track_uncertainty: # save uncertainty and attentions
                        actions, chunk_uncertainty, attentions, num_patches, num_prompt_tokens = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=cfg.use_film,
                            return_uncertainty=True,
                            output_attentions=True,
                        )
                        if chunk_uncertainty is not None:
                            uncertainty_history.append(chunk_uncertainty)
                        if attentions is not None:
                            if isinstance(attentions, tuple):
                                attentions_cpu = tuple(
                                    layer_attn.cpu() if hasattr(layer_attn, "cpu") else layer_attn
                                    for layer_attn in attentions
                                )
                            else:
                                attentions_cpu = (
                                    attentions.cpu() if hasattr(attentions, "cpu") else attentions
                                )
                            del attentions
                            if cfg.enable_self_eval and cfg.self_eval_mode == MAE_D_MODE:
                                vis_lh = compute_visual_entropy_lh_from_attentions(
                                    attentions_cpu,
                                    num_patches,
                                    num_prompt_tokens,
                                    variant="openvla_oft",
                                )
                                if vis_lh is not None:
                                    episode_visual_entropy_queries.append(vis_lh)
                            elif cfg.save_attentions and cfg.save_attn_dir and task_id >= 0 and episode_idx >= 0:
                                save_single_attention_query(
                                    cfg,
                                    attentions_cpu,
                                    task_id,
                                    episode_idx,
                                    query_idx,
                                    task_description,
                                    False,
                                    episode_number,
                                    log_file,
                                    num_patches=num_patches,
                                    num_prompt_tokens=num_prompt_tokens,
                                )
                            del attentions_cpu
                            query_idx += 1
                            if query_idx % 10 == 0:
                                torch.cuda.empty_cache()
                    else: # no uncertainty tracking but save attentions
                        #print(f"[DEBUG run_episode] Calling get_action with save_attentions=True, track_uncertainty=False, output_attentions=True")
                        result = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=cfg.use_film,
                            output_attentions=True,
                        )
                        #print(f"[DEBUG run_episode] get_action returned: type={type(result)}, length={len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
                        #print(f"[DEBUG run_episode] Expecting 4 values: actions, attentions, num_patches, num_prompt_tokens")
                        actions, attentions, num_patches, num_prompt_tokens = result
                        # Handle attentions which is a tuple of layers
                        if attentions is not None:
                            if isinstance(attentions, tuple):
                                attentions_info = f"tuple of {len(attentions)} layers"
                                if len(attentions) > 0 and hasattr(attentions[0], 'shape'):
                                    attentions_info += f", first layer shape={attentions[0].shape}"
                            elif hasattr(attentions, 'shape'):
                                attentions_info = f"shape={attentions.shape}"
                            else:
                                attentions_info = f"type={type(attentions)}"
                        else:
                            attentions_info = "None"
                        #print(f"[DEBUG run_episode] Unpacked: actions type={type(actions)}, actions shape={actions.shape if hasattr(actions, 'shape') else len(actions) if isinstance(actions, (list, tuple)) else 'N/A'}, attentions={attentions_info}")
                    if attentions is not None:
                        if isinstance(attentions, tuple):
                            attentions_cpu = tuple(
                                layer_attn.cpu() if hasattr(layer_attn, "cpu") else layer_attn
                                for layer_attn in attentions
                            )
                        else:
                            attentions_cpu = (
                                attentions.cpu() if hasattr(attentions, "cpu") else attentions
                            )
                        del attentions
                        if cfg.enable_self_eval and cfg.self_eval_mode == MAE_D_MODE:
                            vis_lh = compute_visual_entropy_lh_from_attentions(
                                attentions_cpu,
                                num_patches,
                                num_prompt_tokens,
                                variant="openvla_oft",
                            )
                            if vis_lh is not None:
                                episode_visual_entropy_queries.append(vis_lh)
                        elif cfg.save_attentions and cfg.save_attn_dir and task_id >= 0 and episode_idx >= 0:
                            save_single_attention_weight(
                                cfg,
                                attentions_cpu,
                                task_id,
                                episode_idx,
                                query_idx,
                                task_description,
                                img,
                                False,
                                episode_number,
                                log_file,
                                num_patches=num_patches,
                                num_prompt_tokens=num_prompt_tokens,
                            )
                        del attentions_cpu
                        query_idx += 1
                        if query_idx % 10 == 0:
                            torch.cuda.empty_cache()
                elif cfg.track_uncertainty: # no attention tracking but save uncertainty
                    actions, chunk_uncertainty = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                        return_uncertainty=True,
                    )
                    if chunk_uncertainty is not None:
                        uncertainty_history.append(chunk_uncertainty)
                else: # no attention tracking or uncertainty tracking
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)
            if cfg.action_noise_std > 0:
                action = apply_action_noise(action, cfg.action_noise_std)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # Log uncertainty statistics if requested
    if cfg.track_uncertainty and uncertainty_history:
        avg_unc = float(np.mean(uncertainty_history))
        max_unc = float(np.max(uncertainty_history))
        min_unc = float(np.min(uncertainty_history))
        log_message(
            f"Uncertainty stats (bits) -> mean: {avg_unc:.3f}, min: {min_unc:.3f}, max: {max_unc:.3f}",
            log_file,
        )

    if (
        cfg.enable_self_eval
        and cfg.self_eval_mode == MAE_D_MODE
        and episode_visual_entropy_queries
        and self_eval_json_path
    ):
        mad_record = build_episode_mad_record(
            task_id=task_id + 1,
            task_description=task_description,
            episode_idx=episode_idx + 1,
            repeat_idx=repeat_idx + 1,
            episode_number=episode_number,
            is_success=bool(success),
            episode_visual_entropy_queries=episode_visual_entropy_queries,
            eval_ratios=cfg.attention_eval_ratios_list,
        )
        if mad_record is not None:
            append_online_mad_jsonl(mad_record, self_eval_json_path)

    return success, replay_images, None


def update_episode_success_status(
    cfg: GenerateConfig,
    task_id: int,
    episode_idx: int,
    episode_number: int,
    success: bool,
    log_file=None,
):
    """
    Update success status for all attention files belonging to an episode.
    
    Args:
        cfg: Configuration object
        task_id: Task ID
        episode_idx: Episode index
        episode_number: Episode number
        success: Final success status
        log_file: Optional log file handle
    """
    if not cfg.save_attn_dir:
        return
    
    try:
        # Find all files matching the episode pattern (both .pt and .json)
        import glob
        pattern_pt = f"task_{task_id}_episode_{episode_idx}_query_*_ep_{episode_number}.pt"
        pattern_json = f"task_{task_id}_episode_{episode_idx}_query_*_ep_{episode_number}.json"
        files_to_update = (
            glob.glob(os.path.join(cfg.save_attn_dir, pattern_pt)) +
            glob.glob(os.path.join(cfg.save_attn_dir, pattern_json))
        )
        
        if len(files_to_update) == 0:
            if log_file:
                log_file.write(f"[update_episode_success_status] No files found for patterns: {pattern_pt} or {pattern_json}\n")
                log_file.flush()
            return
        
        updated_count = 0
        for file_path in files_to_update:
            try:
                # Determine file format and load accordingly
                is_json = file_path.endswith('.json')
                
                if is_json:
                    # Load JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Update success status
                    if data.get('success') != success:
                        data['success'] = success
                        
                        # Save back with retry mechanism
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                temp_path = file_path + f".tmp{attempt}"
                                with open(temp_path, 'w') as f:
                                    json.dump(data, f, separators=(',', ':'))
                                os.rename(temp_path, file_path)
                                updated_count += 1
                                break
                            except (OSError, IOError) as e:
                                if attempt == max_retries - 1:
                                    error_msg = f"[update_episode_success_status] Failed to update {file_path}: {e}"
                                    if log_file:
                                        log_file.write(error_msg + "\n")
                                        log_file.flush()
                                    print(error_msg)
                else:
                    # Load PyTorch file
                    data = torch.load(file_path, map_location='cpu')
                    
                    # Two possible formats:
                    # 1) Legacy dict with 'success' key
                    # 2) New fast format tuple: (metadata, tensor_summary, tensor_chunks, tensor_tokens)
                    updated = False
                    if isinstance(data, dict):
                        # Legacy format
                        if data.get('success') != success:
                            data['success'] = success
                            updated = True
                    elif isinstance(data, tuple) and len(data) >= 1:
                        # Fast format with metadata tuple at index 0
                        metadata = data[0]
                        try:
                            # metadata layout:
                            # 0: task_id, 1: episode_idx, 2: query_idx,
                            # 3: episode_number, 4: success, 5: task_description,
                            # 6: valid_layer_indices, 7: num_patches, 8: num_prompt_tokens
                            if isinstance(metadata, tuple) and len(metadata) >= 5:
                                if metadata[4] != success:
                                    # Rebuild metadata tuple with updated success at index 4
                                    metadata = tuple(
                                        (success if i == 4 else v)
                                        for i, v in enumerate(metadata)
                                    )
                                    # Rebuild full tuple with updated metadata
                                    data = (metadata,) + tuple(data[1:])
                                    updated = True
                        except Exception as e:
                            error_msg = f"[update_episode_success_status] Failed to update fast-format metadata for {file_path}: {e}"
                            if log_file:
                                log_file.write(error_msg + "\n")
                                log_file.flush()
                            print(error_msg)
                    
                    if updated:
                        # Save back with retry mechanism
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                temp_path = file_path + f".tmp{attempt}"
                                torch.save(data, temp_path)
                                os.rename(temp_path, file_path)
                                updated_count += 1
                                break
                            except (RuntimeError, OSError, IOError) as e:
                                if attempt == max_retries - 1:
                                    error_msg = f"[update_episode_success_status] Failed to update {file_path}: {e}"
                                    if log_file:
                                        log_file.write(error_msg + "\n")
                                        log_file.flush()
                                    print(error_msg)
            except Exception as e:
                error_msg = f"[update_episode_success_status] Error updating {file_path}: {e}"
                if log_file:
                    log_file.write(error_msg + "\n")
                    log_file.flush()
                print(error_msg)
        
        if updated_count > 0:
            msg = f"[update_episode_success_status] Updated success status for {updated_count} files (task_{task_id}_episode_{episode_idx}_ep_{episode_number}): {success}"
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()
            print(msg)
    
    except Exception as e:
        error_msg = f"[update_episode_success_status] Unexpected error: {e}"
        if log_file:
            log_file.write(error_msg + "\n")
            log_file.flush()
        print(error_msg)
        import traceback
        traceback.print_exc()


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    result_dict,
    date_stamp,
    run_stamp,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    run_id: Optional[str] = None,
    self_eval_records=None,
    self_eval_json_path: str = "",
):
    """Run evaluation for a single task."""
    if self_eval_records is None:
        self_eval_records = []
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Initialize task info dictionary for saving
    task_info = {
        "task_id": task_id,
        "task_description": task_description,
        "task_suite_name": cfg.task_suite_name,
        "run_id": run_id,
        "num_trials_per_task": cfg.num_trials_per_task,
        "episodes": [],
        "task_episodes": 0,
        "task_successes": 0,
        "task_success_rate": 0.0,
    }

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        num_repeats = consistency_num_repeats(cfg)
        repeat_successes = []
        last_replay_images = None

        for repeat_idx in range(num_repeats):
            if num_repeats > 1:
                log_message(
                    f"Repeat {repeat_idx + 1}/{num_repeats} for init {episode_idx + 1}...",
                    log_file,
                )

            episode_number = total_episodes + 1

            result = run_episode(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
                log_file,
                task_id=task_id,
                episode_idx=episode_idx,
                episode_number=episode_number,
                self_eval_json_path=self_eval_json_path,
                repeat_idx=repeat_idx,
            )
            success, replay_images, _ = result
            last_replay_images = replay_images
            repeat_successes.append(bool(success))

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            if last_replay_images is not None:
                save_rollout_video(
                    cfg,
                    last_replay_images,
                    total_episodes,
                    success=success,
                    task_description=task_description,
                    date_stamp=date_stamp,
                    run_stamp=run_stamp,
                    log_file=log_file,
                )

            if cfg.save_attn_dir:
                update_episode_success_status(
                    cfg, task_id, episode_idx, episode_number, success, log_file
                )

            log_message(f"Success: {success}", log_file)
            log_message(f"# episodes completed so far: {total_episodes}", log_file)
            log_message(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)",
                log_file,
            )

        if cfg.enable_self_eval and cfg.self_eval_mode == MAE_D_MODE:
            if repeat_successes and repeat_successes[-1]:
                result_dict["success"].append((task_id, episode_idx))
            else:
                result_dict["failure"].append((task_id, episode_idx))
        elif cfg.enable_self_eval and cfg.self_eval_mode == "consistency":
            consistency_score = append_consistency_record(
                cfg,
                self_eval_records,
                self_eval_json_path,
                task_id,
                task_description,
                episode_idx,
                repeat_successes,
                num_repeats,
            )
            if consistency_score > 0:
                result_dict["success"].append((task_id, episode_idx))
            else:
                result_dict["failure"].append((task_id, episode_idx))
        else:
            if repeat_successes and repeat_successes[-1]:
                result_dict["success"].append((task_id, episode_idx))
            else:
                result_dict["failure"].append((task_id, episode_idx))

        results_dir = cfg.save_attn_dir or (
            os.path.dirname(self_eval_json_path) if self_eval_json_path else cfg.local_log_dir
        )
        result_path = os.path.join(results_dir, f"final_results{cfg.task_start}-{cfg.task_end}.yaml")
        if (episode_idx + 1) % 10 == 0:
            os.makedirs(results_dir, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                yaml.dump(result_dict, f, allow_unicode=True)

        task_info["episodes"].append({
            "episode_idx": episode_idx,
            "episode_number": total_episodes,
            "success": bool(repeat_successes[-1]) if repeat_successes else False,
            "consistency_score": (
                float(sum(repeat_successes) / num_repeats)
                if cfg.enable_self_eval and cfg.self_eval_mode == "consistency"
                else None
            ),
        })

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Update task info
    task_info["task_episodes"] = task_episodes
    task_info["task_successes"] = task_successes
    task_info["task_success_rate"] = task_success_rate

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # # Save task info to a separate file
    # try:
    #     os.makedirs(cfg.local_log_dir, exist_ok=True)
    #     # Create a safe filename from task description
    #     safe_task_desc = task_description.replace(" ", "_").replace("/", "_").replace("\\", "_")
    #     task_info_filename = f"task_{task_id}_{safe_task_desc}"
    #     if run_id:
    #         task_info_filename += f"_{run_id}"
    #     task_info_filepath = os.path.join(cfg.local_log_dir, f"{task_info_filename}.json")
        
    #     with open(task_info_filepath, 'w') as f:
    #         json.dump(task_info, f, indent=2)
    #     log_message(f"Saved task info to: {task_info_filepath}", log_file)
    # except Exception as e:
    #     error_msg = f"[run_task] Failed to save task info: {e}"
    #     log_message(error_msg, log_file)
    #     print(error_msg)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    logger.info(f"[eval_libero] Raw config from CLI / draccus: {cfg}")

    # Validate configuration
    validate_config(cfg)
    configure_self_eval(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Save original task_suite_name before it gets modified by evaluation config
    original_task_suite_name = cfg.task_suite_name
    
    # Set unnorm_key if not provided (use original task_suite_name before any modifications)
    if not cfg.unnorm_key or cfg.unnorm_key == "":
        cfg.unnorm_key = original_task_suite_name

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging (will be updated with task range after task suite initialization)
    # We need to initialize task suite first to get num_tasks, but task suite initialization
    # happens after evaluation_config_path processing. So we'll update logging after that.
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    # # Create attention save directory if needed
    # if cfg.save_attentions:
    #     if not cfg.save_attn_dir:
    #         cfg.save_attn_dir = os.path.join(cfg.local_log_dir, "attentions", run_id)
    #     os.makedirs(cfg.save_attn_dir, exist_ok=True)
    #     log_message(f"Saving attention matrices to: {cfg.save_attn_dir}", log_file)

    # Modify this section: Handle evaluation config for LIBERO-PRO perturbations
    if cfg.evaluation_config_path is not None:
        
        # [CRITICAL FIX] We need to redirect LIBERO to use our local project assets.
        # LIBERO looks for a config.yaml in the folder specified by LIBERO_CONFIG_PATH.
        # We will create a temporary config folder and file pointing to our local paths.
        
        config_dir = setup_libero_config(libero_pro_libero_root())
        print(f"[INFO] LIBERO_CONFIG_PATH set to: {config_dir}")

        with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
            evaluation_cfg = yaml.safe_load(f)

        evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
        evaluation_cfg["task_suite_name"] = cfg.task_suite_name

        use_swap = evaluation_cfg.get("use_swap", False)
        use_object = evaluation_cfg.get("use_object", False)
        use_language = evaluation_cfg.get("use_language", False)
        use_task = evaluation_cfg.get("use_task", False)
        use_environment = evaluation_cfg.get("use_environment", False)

        # Step 1: Check if only one of the use_xxx flags is True
        if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
            # If more than one flag is True, use the temp environment
            bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + cfg.task_suite_name + "_temp/"

            init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_temp/"

            # Check if the directories exist and the log.txt file contents match
            if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
                # If directories don't exist, create them and the log.txt file
                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                # Create the log.txt dynamically based on current flag values
                log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(log_content)  # Write the dynamic state to the log file

                perturbation.create_env(configs=evaluation_cfg)
            else:
                # If directories exist, check the contents of the log.txt file
                with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                    log_contents = log_file.read().strip()

                # Define the expected log content based on the current flags
                expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

                # If the log contents don't match, clean up and recreate the environment
                if log_contents != expected_log:
                    # Remove existing files in both directories
                    for folder in [bddl_file_path, init_file_path]:
                        for root, dirs, files in os.walk(folder, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                    # Create the environment again
                    os.makedirs(init_file_path, exist_ok=True)
                    os.makedirs(bddl_file_path, exist_ok=True)

                    # Write the updated log content based on current flags
                    with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                        log_file.write(expected_log)  # Write the updated log

                    perturbation.create_env(configs=evaluation_cfg)

            # Update task_suite_name with "_temp" suffix
            cfg.task_suite_name = cfg.task_suite_name + "_temp"

        # Step 2: Handle the case when only one use_xxx flag is True
        else:
            # [FIX] Generate a dynamic suffix based on the perturbation type
            # This ensures that different perturbation settings use different folders
            # reducing the need to manually delete the _temp folder.
            suffix = "_temp"
            if use_swap:
                suffix += "_swap"
            elif use_object:
                suffix += "_object"
            elif use_language:
                suffix += "_lang"
            elif use_task:
                suffix += "_task"
            elif use_environment:
                suffix += "_env"
            
            init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + suffix

            if not os.path.exists(init_file_path):
                # Pass the suffix to create_env so it generates the correct folder name
                perturbation.create_env(configs=evaluation_cfg, suffix=suffix)

            cfg.task_suite_name = cfg.task_suite_name + suffix
    
    else:
        print("No evaluation config path provided, using original task suite name")

    # Initialize LIBERO task suite
    # [CRITICAL FIX] Register missing objects from assets
    # Many objects like 'yellow_plate', 'red_bowl' exist as XML files but are not registered in LIBERO's python code.
    # We need to dynamically register them so they can be used in perturbations.
    
    from libero.libero.envs.base_object import register_object, register_visual_change_object
    from libero.libero.envs.objects.google_scanned_objects import GoogleScannedObject
    from libero.libero.envs.objects.articulated_objects import ArticulatedObject
    # Base object class for dynamic registrations and manual mug classes
    from robosuite.models.objects import MujocoXMLObject
    # Force import turbosquid_objects so built-in mug objects are registered
    from libero.libero.envs.objects import turbosquid_objects  # noqa: F401
    from libero.libero.envs import objects
    # import os  <-- REMOVED: os is already imported globally
    
    # Define the path to assets
    # We use the project_libero_root we defined earlier
    if 'project_libero_root' in locals():
        assets_root = os.path.join(project_libero_root, "assets")
    else:
        # Fallback if not defined (e.g. if not using config override)
        assets_root = os.environ.get("LIBERO_ASSET_ROOT", "")
        if not assets_root:
             # Try to guess from imported module
             import libero.libero
             assets_root = os.path.join(os.path.dirname(libero.libero.__file__), "assets")

    print(f"[INFO] Scanning for unregistered objects in: {assets_root}")

    # Explicitly register missing articulated / special objects that require specific logic
    if (
        "yellow_cabinet" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["yellow_cabinet"], "is_close")
    ):
        print("[INFO] Manually registering YellowCabinet")
        @register_object
        class YellowCabinet(ArticulatedObject):
            def __init__(
                self,
                name="yellow_cabinet",
                obj_name="yellow_cabinet",
                joints=[dict(type="free", damping="0.0005")],
            ):
                # [FIX] Use absolute path to the asset file
                if 'project_libero_root' in globals() or 'project_libero_root' in locals():
                     # If we defined a custom root earlier
                     root = locals().get('project_libero_root', globals().get('project_libero_root'))
                     xml_path = os.path.join(root, "assets/articulated_objects/yellow_cabinet.xml")
                else:
                     # Fallback to standard resolution
                     xml_path = "articulated_objects/yellow_cabinet.xml"
                     
                # If the path is absolute, MujocoXMLObject treats it as such.
                # If it's relative, it tries to find it relative to assets dir, BUT
                # ArticulatedObject's __init__ might be hardcoding a path construction.
                # Let's inspect ArticulatedObject's __init__ again?
                # Instead of calling super().__init__ which might enforce a broken path,
                # we can call MujocoXMLObject directly if needed, OR we override the logic.
                
                # Actually, ArticulatedObject.__init__ does:
                # super().__init__(os.path.join(str(absolute_path), f"assets/articulated_objects/{obj_name}.xml"), ...)
                # This 'absolute_path' inside articulated_objects.py is relative to the INSTALLED library,
                # NOT our local project folder. This is why it fails!
                
                # Solution: We must construct the path manually and call MujocoXMLObject directly,
                # bypassing ArticulatedObject's path construction but keeping its logic.
                
                from robosuite.models.objects import MujocoXMLObject
                import numpy as np
                
                # Construct correct path
                # We know assets_root is correct (e.g. .../libero/assets)
                correct_xml_path = os.path.join(assets_root, "articulated_objects/yellow_cabinet.xml")
                
                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                
                # Copied from ArticulatedObject.__init__
                self.category_name = "yellow_cabinet"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

                self.object_properties["articulation"]["default_open_ranges"] = [-0.16, -0.14]
                self.object_properties["articulation"]["default_close_ranges"] = [0.0, 0.005]

            def is_open(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_open_ranges"]):
                    return True
                else:
                    return False

            def is_close(self, qpos):
                if qpos > min(self.object_properties["articulation"]["default_close_ranges"]):
                    return True
                else:
                    return False

    if (
        "yellow_stove" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["yellow_stove"], "turn_on")
    ):
        print("[INFO] Manually registering YellowStove")
        @register_object
        @register_visual_change_object
        class YellowStove(ArticulatedObject):
            def __init__(
                self,
                name="yellow_stove",
                obj_name="yellow_stove",
                joints=[dict(type="free", damping="0.0005")],
            ):
                # Same fix for YellowStove
                from robosuite.models.objects import MujocoXMLObject
                import numpy as np
                
                correct_xml_path = os.path.join(assets_root, "articulated_objects/yellow_stove.xml")
                
                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                
                self.category_name = "yellow_stove"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }
                
                self.rotation = (0, 0)
                self.rotation_axis = "y"

                tracking_sites_dict = {}
                tracking_sites_dict["burner"] = (self.naming_prefix + "burner", False)
                self.object_properties["vis_site_names"].update(tracking_sites_dict)
                self.object_properties["articulation"]["default_turnon_ranges"] = [0.5, 2.1]
                self.object_properties["articulation"]["default_turnoff_ranges"] = [-0.005, 0.0]

            def turn_on(self, qpos):
                if qpos >= min(self.object_properties["articulation"]["default_turnon_ranges"]):
                    # TODO: Set visualization sites to be true
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        True,
                    )
                    return True
                else:
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        False,
                    )
                    return False

            def turn_off(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_turnoff_ranges"]):
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        False,
                    )
                    return True
                else:
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        True,
                    )
                    return False

    # Manually register mugs from local turbosquid assets to ensure availability
    if "porcelain_mug" not in objects.OBJECTS_DICT:
        print("[INFO] Manually registering PorcelainMug")

        @register_object
        class PorcelainMug(MujocoXMLObject):
            def __init__(
                self,
                name="porcelain_mug",
                obj_name="porcelain_mug",
                joints=[dict(type="free", damping="0.0005")],
            ):
                correct_xml_path = os.path.join(
                    assets_root, "turbosquid_objects/porcelain_mug/porcelain_mug.xml"
                )
                super().__init__(
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                self.category_name = "porcelain_mug"

                # Default tabletop rotation settings
                self.rotation = (0.0, 0.0)
                self.rotation_axis = "z"

                # Minimal object_properties to satisfy LIBERO expectations
                articulation_object_properties = {
                    "default_open_ranges": [0.0, 0.0],
                    "default_close_ranges": [0.0, 0.0],
                    "default_turnon_ranges": [0.0, 0.0],
                    "default_turnoff_ranges": [0.0, 0.0],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

    if "white_porcelain_mug" not in objects.OBJECTS_DICT:
        print("[INFO] Manually registering WhitePorcelainMug")

        @register_object
        class WhitePorcelainMug(MujocoXMLObject):
            def __init__(
                self,
                name="white_porcelain_mug",
                obj_name="white_porcelain_mug",
                joints=[dict(type="free", damping="0.0005")],
            ):
                correct_xml_path = os.path.join(
                    assets_root,
                    "turbosquid_objects/white_porcelain_mug/white_porcelain_mug.xml",
                )
                super().__init__(
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                self.category_name = "white_porcelain_mug"

                self.rotation = (0.0, 0.0)
                self.rotation_axis = "z"

                articulation_object_properties = {
                    "default_open_ranges": [0.0, 0.0],
                    "default_close_ranges": [0.0, 0.0],
                    "default_turnon_ranges": [0.0, 0.0],
                    "default_turnoff_ranges": [0.0, 0.0],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

    if (
        "wooden_cabinet" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["wooden_cabinet"], "is_close")
    ):
        print("[INFO] Manually registering WoodenCabinet with articulated behavior")

        @register_object
        class WoodenCabinet(ArticulatedObject):
            def __init__(
                self,
                name="wooden_cabinet",
                obj_name="wooden_cabinet",
                joints=[dict(type="free", damping="0.0005")],
            ):
                from robosuite.models.objects import MujocoXMLObject

                correct_xml_path = os.path.join(
                    assets_root, "articulated_objects/wooden_cabinet.xml"
                )

                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )

                self.category_name = "wooden_cabinet"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

                self.object_properties["articulation"]["default_open_ranges"] = [
                    -0.16,
                    -0.14,
                ]
                self.object_properties["articulation"]["default_close_ranges"] = [
                    0.0,
                    0.005,
                ]

            def is_open(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_open_ranges"]):
                    return True
                else:
                    return False

            def is_close(self, qpos):
                if qpos > min(self.object_properties["articulation"]["default_close_ranges"]):
                    return True
                else:
                    return False
    
    # Helper to register objects
    def register_from_folder(folder_name, base_class):
        # Explicitly reference the re module from outer scope
        import re as re_module
        search_path = os.path.join(assets_root, folder_name)
        if not os.path.exists(search_path):
            print(f"[WARN] Asset folder not found: {search_path}")
            return

        count = 0
        for obj_name in os.listdir(search_path):
            full_path = os.path.join(search_path, obj_name)
            
            # Strategy 1: obj_name is a directory containing obj_name.xml
            # Example: assets/stable_scanned_objects/apple/apple.xml
            if os.path.isdir(full_path):
                xml_path = os.path.join(full_path, f"{obj_name}.xml")
                if os.path.exists(xml_path):
                    # Register logic (same as before)
                    if obj_name.lower() not in objects.OBJECTS_DICT:
                        class_name = "".join(x.title() for x in obj_name.split("_"))
                        def make_init(x_path, o_name):
                            def __init__(self, name=o_name, obj_name=o_name, joints=[dict(type="free", damping="0.0005")]):
                                from robosuite.models.objects import MujocoXMLObject
                                import numpy as np
                                MujocoXMLObject.__init__(self, x_path, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
                                self.category_name = o_name 
                                self.rotation = (np.pi / 2, np.pi / 2)
                                self.rotation_axis = "x"
                                self.object_properties = {"vis_site_names": {}}
                            return __init__
                        new_class = type(class_name, (MujocoXMLObject,), {"__init__": make_init(xml_path, obj_name)})
                        register_object(new_class)
                        count += 1
            
            # Strategy 2: obj_name is an XML file directly
            # Example: assets/articulated_objects/yellow_cabinet.xml
            elif obj_name.endswith(".xml"):
                # obj_name is "yellow_cabinet.xml", we want "yellow_cabinet"
                pure_name = os.path.splitext(obj_name)[0]
                
                # SKIP if it's already manually registered above (prevents overwriting our custom classes)
                if pure_name in [
                    "yellow_cabinet",
                    "yellow_stove",
                    "wooden_cabinet",
                    "porcelain_mug",
                    "white_porcelain_mug",
                ]:
                    continue

                xml_path = full_path
                
                # Force override or register
                class_name = "".join(x.title() for x in pure_name.split("_"))
                def make_init(x_path, o_name):
                    def __init__(self, name=o_name, obj_name=o_name, joints=[dict(type="free", damping="0.0005")]):
                        from robosuite.models.objects import MujocoXMLObject
                        import numpy as np
                        MujocoXMLObject.__init__(self, x_path, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
                        self.category_name = o_name 
                        
                        # Initialize rotation (default x-axis for tabletop objects)
                        self.rotation = (np.pi / 2, np.pi / 2)
                        self.rotation_axis = "x"
                        
                        # Initialize object properties properly
                        # 'articulation' key is needed for articulated objects (e.g., microwave, stove, cabinet)
                        # 'vis_site_names' is needed for all objects
                        # Set default ranges to [0.0, 0.0] instead of [] to prevent IndexError in OpenCloseSampler
                        # If the object is actually articulated, these will be overridden by the specific class
                        articulation_object_properties = {
                            "default_open_ranges": [0.0, 0.0],
                            "default_close_ranges": [0.0, 0.0],
                            "default_turnon_ranges": [0.0, 0.0],
                            "default_turnoff_ranges": [0.0, 0.0],
                        }
                        self.object_properties = {
                            "articulation": articulation_object_properties,
                            "vis_site_names": {},
                        }
                    return __init__
                new_class = type(class_name, (MujocoXMLObject,), {"__init__": make_init(xml_path, pure_name)})
                
                # Manually register to bypass assert check in register_object.
                # Skip if a class with this key already exists (preserve articulated behaviors).
                key = "_".join(re_module.sub(r"([A-Z0-9])", r" \1", new_class.__name__).split()).lower()
                if key not in objects.OBJECTS_DICT:
                    objects.OBJECTS_DICT[key] = new_class
                    count += 1

        print(f"[INFO] Registered {count} new objects from {folder_name}")

    # Scan stable_scanned_objects and stable_hope_objects
    # Note: We use MujocoXMLObject as the base for dynamic registration, mimicking the notebook's CustomObjects
    register_from_folder("stable_scanned_objects", MujocoXMLObject)
    register_from_folder("stable_hope_objects", MujocoXMLObject)
    # Turbosquid objects structure might be similar
    register_from_folder("turbosquid_objects", MujocoXMLObject)

    # Also register objects from articulated_objects
    # articulated_objects can be complex but many are just XML models.
    # We attempt to register them similarly.
    register_from_folder("articulated_objects", MujocoXMLObject)

    # Print all registered objects
    print("\n" + "="*80)
    print("[INFO] All registered objects in OBJECTS_DICT:")
    print("="*80)
    registered_objects = sorted(objects.OBJECTS_DICT.keys())
    for i, obj_name in enumerate(registered_objects, 1):
        obj_class = objects.OBJECTS_DICT[obj_name]
        print(f"  {i:3d}. {obj_name:40s} -> {obj_class.__name__}")
    print(f"\n[INFO] Total: {len(registered_objects)} registered objects")
    print("="*80 + "\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[original_task_suite_name]()
    
    # [Critical Fix] Force update the BDDL file paths if we are using a perturbed suite
    if cfg.task_suite_name != original_task_suite_name:
        print(f"[INFO] Overriding BDDL paths to use perturbed suite: {cfg.task_suite_name}")
        
        # Determine the new folder name (e.g., "libero_spatial_temp" or "libero_spatial_env")
        new_folder_name = cfg.task_suite_name
        
        # We need to monkey-patch the tasks within the suite
        from libero.libero.benchmark import Task, grab_language_from_filename
        import re

        def extract_language_from_bddl(bddl_path):
            try:
                with open(bddl_path, 'r') as f:
                    content = f.read()
                match = re.search(r"\(:language\s*(.*?)\)", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
            except Exception as e:
                print(f"[WARNING] Failed to extract language from {bddl_path}: {e}")
            return None

        new_tasks = []
        for old_task in task_suite.tasks:
            # Construct the full path to the NEW bddl file
            # BDDL files are in: [benchmark_root]/bddl_files/[problem_folder]/[bddl_file]
            # We need to construct this path using our temp config settings
            
            # Assuming our temp config set bddl_files to: project_libero_root/bddl_files
            # and new_folder_name is e.g. "libero_spatial_temp"
            
            project_libero_root = str(libero_pro_libero_root())
            new_bddl_full_path = os.path.join(project_libero_root, "bddl_files", new_folder_name, old_task.bddl_file)
            
            # Extract new language
            new_language = extract_language_from_bddl(new_bddl_full_path)
            if new_language:
                print(f"[INFO] Updated language for {old_task.name}: {new_language}")
            else:
                new_language = old_task.language
                print(f"[WARN] Could not update language for {old_task.name}, keeping original.")

            # Construct new path structure based on how your perturbation.py creates folders
            # Assuming perturbation creates a parallel folder structure
            new_task = Task(
                name=old_task.name,
                language=new_language,  # <--- Updated language!
                problem=old_task.problem,
                problem_folder=new_folder_name,  # <--- Point to the new folder
                bddl_file=old_task.bddl_file,
                init_states_file=old_task.init_states_file 
            )
            new_tasks.append(new_task)
        
        # Replace the tasks list in the suite
        task_suite.tasks = new_tasks
        
    num_tasks = task_suite.n_tasks

    # Determine task range (align with submit_* scripts)
    task_start = 0 if cfg.task_start == -1 else cfg.task_start
    task_end = num_tasks if cfg.task_end == -1 else cfg.task_end

    cfg.task_start = task_start
    cfg.task_end = task_end

    if task_start < 0 or task_start >= num_tasks:
        raise ValueError(f"task_start ({task_start}) must be in range [0, {num_tasks})")
    if task_end <= task_start or task_end > num_tasks:
        raise ValueError(f"task_end ({task_end}) must be in range ({task_start}, {num_tasks}]")

    # Update run_id and log file with task range to avoid filename collisions
    # Close old log file and recreate with task range in run_id
    if log_file:
        log_file.close()
    
    # Recreate logging with task range included in run_id
    cfg.local_log_dir = os.path.join(cfg.local_log_dir, cfg.pretrained_checkpoint.split('/')[-1], cfg.task_suite_name)
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_start=task_start, task_end=task_end)
    #logger.info(f"Logging to local log file (with task range): {local_log_filepath}")
    
    # Create attention save directory if needed
    date_stamp = os.getenv('DATE_STAMP', None)
    run_stamp = os.getenv('RUN_STAMP', None)

    self_eval_records = []
    self_eval_json_path = ""
    if cfg.enable_self_eval:
        results_output_dir = resolve_self_eval_output_dir(cfg, date_stamp=date_stamp, run_stamp=run_stamp)
        os.makedirs(results_output_dir, exist_ok=True)
        self_eval_json_path = os.path.join(
            results_output_dir,
            self_eval_json_filename(
                cfg.self_eval_mode,
                attention_eval_output_name=cfg.attention_eval_output_name,
            ),
        )
        log_message(
            f"[eval_libero] Self-eval ENABLED | mode={cfg.self_eval_mode} | json={self_eval_json_path}",
            log_file,
        )
        if cfg.self_eval_mode == MAE_D_MODE:
            log_message(
                f"[eval_libero] Online MAE-D eval | ratios={cfg.attention_eval_ratios_list} | no attention save",
                log_file,
            )
        else:
            log_message(
                f"[eval_libero] Consistency repeats per init: {cfg.consistency_repeats_per_init}",
                log_file,
            )
            log_message(
                f"[eval_libero] Action noise std: {cfg.action_noise_std}",
                log_file,
            )

    if cfg.save_attentions:
        if cfg.save_attn_dir:
            cfg.save_attn_dir = os.path.join(cfg.save_attn_dir, cfg.task_suite_name, date_stamp, run_stamp)
        else:
            cfg.save_attn_dir = os.path.join('saved_attention/openvla-oft', cfg.pretrained_checkpoint.split('/')[-1], cfg.task_suite_name, date_stamp, run_stamp)
            
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        log_message(f"Saving attention matrices to: {cfg.save_attn_dir}", log_file)

    # Update attention save directory if needed (since run_id changed)
    if cfg.save_attentions:
        # If save_attn_dir was auto-generated from run_id, update it
        old_run_id_base = run_id.split("-")[0] if "-" in run_id else run_id
        if cfg.save_attn_dir and old_run_id_base in cfg.save_attn_dir:
            cfg.save_attn_dir = os.path.join(cfg.local_log_dir, "attentions", run_id)
            os.makedirs(cfg.save_attn_dir, exist_ok=True)
            log_message(f"Updated attention save directory to: {cfg.save_attn_dir}", log_file)
        elif not cfg.save_attn_dir:
            # If save_attn_dir was not set, create it with new run_id
            cfg.save_attn_dir = os.path.join(cfg.local_log_dir, "attentions", run_id)
            os.makedirs(cfg.save_attn_dir, exist_ok=True)
            log_message(f"Attention save directory: {cfg.save_attn_dir}", log_file)

    tasks_to_evaluate = list(range(task_start, task_end))
    num_tasks_to_evaluate = len(tasks_to_evaluate)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Evaluating tasks: {task_start} to {task_end-1} (inclusive, {num_tasks_to_evaluate} tasks)", log_file)

    # Start evaluation (sequential)
    total_episodes, total_successes = 0, 0
    result = {'success': [], 'failure':[]}  # Store individual task results for potential analysis
    for task_id in tqdm.tqdm(tasks_to_evaluate, desc="Running tasks sequentially"):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            result,
            date_stamp,
            run_stamp,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            run_id=run_id,
            self_eval_records=self_eval_records,
            self_eval_json_path=self_eval_json_path,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()



    return final_success_rate


if __name__ == "__main__":
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    eval_libero()
