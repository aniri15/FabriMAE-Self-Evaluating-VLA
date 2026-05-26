"""
run_multiprocess_libero_eval.py — LIBERO evaluation for OpenVLA-OFT.

Note (MAE naming): user-facing metric MAE-D (self_eval_mode=mae-d) uses internal mad
computation (top-k visual entropy). MAE-C is PI-only.
"""

import json
import logging
import os
import sys
import debugpy
import yaml
from collections import deque
# Removed ThreadPoolExecutor - tasks now run sequentially
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import torch
import wandb
from typing import Any
import scipy.stats

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.repo_paths import bundled_libero_root, setup_libero_config
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


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
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
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
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
    max_parallel_tasks: int = 2                      # Number of LIBERO tasks to evaluate concurrently (deprecated: now always sequential)
    task_start: int = -1                             # Start task ID (inclusive, -1 means start from 0)
    task_end: int = -1                               # End task ID (exclusive, -1 means process all tasks)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    track_uncertainty: bool = False                  # Whether to compute Shannon entropy per action query

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    # Attention logging (used by submit_original / submit_perturbations)
    save_attentions: bool = False                    # Whether to save attention matrices (if supported by model)
    save_attn_dir: str = ""                          # Directory to save attention data
    save_attn_format: str = "json"                     # Format to save attention: "json" or "pt" (default: "pt" for PyTorch format)

    enable_self_eval: bool = False                   # Enable self-consistency scoring during rollout
    self_eval_mode: str = "consistency"              # consistency | mae-d
    consistency_repeats_per_init: int = 3            # Same fixed init scene repeats (consistency mode)
    action_noise_std: float = 0.0                    # Stddev of Gaussian noise on first 6 action dims
    self_eval_output_dir: str = ""                   # Optional override for self-eval JSON output dir
    attention_eval_method: str = "mae-d"             # mae-d only (MAE-C is PI-only)
    attention_eval_ratios: str = "0.01"              # Comma-separated top-k ratios for online MAE-D
    attention_eval_output_name: str = DEFAULT_MAE_D_OUTPUT
    libero_benchmark: str = "libero"                 # libero | libero_reflect

    seed: int = 7                                    # Random Seed (for reproducibility)

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

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


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
            
            torch.save((metadata, tensor_summary, tensor_chunks, tensor_tokens), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

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
    
    # Handle dynamic task suite names (e.g., libero_spatial_temp_swap)
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
            try:
                obs, reward, done, info = env.step(action.tolist())
            except Exception as step_err:
                log_message(
                    f"[run_episode] Error in env.step at t={t}: {step_err}",
                    log_file,
                )
                raise
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        import traceback
        error_msg = f"Episode error: {e}\n{traceback.format_exc()}"
        log_message(error_msg, log_file)

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

    if getattr(cfg, "libero_benchmark", "libero") in ("libero_reflect", "libero_pro"):
        from experiments.robot.libero.repo_paths import libero_pro_libero_root

        setup_libero_config(libero_pro_libero_root())
    else:
        setup_libero_config(bundled_libero_root())

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # # Setup logging
    # log_file, local_log_filepath, run_id = setup_logging(cfg, task_start = task_start, task_end = task_end)

    # # If attention saving is requested, make sure directory is set up
    # if cfg.save_attentions:
    #     # If caller did not pass an explicit directory, create one under local_log_dir
    #     if not cfg.save_attn_dir:
    #         cfg.save_attn_dir = os.path.join(cfg.local_log_dir, "attentions", run_id)
    #     os.makedirs(cfg.save_attn_dir, exist_ok=True)
    #     log_message(
    #         f"[eval_libero] Attention saving ENABLED | save_attn_dir={cfg.save_attn_dir}",
    #         log_file,
    #     )
    # else:
    #     log_message("[eval_libero] Attention saving DISABLED", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine task range
    task_start = cfg.task_start if cfg.task_start >= 0 else 0
    task_end = cfg.task_end if cfg.task_end >= 0 else num_tasks
    
    # Validate task range
    if task_start < 0 or task_start >= num_tasks:
        raise ValueError(f"task_start ({task_start}) must be in range [0, {num_tasks})")
    if task_end <= task_start or task_end > num_tasks:
        raise ValueError(f"task_end ({task_end}) must be in range ({task_start}, {num_tasks}]")

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_start = task_start, task_end = task_end)

    # If attention saving is requested, make sure directory is set up
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
            cfg.save_attn_dir = os.path.join('saved_attention/openvla', cfg.pretrained_checkpoint.split('/')[-1], cfg.task_suite_name, date_stamp, run_stamp)
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        log_message(
            f"[eval_libero] Attention saving ENABLED | save_attn_dir={cfg.save_attn_dir}",
            log_file,
        )
    else:
        log_message("[eval_libero] Attention saving DISABLED", log_file)

    # Update run_id and log file with task range to avoid filename collisions
    # Close old log file and recreate with task range in run_id
    #if log_file:
    #    log_file.close()
    
    # Recreate logging with task range included in run_id
    #log_file, local_log_filepath, run_id = setup_logging(cfg, task_start=task_start, task_end=task_end)
    #logger.info(f"Logging to local log file (with task range): {local_log_filepath}")
    
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

    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Total tasks in suite: {num_tasks}")
    print(f"Evaluating tasks: {task_start} to {task_end-1} (inclusive, {num_tasks_to_evaluate} tasks)")
    print(f"benchmark_dict: {benchmark_dict}")
    print(f"task_suite: {task_suite}")

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Evaluating tasks: {task_start} to {task_end-1} ({num_tasks_to_evaluate} tasks)", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    result = {'success': [], 'failure':[]}
    # Run tasks sequentially (no internal parallelism)
    logger.info(f"Evaluating {num_tasks_to_evaluate} tasks (range [{task_start}, {task_end})) sequentially")

    for task_id in tasks_to_evaluate:
        # Use disjoint episode numbering ranges to avoid filename collisions
        #episode_offset = task_id * cfg.num_trials_per_task
        try:
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
            #task_eps = task_total_eps - episode_offset
            #total_episodes += task_eps
            #total_successes += task_total_succ
            logger.info(f"[task {task_id}] finished: episodes={total_episodes}, successes={total_successes}")
        except Exception as e:
            logger.error(f"[task {task_id}] failed: {e}", exc_info=True)
            log_message(f"Task {task_id} generated an exception: {e}", log_file)

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
