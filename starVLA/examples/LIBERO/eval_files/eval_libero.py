"""
eval_libero.py — LIBERO client for StarVLA / PI policy server.

Note (MAE naming): user-facing metric MAE-C (attention_eval_method=mae-c) uses internal mac
computation (bottom-k visual entropy). MAE-D is OpenVLA/OFT-only.
"""
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time
import torch

import imageio
import numpy as np
import tqdm
import tyro
from typing import Any, Optional
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.LIBERO.eval_files.action_consistency_utils import (
    build_action_consistency_record,
    compute_action_agreement,
    save_repeat_action_array,
)
from examples.LIBERO.eval_files.mae_metric_naming import (
    DEFAULT_MAE_C_OUTPUT,
    MAE_C_METHOD,
    MAE_D_MODE,
    MAE_NAMING_NOTE,
    external_metric_label,
    internal_compute_metric,
    mae_c_score_key,
    normalize_attention_method,
)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    consistency_repeats_per_init: int = 1  # Repeat each fixed initial state this many times

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    save_attention = True
    action_noise_std: float = 0.0  # stddev of Gaussian noise added to first 6 action dims during eval

    save_attn_dir: str = "saved_attentions/Pi"
    attention_eval_mode: str = "save"  # save | eval
    attention_eval_method: str = "mae-c"  # mae-c | mae-d
    attention_eval_ratios: str = "0.01"
    attention_eval_output_name: str = DEFAULT_MAE_C_OUTPUT
    attention_eval_action_output_name: str = "online_action_consistency_scores.jsonl"
    save_repeat_actions: bool = True
    eval_results_root: str = "eval_results"


def _parse_ratios(text: str):
    ratios = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = float(chunk)
        if val <= 0 or val > 1:
            raise ValueError(f"Invalid ratio: {val}")
        ratios.append(val)
    if not ratios:
        raise ValueError("No valid ratios parsed")
    return ratios


def _stack_trim_mats(mats):
    min_l = min(m.shape[0] for m in mats)
    min_h = min(m.shape[1] for m in mats)
    trimmed = [m[:min_l, :min_h] for m in mats]
    return np.stack(trimmed, axis=0)


def _compute_layer_k_entropy_score(entropy_lh, ratio, select_bottom, negate_mean):
    num_layers, num_heads = entropy_lh.shape
    k = max(1, int(np.ceil(num_heads * ratio)))
    scores = np.zeros(num_layers, dtype=np.float64)
    for l in range(num_layers):
        row = np.nan_to_num(entropy_lh[l], nan=0.0, posinf=0.0, neginf=0.0)
        if select_bottom:
            idx = np.argsort(row)[:k]
        else:
            idx = np.argsort(row)[-k:]
        val = float(np.mean(row[idx]))
        scores[l] = -val if negate_mean else val
    return scores


def _compute_layer_k_entropy_scores_for_ratios(entropy_lh, ratios, select_bottom, negate_mean):
    """
    Compute layer-wise top-k entropy means for multiple ratios efficiently.
    Sort each layer once, then reuse cumulative sums across all ratios.
    """
    arr = np.nan_to_num(np.asarray(entropy_lh, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected (L, H) entropy matrix, got shape: {arr.shape}")
    num_layers, num_heads = arr.shape
    if num_layers == 0 or num_heads == 0:
        return {ratio: np.zeros((0,), dtype=np.float64) for ratio in ratios}

    sorted_rows = np.sort(arr, axis=1)  # ascending
    cumsum_rows = np.cumsum(sorted_rows, axis=1)
    total_rows = cumsum_rows[:, -1]

    ratio_scores = {}
    for ratio in ratios:
        k = max(1, int(np.ceil(num_heads * ratio)))
        if select_bottom:
            vals = cumsum_rows[:, k - 1] / k
        else:
            vals = (total_rows - cumsum_rows[:, num_heads - k - 1]) / k if k < num_heads else total_rows / k
        if negate_mean:
            vals = -vals
        ratio_scores[ratio] = vals
    return ratio_scores


def _compute_visual_entropy_lh_from_attentions(attentions):
    """
    Match save_attention processing exactly:
    use attentions['visual_attn'][-1], entropy over token axis, then mean over 8 action rows.
    Returns (L, H) visual entropy matrix for one query file.
    """
    if not isinstance(attentions, dict):
        return None
    visual_attns = attentions.get("visual_attn", None)
    if visual_attns is None or len(visual_attns) == 0:
        return None

    visual_attns = visual_attns[-1]
    layers_vis_entropy = []

    def batch_entropy(probs, axis=-1, epsilon=1e-12):
        row_sums = np.sum(probs, axis=axis, keepdims=True)
        probs_norm = probs / (row_sums + epsilon)
        probs_norm = np.clip(probs_norm, epsilon, 1.0)
        return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

    for visual_attn in visual_attns:
        if visual_attn is None:
            continue
        # visual_attn: (1, H, 8, V). Vectorize over all heads to reduce Python overhead.
        vis_probs = np.asarray(visual_attn[0], dtype=np.float64)
        vis_ent = batch_entropy(vis_probs)  # (H, 8)
        head_vis_entropy = np.mean(vis_ent, axis=-1).astype(np.float32)  # mean over 8 action queries
        layers_vis_entropy.append(head_vis_entropy)

    if not layers_vis_entropy:
        return None
    return np.stack(layers_vis_entropy, axis=0)  # (L, H)

def save_single_attention_weight(
    save_attn_dir: str,
    attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    img: Any,
    success: bool,
    episode_number: int,
    log_file=None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not save_attn_dir or attentions is None:
        return

    try:
        os.makedirs(save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        ACTION_DIM = 7
        NUM_ACTIONS_CHUNK = 8
        NUM_ACTION_TOKENS = 56 # 7 * 8
        GRID_SHAPE = (NUM_ACTIONS_CHUNK, ACTION_DIM) # (8, 7)


        # Containers for stacking
        layers_summary = [] # List of (Head, 6, 8, 7)
        layers_chunks = []  # List of (Head, 8, 8, 7)
        layers_tokens = []  # List of (Head, 56, 8, 7)
        valid_layer_indices = []
        layer_visuals = []

        # Helper: Entropy
        def batch_entropy(probs, axis=-1, epsilon=1e-12):
            row_sums = np.sum(probs, axis=axis, keepdims=True)
            probs_norm = probs / (row_sums + epsilon)
            probs_norm = np.clip(probs_norm, epsilon, 1.0)
            return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

        # --- Processing Loop ---
        if isinstance(attentions, dict):
            visual_attns = attentions["visual_attn"][-1]
            text_attns = attentions["text_attn"][-1]
            for l, visual_attn in enumerate(visual_attns):
                text_attn = text_attns[l]
                if visual_attn is None or text_attn is None:
                    continue
                
                # Move to CPU & Numpy
                # if hasattr(layer_attn, 'cpu'):
                #     layer_attn = layer_attn.float().cpu().numpy()
                # else:
                #     layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                num_heads = visual_attn.shape[1]
                num_patches = visual_attn.shape[-1]
                num_prompt_tokens = text_attn.shape[-1]
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 4, NUM_ACTIONS_CHUNK), dtype=np.float32)
                # Chunks: 8 chunks
                # head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                # head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                head_visual_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, num_patches))

                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    

                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_sum = visual_attn[0, h].sum(axis=-1)
                    txt_sum = text_attn[0, h].sum(axis=-1)
                    
                    # Entropies: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_ent = batch_entropy(visual_attn[0, h])
                    txt_ent = batch_entropy(text_attn[0, h])
                    
                    head_visual_stack[h] = visual_attn[0, h]
                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = vis_ent
                    head_summary_stack[h, 3] = txt_ent

                    # 2. Chunk Sums (Vectorized)
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    # (56, seq_len) -> (56, 56)
                    # act_to_act = head_block[:, action_start:action_end] # (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index

                    # for i in range(NUM_ACTIONS_CHUNK):
                    #     c_start = i * ACTION_DIM
                    #     c_end = c_start + ACTION_DIM
                    #     # Sum cols (attention TO this chunk)
                    #     # (56,) -> (8, 7)
                    #     head_chunk_stack[h, i] = act_to_act[:, c_start:c_end].sum(axis=1).reshape(GRID_SHAPE)

                    # 3. Token Values (Vectorized)
                    # act_to_act is (56 rows queries, 56 cols targets)
                    # We want to save columns. column j = attention TO token j FROM all 56 queries.
                    # We need to reshape each column to (8, 7)
                    # act_to_act.T is (56 targets, 56 queries)
                    # Reshape to (56 targets, 8, 7)

                    # head_token_stack[h] = act_to_act.T.reshape(NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM)

                # Store layer result
                layers_summary.append(head_summary_stack)
                layer_visuals.append(head_visual_stack)
                # layers_chunks.append(head_chunk_stack)
                # layers_tokens.append(head_token_stack)
                valid_layer_indices.append(l)

                del head_summary_stack, head_visual_stack

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary))
            tensor_visuals =  torch.from_numpy(np.stack(layer_visuals))# (L, H, 6, 8, 7)
            # tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            # tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)

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
            save_path = os.path.join(save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            torch.save((metadata, tensor_summary, tensor_visuals, img), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

def save_single_attention_query(
    save_attn_dir: str,
    attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    success: bool,
    episode_number: int,
    log_file=None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not save_attn_dir or attentions is None:
        return

    try:
        os.makedirs(save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        ACTION_DIM = 7
        NUM_ACTIONS_CHUNK = 8
        NUM_ACTION_TOKENS = 56 # 7 * 8
        GRID_SHAPE = (NUM_ACTIONS_CHUNK, ACTION_DIM) # (8, 7)


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
        if isinstance(attentions, dict):
            visual_attns = attentions["visual_attn"]
            text_attns = attentions["text_attn"]
            for l, visual_attn in enumerate(visual_attns):
                text_attn = text_attns[l]
                if visual_attn is None or text_attn is None:
                    continue
                
                # Move to CPU & Numpy
                # if hasattr(layer_attn, 'cpu'):
                #     layer_attn = layer_attn.float().cpu().numpy()
                # else:
                #     layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                num_heads = visual_attn.shape[1]
                num_patches = visual_attn.shape[-1]
                num_prompt_tokens = text_attn.shape[-1]
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 4, NUM_ACTIONS_CHUNK), dtype=np.float32)
                # Chunks: 8 chunks
                # head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                # head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)

                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    

                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_sum = visual_attn[0, h].sum(axis=-1)
                    txt_sum = text_attn[0, h].sum(axis=-1)
                    
                    # Entropies: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_ent = batch_entropy(visual_attn[0, h])
                    txt_ent = batch_entropy(text_attn[0, h])
                    

                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = vis_ent
                    head_summary_stack[h, 3] = txt_ent

                    # 2. Chunk Sums (Vectorized)
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    # (56, seq_len) -> (56, 56)
                    # act_to_act = head_block[:, action_start:action_end] # (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index

                    # for i in range(NUM_ACTIONS_CHUNK):
                    #     c_start = i * ACTION_DIM
                    #     c_end = c_start + ACTION_DIM
                    #     # Sum cols (attention TO this chunk)
                    #     # (56,) -> (8, 7)
                    #     head_chunk_stack[h, i] = act_to_act[:, c_start:c_end].sum(axis=1).reshape(GRID_SHAPE)

                    # 3. Token Values (Vectorized)
                    # act_to_act is (56 rows queries, 56 cols targets)
                    # We want to save columns. column j = attention TO token j FROM all 56 queries.
                    # We need to reshape each column to (8, 7)
                    # act_to_act.T is (56 targets, 56 queries)
                    # Reshape to (56 targets, 8, 7)

                    # head_token_stack[h] = act_to_act.T.reshape(NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM)

                # Store layer result
                layers_summary.append(head_summary_stack)
                # layers_chunks.append(head_chunk_stack)
                # layers_tokens.append(head_token_stack)
                valid_layer_indices.append(l)

                del head_summary_stack 

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary)) # (L, H, 6, 8, 7)
            # tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            # tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)

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
            save_path = os.path.join(save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            torch.save((metadata, tensor_summary), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    repo_root = Path(__file__).resolve().parents[3]

    def _resolve_repo_path(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else (repo_root / path).resolve())

    args.save_attn_dir = _resolve_repo_path(args.save_attn_dir)
    args.eval_results_root = _resolve_repo_path(args.eval_results_root)
    args.video_out_path = _resolve_repo_path(args.video_out_path)

    # Set random seed
    np.random.seed(args.seed)
    if args.action_noise_std < 0:
        raise ValueError(f"action_noise_std must be >= 0, got {args.action_noise_std}")
    if args.consistency_repeats_per_init <= 0:
        raise ValueError(
            f"consistency_repeats_per_init must be > 0, got {args.consistency_repeats_per_init}"
        )

    date_stamp = os.getenv('DATE_STAMP', None)
    run_stamp = os.getenv('RUN_STAMP', None)
    args.save_attn_dir = os.path.join(args.save_attn_dir, args.task_suite_name) + f'/{date_stamp}/{run_stamp}'
    args.video_out_path = os.path.join(args.video_out_path, args.task_suite_name) + f'/{date_stamp}/{run_stamp}'
    if args.attention_eval_mode == "eval":
        eval_mode_root = Path(args.eval_results_root) / args.task_suite_name / date_stamp / run_stamp
        args.save_attn_dir = str(eval_mode_root / "attention")
        args.video_out_path = str(eval_mode_root / "videos")
        action_dir = str(eval_mode_root / "actions")
    else:
        action_dir = os.path.join(args.save_attn_dir, "actions")
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )


    if args.attention_eval_mode not in ("save", "eval"):
        raise ValueError(f"Unsupported attention_eval_mode: {args.attention_eval_mode}")
    logging.info(f"[PI self-eval] {MAE_NAMING_NOTE}")
    args.attention_eval_method = normalize_attention_method(args.attention_eval_method)
    if args.attention_eval_mode == "eval" and args.attention_eval_method != MAE_C_METHOD:
        logging.warning(
            "Pi eval mode only supports visual MAE-C. "
            f"Override attention_eval_method to '{MAE_C_METHOD}'."
        )
        args.attention_eval_method = MAE_C_METHOD
    if internal_compute_metric(args.attention_eval_method) != "mac" and args.attention_eval_mode == "eval":
        raise ValueError("PI online attention eval requires attention_eval_method=mae-c")
    eval_ratios = _parse_ratios(args.attention_eval_ratios)
    eval_wall_start = time.time()

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_policy_infer_sec = 0.0
    total_attn_to_mac_sec = 0.0
    total_attn_to_mac_steps = 0
    task_attn_timing = {}
    online_eval_records = []
    online_action_consistency_records = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            scenario_repeat_actions = []
            scenario_repeat_meta = []
            for repeat_idx in range(args.consistency_repeats_per_init):
                logging.info(f"\nTask: {task_description}")
                client_model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                replay_images = []
                full_actions = []
                step = 0
                query_idx = 0
                done = False
                episode_visual_entropy_queries = []
                logging.info(
                    f"Starting episode {task_episodes + 1} "
                    f"(repeat {repeat_idx + 1}/{args.consistency_repeats_per_init})..."
                )

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(img)
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.primary": np.expand_dims(img, axis=0),
                        "observation.wrist_image": np.expand_dims(wrist_img, axis=0),
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }
                    example_dict = {
                        "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                        "lang": observation["instruction"][0],
                    }

                    start_time = time.time()
                    response = client_model.step(example=example_dict, step=step)
                    end_time = time.time()
                    total_policy_infer_sec += (end_time - start_time)

                    raw_action = response["raw_action"]
                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)
                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                    if args.action_noise_std > 0:
                        delta_action[:6] = delta_action[:6] + np.random.normal(
                            0.0, args.action_noise_std, size=6
                        ).astype(np.float32)

                    full_actions.append(delta_action)
                    obs, reward, done, info = env.step(delta_action.tolist())

                    if response["attn_weights"] is not None:
                        if args.attention_eval_mode == "save" and args.save_attention:
                            save_single_attention_weight(
                                args.save_attn_dir,
                                response["attn_weights"],
                                task_id + 1,
                                episode_idx + 1,
                                query_idx + 1,
                                task_description,
                                img,
                                done,
                                total_episodes + 1,
                            )
                            query_idx += 1
                        elif args.attention_eval_mode == "eval":
                            attn_to_mac_start = time.time()
                            vis_lh = _compute_visual_entropy_lh_from_attentions(response["attn_weights"])
                            if vis_lh is not None:
                                episode_visual_entropy_queries.append(vis_lh)
                                layer_scores_by_ratio = _compute_layer_k_entropy_scores_for_ratios(
                                    vis_lh,
                                    ratios=eval_ratios,
                                    select_bottom=True,
                                    negate_mean=False,
                                )
                                step_metric_scores = {
                                    mae_c_score_key(ratio): float(np.mean(layer_scores_by_ratio[ratio]))
                                    for ratio in eval_ratios
                                }
                                step_attn_to_mac_sec = time.time() - attn_to_mac_start
                                total_attn_to_mac_sec += step_attn_to_mac_sec
                                total_attn_to_mac_steps += 1
                                step_task_id = int(task_id + 1)
                                if step_task_id not in task_attn_timing:
                                    task_attn_timing[step_task_id] = {"total_sec": 0.0, "steps": 0}
                                task_attn_timing[step_task_id]["total_sec"] += step_attn_to_mac_sec
                                task_attn_timing[step_task_id]["steps"] += 1
                                logging.info(
                                    "[ATTN_MAE-C_STEP_TIMING] "
                                    f"task_id={task_id + 1} episode={task_episodes + 1} step={step} "
                                    f"elapsed_sec={step_attn_to_mac_sec:.6f} scores={step_metric_scores}"
                                )
                                query_idx += 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                run_tag = f"episode{episode_idx}_repeat{repeat_idx}"
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{run_tag}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                if full_actions:
                    full_actions = np.stack(full_actions)

                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

                action_path = ""
                if args.save_repeat_actions and full_actions is not None and len(full_actions) > 0:
                    action_path = save_repeat_action_array(
                        action_dir,
                        task_id=task_id + 1,
                        episode_idx=episode_idx + 1,
                        repeat_idx=repeat_idx + 1,
                        actions=full_actions,
                    )
                    scenario_repeat_actions.append(full_actions)

                if args.attention_eval_mode == "eval" and episode_visual_entropy_queries:
                    episode_entropy_lh = np.nanmean(_stack_trim_mats(episode_visual_entropy_queries), axis=0)
                    layer_scores_by_ratio = _compute_layer_k_entropy_scores_for_ratios(
                        episode_entropy_lh,
                        ratios=eval_ratios,
                        select_bottom=True,
                        negate_mean=False,
                    )
                    metric_scores = {
                        mae_c_score_key(ratio): float(np.mean(layer_scores_by_ratio[ratio]))
                        for ratio in eval_ratios
                    }
                    online_eval_records.append(
                        {
                            "task_id": int(task_id + 1),
                            "task_description": task_description,
                            "episode_idx": int(episode_idx + 1),
                            "repeat_idx": int(repeat_idx + 1),
                            "episode_number": int(total_episodes + 1),
                            "is_success": bool(done),
                            "num_queries_used": int(len(episode_visual_entropy_queries)),
                            "attention_eval_method": external_metric_label(MAE_C_METHOD),
                            "scores": metric_scores,
                            "action_path": action_path,
                        }
                    )
                scenario_repeat_meta.append(
                    {
                        "task_id": int(task_id + 1),
                        "task_description": task_description,
                        "episode_idx": int(episode_idx + 1),
                        "repeat_idx": int(repeat_idx + 1),
                        "episode_number": int(total_episodes + 1),
                        "is_success": bool(done),
                        "action_path": action_path,
                    }
                )

            scenario_action_consistency = compute_action_agreement(scenario_repeat_actions)
            for meta in scenario_repeat_meta:
                online_action_consistency_records.append(
                    build_action_consistency_record(
                        task_id=meta["task_id"],
                        task_description=meta["task_description"],
                        episode_idx=meta["episode_idx"],
                        repeat_idx=meta["repeat_idx"],
                        episode_number=meta["episode_number"],
                        is_success=meta["is_success"],
                        action_path=meta["action_path"],
                        scenario_action_consistency=scenario_action_consistency,
                        num_repeats_in_scenario=int(args.consistency_repeats_per_init),
                    )
                )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")
    total_wall_sec = time.time() - eval_wall_start
    logging.info(f"Total wall time (sec): {total_wall_sec:.3f}")
    logging.info(f"Total policy infer time (sec): {total_policy_infer_sec:.3f}")
    if args.attention_eval_mode == "eval":
        logging.info(
            "[ATTN_MAE-C_TIMING_SUMMARY] "
            f"total_steps={total_attn_to_mac_steps} "
            f"total_sec={total_attn_to_mac_sec:.6f} "
            f"avg_sec_per_step={total_attn_to_mac_sec / max(1, total_attn_to_mac_steps):.6f}"
        )

    if args.attention_eval_mode == "eval":
        os.makedirs(args.save_attn_dir, exist_ok=True)
        out_path = os.path.join(args.save_attn_dir, args.attention_eval_output_name)
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in online_eval_records:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Saved online MAE-C eval scores to: {out_path}")
        action_out_path = os.path.join(args.save_attn_dir, args.attention_eval_action_output_name)
        with open(action_out_path, "w", encoding="utf-8") as fout:
            for item in online_action_consistency_records:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Saved online action consistency scores to: {action_out_path}")

    timing_payload = {
        "attention_eval_mode": args.attention_eval_mode,
        "attention_eval_method_effective": (
            external_metric_label(MAE_C_METHOD)
            if args.attention_eval_mode == "eval"
            else external_metric_label(args.attention_eval_method)
        ),
        "task_suite_name": args.task_suite_name,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_wall_sec": float(total_wall_sec),
        "total_policy_infer_sec": float(total_policy_infer_sec),
        "total_attn_to_mac_sec": float(total_attn_to_mac_sec),
        "total_attn_to_mac_steps": int(total_attn_to_mac_steps),
        "avg_attn_to_mac_sec_per_step": float(total_attn_to_mac_sec / max(1, total_attn_to_mac_steps)),
        "avg_wall_sec_per_episode": float(total_wall_sec / max(1, total_episodes)),
        "avg_policy_infer_sec_per_episode": float(total_policy_infer_sec / max(1, total_episodes)),
        "action_noise_std": float(args.action_noise_std),
    }

    if args.attention_eval_mode == "eval":
        date_stamp = os.environ.get("DATE_STAMP", dt.datetime.now().strftime("%Y%m%d"))
        run_stamp = os.environ.get("RUN_STAMP", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        timing_root = Path(args.eval_results_root) / args.task_suite_name / date_stamp / run_stamp
        timing_root.mkdir(parents=True, exist_ok=True)

        per_task_payload = []
        for task_id_key in sorted(task_attn_timing.keys()):
            item = task_attn_timing[task_id_key]
            steps_cnt = int(item["steps"])
            total_sec = float(item["total_sec"])
            task_payload = {
                "date": date_stamp,
                "run_stamp": run_stamp,
                "task_suite_name": args.task_suite_name,
                "task_id": int(task_id_key),
                "total_attn_to_mac_steps": steps_cnt,
                "total_attn_to_mac_sec": total_sec,
                "avg_attn_to_mac_sec_per_step": float(total_sec / max(1, steps_cnt)),
            }
            per_task_payload.append(task_payload)

        single_timing_path = timing_root / f"attn_mae-c_timing_{run_stamp}.json"
        with open(single_timing_path, "w", encoding="utf-8") as sf:
            json.dump(
                {
                    **timing_payload,
                    "date": date_stamp,
                    "run_stamp": run_stamp,
                    "tasks": per_task_payload,
                },
                sf,
                ensure_ascii=False,
                indent=2,
            )
        logging.info(f"Saved attention->MAE-C timing summary to: {single_timing_path}")
    else:
        os.makedirs(args.save_attn_dir, exist_ok=True)
        timing_path = os.path.join(args.save_attn_dir, "eval_timing_summary.json")
        with open(timing_path, "w", encoding="utf-8") as tf:
            json.dump(timing_payload, tf, ensure_ascii=False, indent=2)
        logging.info(f"Saved timing summary to: {timing_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("127.0.0.1", 5678))
    print("🔍 Waiting for VSCode attach on 127.0.0.1:5678 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)