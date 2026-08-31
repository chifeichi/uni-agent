#!/usr/bin/env bash
# Common launcher for the GSM8K PD/no-PD correctness comparison.
#
# This deliberately reuses the existing Qwen3.5 Megatron launchers so that
# model, training, and rollout settings stay aligned with the performance
# experiments. Only the agent-specific dataset/reward/rollout pieces are
# overridden here.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <no_pd|1p_3d> [hydra overrides...]" >&2
    exit 2
fi

MODE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${MODE}" in
    no_pd)
        BASE_SCRIPT="${SCRIPT_DIR}/run_train_no_pd.sh"
        DEFAULT_EXPERIMENT_SUFFIX="no_pd"
        ;;
    1p_3d)
        BASE_SCRIPT="${SCRIPT_DIR}/run_train_1p_3d.sh"
        DEFAULT_EXPERIMENT_SUFFIX="1p_3d"
        ;;
    *)
        echo "Unknown mode '${MODE}'; expected no_pd or 1p_3d" >&2
        exit 2
        ;;
esac

# The preprocessing command writes train.parquet and test.parquet here by
# default. Override GSM8K_DATA_DIR, TRAIN_DATA, or VAL_DATA when necessary.
GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-/mnt/share/t00986241/latest_release/uni-agent/data/gsm8k}"
export TRAIN_DATA="${TRAIN_DATA:-${GSM8K_DATA_DIR}/train.parquet}"
export VAL_DATA="${VAL_DATA:-${GSM8K_DATA_DIR}/test.parquet}"

# Small, reproducible correctness run. Both entry scripts inherit exactly the
# same values; their only meaningful difference is the rollout PD topology.
export PROMPT_LENGTH="${PROMPT_LENGTH:-2048}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export N="${N:-8}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:--1}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:-1}"
export PROJECT_NAME="${PROJECT_NAME:-qwen35_gsm8k_pd_correctness}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k_${DEFAULT_EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"

SEED="${SEED:-42}"

# vLLM-Ascend's batch-invariant matmul currently accepts only 2-D inputs.
# Qwen3.5 exercises higher-rank linear inputs during profile_run, so enabling
# VERL full determinism would make model initialization fail before rollout.
# Keep fixed data/rollout seeds for comparable runs, but disable this kernel
# substitution explicitly (including stale values inherited by Ray workers).
export VLLM_BATCH_INVARIANT=0
export VERL_FULL_DETERMINISM=0

COMMON_OVERRIDES=(
    algorithm.adv_estimator=grpo
    data.shuffle=False
    data.validation_shuffle=False
    data.seed="${SEED}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.custom_cls.path=null
    data.custom_cls.name=null
    reward.custom_reward_function.path=null
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=null
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.seed="${SEED}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.actor.data_loader_seed="${SEED}"
    trainer.val_before_train=True
    trainer.resume_mode=disable
)

exec bash "${BASE_SCRIPT}" "${COMMON_OVERRIDES[@]}" "$@"
