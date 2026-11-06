import os
os.environ['MUJOCO_GL'] = 'egl'

import csv
import json
import time
import mediapy as media

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from simple_parsing import ArgumentParser, field
from datetime import datetime, timedelta
from typing import Optional

from agents import create_agent
from output_paths import build_run_dirname, format_timestamp
from utils import setup_environment_variables, seed_everything, get_experiment_data, print_summary_table

from builderbench.creative_cube_env import CreativeCubeEnv
from builderbench.creative_cube_language_env import CreativeCubeLanguageWrapper, ENV_SYSTEM_PROMPT

@dataclass
class ClientConfig:
    client_name: str = "vllm"
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_url: str = "http://localhost:8080/v1"
    generate_kwargs: dict = field(default_factory=dict, type=json.loads)

@dataclass
class AgentConfig:
    agent_name: str = "naive"
    agent_kwargs: dict = field(default_factory=dict, type=json.loads)

@dataclass
class Args:
    level_id: str = 'cube-1-task-1'
    seed: int = 0
    num_episodes: int = 1
    early_stop_on_success: bool = True
    output_dir: str = 'outputs'
    record_video: bool = True
    video_fps: int = 30
    video_stride: int = 1
    max_consecutive_invalid_actions: Optional[int] = 5
    per_episode_input_token_limit: Optional[int] = None
    per_episode_output_token_limit: Optional[int] = None
    llm_config: ClientConfig = field(default_factory=ClientConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    
def main(config: Args):
    # set global seed
    global_seed_info = seed_everything(config.seed)

    # set environment variables for API keys
    setup_environment_variables(original_cwd=Path(__file__).resolve().parent)

    # create output directory
    timestamp = format_timestamp(datetime.now())
    run_dirname = build_run_dirname(config.level_id, config.seed, timestamp)
    output_path = os.path.join(config.output_dir, config.agent_config.agent_name, config.llm_config.model_id, run_dirname)
    Path(output_path).mkdir(exist_ok=False, parents=True)

    # capture run metadata and config
    run_metadata, run_config = get_experiment_data(config, timestamp, global_seed_info)
    with open(os.path.join(output_path, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4, default=str)
    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4, default=str)

    # create environment
    env = CreativeCubeLanguageWrapper(CreativeCubeEnv(config.level_id))
    env.reset(seed=config.seed)
    obs, info = None, None

    # create agent
    agent = create_agent(config, env_system_prompt=ENV_SYSTEM_PROMPT)

    episodes_path = os.path.join(output_path, "eval_summary.jsonl")
    with open(episodes_path, "w", encoding="utf-8") as episodes_file:
        
        for episode_idx in range(config.num_episodes):

            csv_filename = os.path.join(output_path, f"episode-{episode_idx:02d}.csv")

            with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(
                    ["Observation", "Action", "Auxiliary response"]
                    + [f"Success_block_{i}" for i in range(env.unwrapped._num_cubes)]
                    + [f"Easy_Success_block_{i}" for i in range(env.unwrapped._num_cubes)]
                    + ["End effector success", "Input tokens", "Output tokens", "Done"]
                )

                pbar_desc = f"Task: {config.level_id}, Episode ID: {episode_idx:02d}"
                pbar = tqdm(
                    total=env.unwrapped._total_timesteps,
                    desc=pbar_desc,
                    leave=True,
                    dynamic_ncols=True,
                )
                
                if config.record_video: frames = []
                agent.reset(terminal_obs=obs, terminal_info=info)
                obs, info = env.reset()
                done = False
                episode_num_actions = 0
                episode_step = 0
                episode_return = 0
                episode_input_tokens = 0
                episode_output_tokens = 0
                consecutive_invalid_actions = 0
                episode_start_time = time.time()
                while not done:
                    client_response = agent.act(obs, info)
                    action, action_valid = env.check_action_validity(client_response.completion)
                    auxiliary = client_response.auxiliary if hasattr(client_response, "auxiliary") else ""
                    
                    csv_writer.writerow(
                                [
                                    obs,
                                    action,
                                    auxiliary,
                                ]
                                + [str(info[f"privileged/block_{i}_success"][0]) for i in range(env.unwrapped._num_cubes)]
                                + [str(info[f"privileged/block_{i}_easy_success"][0]) for i in range(env.unwrapped._num_cubes)]
                                + [str(info['privileged/effector_success'][0])]
                                + [
                                    client_response.input_tokens,
                                    client_response.output_tokens,
                                    str(done),
                                ]
                            )
                    episode_input_tokens += client_response.input_tokens
                    episode_output_tokens += client_response.output_tokens

                    obs, reward, terminated, truncated, info = env.step(
                        action,
                        render=config.record_video,
                        video_stride=config.video_stride,
                    )

                    if config.per_episode_input_token_limit and episode_input_tokens >= config.per_episode_input_token_limit:
                        truncated = True
                    if config.per_episode_output_token_limit and episode_output_tokens >= config.per_episode_output_token_limit:
                        truncated = True

                    if not action_valid:
                        consecutive_invalid_actions += 1
                        obs = f"Note: Your previous output did not contain a valid action. Executed a default action.\n\n" + obs
                    else:
                        consecutive_invalid_actions = 0

                    if (
                        config.max_consecutive_invalid_actions is not None
                        and consecutive_invalid_actions > config.max_consecutive_invalid_actions
                    ):
                        truncated = True

                    done = terminated or truncated
                    episode_num_actions += 1
                    episode_step += info['action_steps']
                    episode_return += reward

                    if config.record_video:
                        current_frames = []
                        for oh in info['observation_history']:
                            if oh['rendered_frame'] is not None:
                                current_frames.append(oh['rendered_frame'])
                        frames.extend(current_frames)

                    pbar.update(info['action_steps'])

                csv_writer.writerow(
                            [
                                obs,
                                'NA',
                                'NA',
                            ]
                            + [str(info[f"privileged/block_{i}_success"][0]) for i in range(env.unwrapped._num_cubes)]
                            + [str(info[f"privileged/block_{i}_easy_success"][0]) for i in range(env.unwrapped._num_cubes)]
                            + [str(info['privileged/effector_success'][0]), "NA", "NA", str(done)]
                        )

                cube_successes = [
                        bool(info[f"privileged/block_{i}_success"][0]) 
                        for i in range(env.unwrapped._num_cubes)
                    ]
                cube_easy_successes = [
                    bool(info[f"privileged/block_{i}_easy_success"][0])
                    for i in range(env.unwrapped._num_cubes)
                ]
                cube_masks = [
                    bool(env.unwrapped._goal_masks[i])
                    for i in range(env.unwrapped._num_cubes)
                ]
                end_effector_success = bool(info['privileged/effector_success'][0])
                episode_success = bool(all(cube_successes)) and end_effector_success
                episode_easy_success = bool(all(cube_easy_successes))
                episode_time = timedelta(seconds=int(time.time() - episode_start_time))
                episode_record = {
                    "episode_idx": episode_idx,
                    "success": episode_success,
                    "easy_success": episode_easy_success,
                    "end_effector_success": end_effector_success,
                    "return": episode_return,
                    "steps": episode_step,
                    "num_actions": episode_num_actions,
                    "tokens_in": episode_input_tokens,
                    "tokens_out": episode_output_tokens,
                    "episode_time": episode_time,
                    "per_cube_success": cube_successes,
                    "per_cube_easy_success": cube_easy_successes,
                    "per_cube_mask": cube_masks,
                }
                episodes_file.write(json.dumps(episode_record, default=str, indent=2) + "\n\n")
                episodes_file.flush()

                if config.record_video:
                    media.write_video(os.path.join(output_path, f"episode-{episode_idx:02d}.mp4"), frames, fps=config.video_fps)

                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)

                pbar.set_postfix({"Status": "Complete"}) 
                pbar.close()

                if episode_success and config.early_stop_on_success:
                    print(f"\nEpisode {episode_idx} was successful. Early stopping the entire run.")
                    break

    time.sleep(1)
    print_summary_table(output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()
    
    main(args.config)