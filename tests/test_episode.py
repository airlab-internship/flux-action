"""Run a released DROID policy on one frame from each of a few klee9/pi05_franka_v4 episodes.

Compares the predicted plans to what the arm actually did next, and saves the future video the model
predicted with them next to what the cameras recorded. Downloads only the chosen episodes.
Nothing is sent to a robot.

uv run --extra data --extra encoders python tests/test_fr3.py
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MAX_JOINT_DELTA = 0.2  # rad per step at action 1.0, same value as pi05 convert_to_lerobot
BAR = 28  # black rows above and below the 4:3 image in the 224 x 224 frames
CHUNK = 32
EPISODES = [0, 1, 4, 36, 37]  # all four object and plate pairs, in both task wordings


def load_rgb(cell):
    """Crop the bars, then fit into the 360 x 640 frames DROID expects with black sides.

    Cropping to 16:9 instead would cut off the bottom of the view, where the plates are.
    """
    image = Image.open(io.BytesIO(cell["bytes"])).convert("RGB").crop((0, BAR, 224, 224 - BAR))
    image = image.resize((480, 360), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 360))
    canvas.paste(image, (80, 0))
    return np.asarray(canvas, dtype=np.uint8)


def load_episode(repo_id, episode):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = {"repo_id": repo_id, "repo_type": "dataset"}
    name = f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.read_table(hf_hub_download(filename=name, **repo))
    tasks = [json.loads(line) for line in Path(hf_hub_download(filename="meta/tasks.jsonl", **repo)).open()]
    tasks = {row["task_index"]: row["task"] for row in tasks}

    joints = np.asarray(table["joint_position"].to_pylist(), dtype=np.float32)
    gripper = np.asarray(table["gripper_position"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    assert len(joints) > CHUNK, f"episode needs more than {CHUNK} steps, got {len(joints)}"
    # actions are normalized velocities: recover the absolute joint targets the policy predicts
    targets = np.concatenate([joints + actions[:, :7] * MAX_JOINT_DELTA, actions[:, 7:]], axis=1)
    return table, tasks[table["task_index"][0].as_py()], joints, gripper, targets


def pick_frame(targets, joints):
    """The frame with the most joint travel over the next chunk."""
    travel = np.abs(targets[CHUNK - 1 :, :7] - joints[: len(joints) - CHUNK + 1]).sum(axis=1)
    return int(travel.argmax())


def write_observation(path, table, frame, state, right):
    base = load_rgb(table["exterior_image_1_left"][frame].as_py())
    wrist = load_rgb(table["wrist_image_left"][frame].as_py())
    # The released DROID policy expects two exterior cameras and this dataset has one.
    arrays = {
        "images.wrist": wrist,
        "images.left": base,
        "images.right": base if right == "base" else np.zeros_like(base),
        "state": state,
    }
    np.savez(path, **arrays)


def capture_video(policy):
    """The model denoises the future video together with the actions but returns only the actions. Keep both.

    Returns a dict that holds the latent tokens of the last prediction: "condition" for the observed frame,
    "future" for the predicted frames.
    """
    from flux_action.inference import sampling

    captured = {}
    encode_canvas = policy._encode_canvas

    def keep_condition(canvas, state):
        condition = encode_canvas(canvas, state)
        captured["condition"] = condition["x_video_cond"]
        return condition

    def keep_future(solver):
        def run(*args, **kwargs):
            out = solver(*args, **kwargs)
            captured["future"] = out["x_video"]
            return out

        return run

    policy._encode_canvas = keep_condition
    sampling.cosmos_unipc_order2 = keep_future(sampling.cosmos_unipc_order2)
    sampling.euler = keep_future(sampling.euler)
    return captured


def decode_video(policy, captured):
    """Latent tokens -> uint8 frames (T, H, W, 3). Frame 0 is the observed frame passed through the VAE."""
    from einops import rearrange
    from flux_action.processing import packing

    cfg = policy.config
    h, w = cfg.latent_hw
    # Decode the observed latent first, as in training windows, so the predicted latents land on frames 4, 8, ...
    tokens = torch.cat([captured["condition"], captured["future"].to(captured["condition"].dtype)], dim=1)
    latents = rearrange(tokens, "b (t h w) c -> b c t h w", h=h, w=w)
    with torch.inference_mode():
        pixels = policy.video_vae.decode(latents.to(torch.bfloat16))  # (1, 3, 4 * T_lat - 3, 32 h, 32 w)
    content_h, content_w = packing.content_hw(cfg.camera_layout, cfg.canvas_hw)
    pixels = pixels[0, :, :, :content_h, :content_w].float().add(1).mul(127.5).round().clamp(0, 255)
    return pixels.to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def recorded_video(table, start, length, right):
    """What the cameras recorded over the plan, in the DROID layout: wrist on top, the two exteriors below."""
    frames = []
    for frame in range(start, start + length):
        frame = min(frame, table.num_rows - 1)
        wrist = load_rgb(table["wrist_image_left"][frame].as_py())
        base = Image.fromarray(load_rgb(table["exterior_image_1_left"][frame].as_py()))
        base = np.asarray(base.resize((320, 180), Image.Resampling.BILINEAR))
        right_image = base if right == "base" else np.zeros_like(base)
        frames.append(np.concatenate([wrist, np.concatenate([base, right_image], axis=1)], axis=0))
    return np.stack(frames)


def write_video(path, frames, fps):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def save_comparison(output, episode, predicted, recorded, fps):
    """An mp4 with the prediction on the left and the recording on the right, and a still of every 8th frame."""
    write_video(output / f"episode_{episode:06d}_video.mp4", np.concatenate([predicted, recorded], axis=2), fps)
    columns = range(0, len(predicted), 8)
    rows = [np.concatenate([video[i] for i in columns], axis=1) for video in (predicted, recorded)]
    sheet = Image.fromarray(np.concatenate(rows, axis=0))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.Resampling.BILINEAR)
    sheet.save(output / f"episode_{episode:06d}_frames.png")


def patch_fp8_emulation():
    """Emulate torch._scaled_mm in BF16 for GPUs without native FP8. Compare against BF16 before trusting it."""
    import flux_action.models.transformer_inf_fp8r as fp8r

    def emulated_scaled_mm(activation_q, weight_q_nk, activation_scale, weight_scale):
        a = activation_q.to(torch.bfloat16) * activation_scale[:, None].to(torch.bfloat16)
        b = weight_q_nk.T.to(torch.bfloat16) * weight_scale[None, :].to(torch.bfloat16)
        return a @ b

    fp8r._scaled_mm = emulated_scaled_mm
    fp8r.FP8RInferenceDiT._check_runtime_device = staticmethod(lambda device: None)


def report(plan, targets, joints, frame):
    """Compare the plan to the recorded targets, and to simply holding the current pose."""
    future = targets[frame : frame + CHUNK]
    start = joints[frame]
    joint_error = np.abs(plan[:, :7] - future[:, :7])
    hold_error = np.abs(start - future[:, :7])
    np.set_printoptions(precision=3, suppress=True)
    print(f"Plan joint error:      {joint_error.mean():.4f} rad, per joint {joint_error.mean(axis=0)}")
    print(f"Hold-still error:      {hold_error.mean():.4f} rad, per joint {hold_error.mean(axis=0)}")
    print(f"Plan gripper error:    {np.abs(plan[:, 7] - future[:, 7]).mean():.4f}")
    print(f"First command jump:    {np.abs(plan[0, :7] - start).max():.4f} rad (largest joint)")
    print(f"Recorded joint travel: {np.abs(future[-1, :7] - start).max():.4f} rad (largest joint)")
    return float(joint_error.mean()), float(hold_error.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="klee9/pi05_franka_v4")
    parser.add_argument("--episodes", type=int, nargs="+", default=EPISODES)
    parser.add_argument("--checkpoint", default="outputs/droid")
    parser.add_argument("--subfolder", default="variants/fp8r", help="Empty string for the root BF16 package")
    parser.add_argument("--right", choices=("base", "black"), default="base", help="What fills images.right")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/fr3-inference"))
    args = parser.parse_args()

    from flux_action.inference.offline import load_observation
    from flux_action.inference.precision import prepare_for_serving
    from flux_action.policy import FluxActionPolicy

    def print_gpu_memory(label):
        free, total = torch.cuda.mem_get_info()
        print(f"{label}: {free / 1024**3:.1f} GiB free of {total / 1024**3:.1f} GiB")

    print_gpu_memory("GPU before loading")

    args.output.mkdir(parents=True, exist_ok=True)

    if "fp8r" in args.subfolder:
        patch_fp8_emulation()
    # Load on the CPU first. Loading straight onto the GPU also puts the 8.9 GB text encoder there.
    policy = FluxActionPolicy.from_pretrained(args.checkpoint, subfolder=args.subfolder or None, device="cpu")
    print("Quantization:", policy.config.quantization)
    policy.config.inference_seed = args.seed  # predictions take their noise from this, not torch.manual_seed
    prepare_for_serving(policy, device="cuda", compile_dit=False, offload_text_encoder=True)
    # Encode captions on the CPU too, instead of the encoder visiting the GPU next to the DiT and VAE.
    policy._offload_text_encoder = False
    print_gpu_memory("GPU after loading")
    captured = capture_video(policy)

    results = []
    for episode in args.episodes:
        table, task, joints, gripper, targets = load_episode(args.repo, episode)
        frame = pick_frame(targets, joints)
        state = np.append(joints[frame], gripper[frame]).astype(np.float32)
        print(f"\nEpisode {episode}: {len(joints)} steps, frame {frame}, task: {task!r}")

        observation_path = args.output / f"episode_{episode:06d}.npz"
        write_observation(observation_path, table, frame, state, args.right)
        observation = load_observation(observation_path, policy.config, task, "cuda")

        with torch.inference_mode():
            policy.predict_action_chunk(observation)  # a caption seen for the first time is encoded here
        first_peak = torch.cuda.max_memory_allocated() / 1024**3

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            plan = policy.predict_action_chunk(observation)  # caption is cached now
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"First call peak: {first_peak:.1f} GiB, inference peak: {peak:.1f} GiB, latency: {latency:.0f} ms")

        plan = plan[0].float().cpu().numpy()
        assert plan.shape == (CHUNK, 8), f"unexpected plan shape {plan.shape}"
        plan_error, hold_error = report(plan, targets, joints, frame)
        np.save(args.output / f"episode_{episode:06d}_actions.npy", plan)

        predicted = decode_video(policy, captured)
        recorded = recorded_video(table, frame, len(predicted), args.right)
        assert predicted.shape == recorded.shape, f"predicted {predicted.shape}, recorded {recorded.shape}"
        save_comparison(args.output, episode, predicted, recorded, int(policy.config.fps))
        video_error = np.abs(predicted[1:].astype(np.float32) - recorded[1:]).mean()
        print(f"Video: {len(predicted)} frames, mean pixel error vs recording {video_error:.1f} / 255")
        results.append((plan_error, hold_error, peak, latency))

    plan_errors, hold_errors, peaks, latencies = np.array(results).T
    print(f"\n=== {len(results)} episodes ===")
    print(f"Plan joint error:  {plan_errors.mean():.4f} rad")
    print(f"Hold-still error:  {hold_errors.mean():.4f} rad")
    print(f"Plan beat hold-still in {(plan_errors < hold_errors).sum()}/{len(results)} episodes")
    print(f"Largest inference peak: {peaks.max():.1f} GiB, slowest prediction: {latencies.max():.0f} ms")
    print(f"Saved plans and videos (prediction left, recording right) under {args.output}")


if __name__ == "__main__":
    main()
