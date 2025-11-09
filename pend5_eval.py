import argparse
import os
import pickle
from importlib import metadata

import torch

########   movie record switch   ########
#    To record:                         #
#    python3 pend5_eval.py -R           #
#    python3 pend5_eval.py --record     #
#                                       #
#    To not record:                     #
#    python3 pend5_eval.py              #
#########################################

#try:
#    try:
#        if metadata.version("rsl-rl"):
#            raise ImportError
#    except metadata.PackageNotFoundError:
#        if metadata.version("rsl-rl-lib") != "2.2.4":
#            raise ImportError
#except (metadata.PackageNotFoundError, ImportError) as e:
#    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

#from go2_env import Go2Env
from pend5_env import Pend2Env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="param_10") ### put into "output folder name" ###
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-R", "--record", action="store_true", help="Enable camera recording to save video.")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Pend2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Get the camera from the Scene object in genesis (added in Pend5_env)
    cam = env.camera
    
    # Checking the recording flag and preparing to record
    is_recording = args.record and (cam is not None)

    if is_recording:
        if cam is None:
            print("Error: Camera object not found. Ensure 'show_viewer=True' in Pend2Env initialization.")
            if env.scene.visualizer is not None:
                 print("Falling back to viewer camera for rendering.")
                 cam = env.scene.visualizer.camera
            else:
                 print("Exiting evaluation due to missing camera.")
                 is_recording = False # Disable Recording
        
        # Start recording
        if is_recording:
            cam.start_recording()
            print("🎬 Camera recording started. Target MAX_STEPS...")
    elif args.record and cam is None:
         print("Warning: Recording requested, but Camera object not found. Continuing evaluation without recording.")


    # Define the number of execution steps
    # Step count conversion: dt=0.010sec => 100steps/sec,  10sec => max_steps=1000
    MAX_STEPS = 1000
    step_count = 0
    
    # Getting fps
    # env.dt = 0.010, fps = 1 / 0.010 = 100[fps]
    FPS = int(1.0 / env.dt) 

    # Run Loop
    obs, _ = env.reset()
    with torch.no_grad():
        while step_count < MAX_STEPS:
            actions = policy(obs)
            obs, rews, dones, truncated, infos = env.step(actions)
            
            # Perform rendering (capture image to recording stream), only if recording is enabled
            if is_recording:
                cam.render() 
                
            step_count += 1
            if step_count % 100 == 0:
                 print(f"Step: {step_count}/{MAX_STEPS} ({step_count/FPS:.1f}s)")

    # Stop recording and save the file
    if is_recording:
        output_filename = 'evaluation_video.mp4'
        cam.stop_recording(save_to_filename=output_filename, fps=FPS)
        print(f"✅ Camera recording stopped and saved to '{output_filename}' (Total steps: {step_count}, Duration: {step_count/FPS:.1f}s).")
    else:
        print(f"Evaluation finished (Total steps: {step_count}, Duration: {step_count/FPS:.1f}s). No video recording performed.")


if __name__ == "__main__":
    main()
