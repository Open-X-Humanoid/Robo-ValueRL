import socket
import pickle
import time
import struct
import numpy as np
import threading
import argparse
import traceback
import os
from typing import Any, Dict, Optional
from datetime import datetime
from pynput import keyboard

# Import your real environment class
from rl_envs.realworld_x_humanoid_env import RealWorld_X_Humanoid_Env
from dataset.online_buffer import RealWorldOnlineBuffer
from config.data_config import X_HUMANOID_Online_DATA_SPEC

# ========== Keyboard listener module ==========
get_lock = threading.Lock()
stop_event = threading.Event()
listen_keyboard_input = None

def on_press(key):
    """Callback for keypress events"""
    global listen_keyboard_input
    try:
        if key.char == 'y':
            with get_lock:
                listen_keyboard_input = "y"
        elif key.char == 'n':
            with get_lock:
                listen_keyboard_input = "n"
        elif key.char == 's':
            with get_lock:
                listen_keyboard_input = "s"
        elif key.char == 'q':
            with get_lock:
                stop_event.set()
                return False
    except AttributeError:
        pass

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    with get_lock:
        print("Keyboard listener stopped.", flush=True)

class RemoteAgent:
    """Remote Agent client: handles communication with the GPU server for large-model inference"""
    def __init__(self, host='localhost', port=8888, max_retries=3):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.socket = None
        self.connect()

    def connect(self):
        print(f"[*] Connecting to inference server {self.host}:{self.port}...")
        for attempt in range(self.max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print("[+] Server connected successfully!")
                return
            except Exception as e:
                print(f"[-] Connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if self.socket: self.socket.close()
                time.sleep(1)
        raise ConnectionError(f"Unable to connect to inference server {self.host}:{self.port}")

    def send_data(self, data):
        serialized = pickle.dumps(data)
        length = struct.pack('>I', len(serialized))
        self.socket.sendall(length)
        self.socket.sendall(serialized)
    
    def recv_data(self):
        raw_length = self.recvall(4)
        if not raw_length: return None
        length = struct.unpack('>I', raw_length)[0]
        data = self.recvall(length)
        return pickle.loads(data) if data else None
    
    def recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return bytes(data)
    
    def predict_action(self, obs):
        """Send an observation, get the action sequence [chunk_size, action_dim]"""
        try:
            self.send_data({'type': 'inference', 'obs': obs})
            response = self.recv_data()
            if response and response['status'] == 'success':
                return np.array(response['actions'])
            raise RuntimeError(f"Inference failed: {response.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"[-] Remote inference error: {e}")
            raise

    def predict_rtc_action(self, obs, a_prev, d, s):
        """
        Send an RTC inference request
        :param obs: current observation
        :param a_prev: the remaining part of the previous action chunk (used for inpainting guidance)
        :param d: predicted inference delay in steps
        :param s: execution step progress
        """
        try:
            # Construct the request containing the RTC parameters
            payload = {
                'type': 'inference',
                'obs': obs,
                'a_prev': a_prev if a_prev is not None else None,
                'd': d,
                's': s
            }
            self.send_data(payload)
            response = self.recv_data()
            if response and response['status'] == 'success':
                return np.array(response['actions'])
            raise RuntimeError(f"RTC inference failed: {response.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"[-] Remote RTC inference error: {e}")
            raise

    def close(self):
        try:
            if self.socket:
                self.send_data({'type': 'shutdown'})
                self.socket.close()
        except: pass



import cv2
def display_observation_images(obs, step_count, episode_num):
    """Display the real-time image observations from all cameras
    
    Images are preprocessed: resize to 256x256, then center crop to 224x224,
    then scaled up 2x for better visibility in the display.
    """
    try:
        # Extract images from observation (remove batch dimension)
        base_img = obs['image']['base_0_rgb'][0]  # [H, W, 3]
        left_img = obs['image']['left_wrist_0_rgb'][0]  # [H, W, 3]
        right_img = obs['image']['right_wrist_0_rgb'][0]  # [H, W, 3]
        
        # Convert RGB to BGR for cv2 display (images are in RGB format from post_process_obs)
        base_img_bgr = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
        right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
        
        def preprocess_image(img):
            """Resize to 256x256, then center crop to 224x224"""
            # Step 1: Resize to 256x256
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Step 2: Center crop to 224x224
            h, w = img_resized.shape[:2]
            start_y = (h - 224) // 2
            start_x = (w - 224) // 2
            img_cropped = img_resized[start_y:start_y+224, start_x:start_x+224]
            
            return img_cropped
        
        # Apply preprocessing: resize to 256x256, then center crop to 224x224
        base_img_bgr = preprocess_image(base_img_bgr)
        left_img_bgr = preprocess_image(left_img_bgr)
        right_img_bgr = preprocess_image(right_img_bgr)
        
        # Scale up for better visibility (224x224 -> 448x448 for 2x zoom)
        display_scale = 2
        new_h, new_w = 224 * display_scale, 224 * display_scale
        base_img_bgr = cv2.resize(base_img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        left_img_bgr = cv2.resize(left_img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        right_img_bgr = cv2.resize(right_img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Add text labels to images
        font_scale = 0.6
        thickness = 2
        cv2.putText(base_img_bgr, f"Ep {episode_num} Step {step_count} - Base", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(left_img_bgr, f"Ep {episode_num} Step {step_count} - Left Wrist", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(right_img_bgr, f"Ep {episode_num} Step {step_count} - Right Wrist", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # Concatenate images horizontally for display
        combined_img = np.hstack([base_img_bgr, left_img_bgr, right_img_bgr])
        
        # Display the combined image
        cv2.imshow('Real-time Observation (Base | Left Wrist | Right Wrist)', combined_img)
        cv2.waitKey(1)  # Non-blocking wait (1ms) to update display
        
    except Exception as e:
        print(f"[!] Error displaying image: {e}", flush=True)

import numpy as np
import threading
import time
class RTCEnvClient:
    """
    RTC (Real-Time Consistency) client

    Core parameters:
    - d (delay): fixed estimated inference delay in steps, used for RTC guidance constraints
    - s (execution_steps): number of steps executed after each inference, determines when the next inference is triggered
    """
    def __init__(
        self,
        remote_agent: Any,
        env: Any,
        chunk_size: int = 100,
        control_freq: float = 30.0,
        d: int = 5,   # Fixed inference delay estimate (delay)
        s: int = 5,   # Fixed execution step length (execution steps)
    ):
        self.remote_agent = remote_agent
        self.env = env
        self.H = chunk_size  
        self.d = d  # Fixed delay
        self.s = s  # Fixed execution step length
        self.control_freq = control_freq
        
        self.lock = threading.Lock()
        self.running = False
        self.latest_obs = None
        
        # Shared state
        self.t = 0              # Number of steps executed since the current chunk started
        self.A_cur = None       # Action chunk currently being executed

        self.inference_thread = None
        self.inference_condition = threading.Condition(self.lock)
        self.is_first_inference = True

        # New action chunk pending switch (inference done but switch time not yet reached)
        self.A_pending = None
        self.t_switch_at = None  # Switch when t reaches this value

        print(f"[RTC] Initialization complete: d={self.d}, s={self.s}, H={self.H}, freq={self.control_freq}Hz")

    def start(self):
        """Start the client (should only be called once)"""
        if self.inference_thread is not None and self.inference_thread.is_alive():
            print("[RTC] Client already running, skipping start")
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("[RTC] Inference thread started")

    def stop(self):
        """Stop the client"""
        self.running = False
        with self.inference_condition:
            self.inference_condition.notify_all()
        if self.inference_thread:
            self.inference_thread.join()

    def reset(self):
        """Reset state (called at the start of each episode)"""
        with self.lock:
            self.is_first_inference = True
            self.A_cur = None
            self.A_pending = None
            self.t = 0
            self.t_switch_at = None
            self.latest_obs = None
        print(f"[RTC] State reset")

    def get_action(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """[Control thread] Get a single action for the current time step"""
        with self.lock:
            self.latest_obs = obs

            # Proactively wake up the inference thread on the first inference
            if self.is_first_inference:
                self.inference_condition.notify_all()

            # ========== Check whether we need to switch to the pending new chunk ==========
            if self.A_pending is not None and self.t >= self.t_switch_at:
                # Time to switch has arrived, perform the switch
                self.A_cur = self.A_pending
                self.A_pending = None

                # After switching, t continues from position d (skipping the frozen region)
                print(f"the t_switch_at is: {self.t_switch_at}")
                print(f"the t is: {self.t}")
                self.t = self.d
                self.t_switch_at = None

                print(f"[RTC] Switched to new chunk! Starting execution from t=0")

            if self.A_cur is None:
                return None

            # Bounds check: prevent index out of range
            if self.t >= len(self.A_cur):
                # Actions exhausted, return the last action (hold pose)
                return self.A_cur[-1]

            # Consume one action from the current chunk and increment t
            action = self.A_cur[self.t]
            self.t += 1

            # Wake up the background inference thread once the execution step count reaches s
            if self.t >= self.s:
                self.inference_condition.notify_all()

            return action

    def _inference_loop(self):
        """[Inference thread] Asynchronously generates the repaired new chunk in the background"""
        print(f"[RTC inference] Background thread started (fixed mode: d={self.d}, s={self.s})")

        while self.running:
            with self.inference_condition:
                # Wait for the trigger condition:
                # 1. First inference: any observation is enough
                # 2. Subsequent inference: execution steps >= s and no pending chunk
                while self.running:
                    if self.is_first_inference and self.latest_obs is not None:
                        break
                    if self.A_cur is not None and self.t >= self.s and self.A_pending is None:
                        break
                    self.inference_condition.wait(timeout=0.01)

            if not self.running:
                break

            with self.lock:
                obs = self.latest_obs
                if obs is None:
                    continue

                self.t = self.t - 1
                # Record t at the moment inference was triggered (used to compute the switch time)
                t_trigger = self.t

                # Prepare A_prev: take the part of the current chunk starting from t_trigger
                # This part of the action is currently/about to be executed, and the new chunk needs to stay consistent with it
                a_prev_absolute = self.A_cur[t_trigger:] if self.A_cur is not None else None

                # Get the current state (used for delta conversion)
                current_state = obs.get('state', None)

            try:
                # ========== Delta conversion logic (mixed mode) ==========
                # Convert absolute actions to delta (relative to the current state)
                # Note: dimension 7 and dimension -1 (gripper) use absolute control, other dimensions use delta
                a_prev_mixed = None
                if a_prev_absolute is not None and current_state is not None:
                    # Make sure current_state is a numpy array with matching dimensions
                    if not isinstance(current_state, np.ndarray):
                        current_state = np.array(current_state)

                    # Make sure current_state is at least 2D
                    if current_state.ndim == 1:
                        current_state = current_state[None, :]
                        print(f"[RTC] current_state is 1D, reshaped to 2D: {current_state.shape}")

                    # Make sure a_prev_absolute is a 2D array
                    if a_prev_absolute.ndim == 1:
                        a_prev_absolute = a_prev_absolute[None, :]
                        print(f"[RTC] Warning: a_prev_absolute is 1D, reshaped to 2D: {a_prev_absolute.shape}")

                    # Make a copy for modification
                    a_prev_mixed = a_prev_absolute.copy()

                    # Get the action dimension
                    action_dim = a_prev_absolute.shape[-1]

                    # Define the gripper dimension indices (absolute control)
                    gripper_indices = [7, action_dim - 1]

                    # Apply delta conversion to non-gripper dimensions
                    # Create a mask where True means delta conversion is needed
                    delta_mask = np.ones(action_dim, dtype=bool)
                    for idx in gripper_indices:
                        delta_mask[idx] = False

                    # Perform the delta conversion: a_delta = a_absolute - state
                    # Only subtract for dimensions where delta_mask=True
                    # a_prev_mixed[:, delta_mask] = a_prev_absolute[:, delta_mask] - current_state[delta_mask]
                    # print("the current_state is:", current_state.shape)
                    a_prev_mixed[:, delta_mask] = a_prev_absolute[:, delta_mask] - current_state[0, delta_mask][None, :]

                    # a_prev_mixed[:, ] = a_prev_absolute[:, ] - current_state
                    # for idx in gripper_indices:
                    #     a_prev_mixed[:, idx] = a_prev_absolute[:, idx]


                    print(f"[RTC] Mixed delta conversion: total dim={action_dim}, "
                          f"gripper dims (absolute)={gripper_indices}, "
                          f"delta dims={np.where(delta_mask)[0].tolist()}")

                # Call the remote Agent's RTC interface (passing in the mixed-form a_prev)
                mixed_actions = self.remote_agent.predict_rtc_action(
                    obs,
                    a_prev_mixed,  # Send the mixed form (gripper absolute, others delta)
                    d=self.d,
                    s=self.s
                )

                # Make sure mixed_actions is a 2D array
                if mixed_actions.ndim == 1:
                    mixed_actions = mixed_actions[None, :]  # (action_dim,) -> (1, action_dim)
                    print(f"[RTC] Warning: mixed_actions is 1D, reshaped to 2D: {mixed_actions.shape}")

                # ========== Convert back to absolute action ==========
                absolute_actions = None
                if current_state is not None:
                    # Add the state back to the delta dimensions, keep the gripper dimensions unchanged
                    action_dim = mixed_actions.shape[-1]
                    gripper_indices = [7, action_dim - 1]

                    # Create the mask
                    delta_mask = np.ones(action_dim, dtype=bool)
                    for idx in gripper_indices:
                        delta_mask[idx] = False

                    # Make a copy
                    absolute_actions = mixed_actions.copy()

                    # absolute_actions[:, ] = mixed_actions[:, ] + current_state
                    # for idx in gripper_indices:
                    #     absolute_actions[:, idx] = mixed_actions[:, idx]

                    absolute_actions[:, delta_mask] = mixed_actions[:, delta_mask] + current_state[0,delta_mask][None, :]

                    print(f"[RTC] Mixed->Absolute conversion: gripper dims kept absolute, "
                          f"delta dims added back to state")
                else:
                    # If there's no state, use it directly (compatible with the first frame)
                    absolute_actions = mixed_actions

                with self.lock:
                    if self.is_first_inference:
                        # First inference: use directly, start from t=0
                        self.A_cur = absolute_actions
                        self.t = 0
                        self.is_first_inference = False
                        print(f"[RTC] First-frame action chunk ready! Starting execution from t=0")
                    else:
                        # Subsequent inference: set as pending, wait for the switch time
                        self.A_pending = absolute_actions

                        # Switch time = t at trigger + d (after the frozen region finishes)
                        self.t_switch_at = t_trigger + self.d

                        t_now = self.t
                        wait_steps = max(0, self.t_switch_at - t_now)

                        print(f"[RTC] New chunk ready! t_trigger={t_trigger}, t_now={t_now}, "
                              f"t_switch_at={self.t_switch_at}, need to wait {wait_steps} steps")

            except Exception as e:
                print(f"[RTC] Inference failed: {e}")
                traceback.print_exc()
                time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--config', type=str, default='./x_humanoid_configuration.toml')
    parser.add_argument('--task', type=str, default='separate the blocks and sort by color into different plates')
    parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size H')
    parser.add_argument('--d', type=int, default=3, help='Fixed inference delay estimate (delay)')
    parser.add_argument('--s', type=int, default=8, help='Fixed execution step length (execution steps)')
    parser.add_argument('--freq', type=float, default=30.0, help='Control frequency Hz')
    parser.add_argument('--save_dir', type=str, default='./tmp/best_rtc_data_rollout', help='Trajectory save directory')
    parser.add_argument('--episodes', type=int, default=20, help='Maximum number of episodes to run')
    args = parser.parse_args()

    # Add a timestamp to the save directory to avoid overwriting existing data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_with_timestamp = os.path.join(args.save_dir, f"{args.task}_{timestamp}")
    save_dir_with_timestamp = save_dir_with_timestamp.replace(" ", "_")

    # Start the keyboard listener thread
    keyboard_thread = threading.Thread(target=start_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    # 1. Initialize the hardware environment
    env = RealWorld_X_Humanoid_Env(real_world_config_path=args.config, task_name=args.task)

    # 2. Initialize the remote Agent
    agent = RemoteAgent(host=args.host, port=args.port)

    # 3. Initialize the online data buffer
    online_buffer = RealWorldOnlineBuffer(
        temp_dir=save_dir_with_timestamp,
        data_spec=X_HUMANOID_Online_DATA_SPEC,
        buffer_capacity=100000,
        fps=args.freq
    )

    # 4. Create the RTC client (using fixed d and s)
    client = RTCEnvClient(
        remote_agent=agent,
        env=env,
        chunk_size=args.chunk_size,
        control_freq=args.freq,
        d=args.d,  # Fixed delay
        s=args.s   # Fixed execution step length
    )

    control_period = 1.0 / client.control_freq

    print("=" * 50)
    print("Keyboard control instructions:")
    print("  [y] - Save current trajectory as success")
    print("  [n] - Save current trajectory as failure")
    print("  [s] - Skip current trajectory (do not save)")
    print("  [q] - Quit the program")
    print("=" * 50)
    print(f"Control frequency: {args.freq} Hz")
    print(f"Chunk size: {args.chunk_size}")
    print(f"RTC parameters: d={args.d}, s={args.s}")
    print("=" * 50)

    # Start the RTC client (only starts once)
    client.start()

    try:
        for episode in range(args.episodes):
            # Reset keyboard input state
            global listen_keyboard_input
            with get_lock:
                listen_keyboard_input = None

            print(f"\n[Episode {episode + 1}/{args.episodes}] Preparing environment...")
            env.prepare()
            time.sleep(10)

            print(f"[+] Starting RTC control loop (Episode {episode + 1}/{args.episodes})")

            # Reset RTC client state
            client.reset()

            print("[*] Waiting for first-frame inference...")
            while client.is_first_inference:
                first_obs = env.get_obs()
                client.get_action(first_obs)
                time.sleep(0.1)

            print("[+] First-frame inference complete, starting real-time control")
            prev_obs = env.get_obs()
            step_count = 0

            while True:
                if stop_event.is_set():
                    break

                t_cycle_start = time.perf_counter()

                obs = env.get_obs()
                action = client.get_action(obs)

                if action is not None:
                    # Special gripper handling
                    save_action = action.copy()
                    action[7] = 1 if action[7] > 0.5 else action[7]
                    action[15] = 1 if action[15] > 0.5 else action[15]
                    action[7] = np.clip(action[7], 0, 1)
                    action[15] = np.clip(action[15], 0, 1)

                    display_observation_images(prev_obs, step_count, episode + 1)
                    print("the action is:", action)
                    # import pdb; pdb.set_trace()
                    next_obs = env.step(action)
                    if next_obs is False:
                        print(f"[Episode {episode + 1}] [✗] Action execution failed")
                        online_buffer.save_trajectory(success=False)
                        print(f"[Episode {episode + 1}] [✗] Saved failed trajectory (total {step_count} steps)")
                        raise ValueError("Action execution failed")

                    # Save data for every step
                    online_buffer._save_step(prev_obs, save_action, next_obs)
                    prev_obs = next_obs
                    step_count += 1

                    if step_count % 50 == 0:
                        print(f"[Episode {episode + 1}] Executed {step_count} steps")

                # Check keyboard input
                with get_lock:
                    if listen_keyboard_input == "y":
                        online_buffer.save_trajectory(success=True)
                        print(f"[Episode {episode + 1}] [✓] Saved successful trajectory (total {step_count} steps)")
                        break
                    elif listen_keyboard_input == "n":
                        online_buffer.save_trajectory(success=False)
                        print(f"[Episode {episode + 1}] [✗] Saved failed trajectory (total {step_count} steps)")
                        break
                    elif listen_keyboard_input == "s":
                        online_buffer.clear_single_traj_buffer()
                        print(f"[Episode {episode + 1}] [~] Skipped trajectory (total {step_count} steps, not saved)")
                        break

                # Frequency compensation
                t_used = time.perf_counter() - t_cycle_start
                if t_used < control_period:
                    time.sleep(control_period - t_used)

            # Stop the RTC client


            if stop_event.is_set():
                print("[!] Exiting program")
                break

            # Ask whether to continue
            with get_lock:
                listen_keyboard_input = None
            time.sleep(0.1)

            print(f"[Episode {episode + 1}/{args.episodes}] Completed {episode + 1} tasks so far")
            continue_flag = input("Continue to the next task? (Y/n) ")

            # Handle possible two-letter input
            continue_flag_clean = continue_flag.lower().strip()
            if continue_flag_clean == "n" or continue_flag_clean == "nn" or continue_flag_clean == "yn" or continue_flag_clean == "sn":
                print(f"[!] User chose to exit, completed {episode + 1} tasks")
                break

        print(f"\n[Done] Saved {online_buffer.traj_num} trajectories in total")

    except KeyboardInterrupt:
        print("\n[*] Manually stopped by user")
    except Exception as e:
        print(f"[!] Error: {e}")
        traceback.print_exc()
    finally:
        if not stop_event.is_set():
            stop_event.set()
        client.stop()
        agent.close()
        print("[Cleanup complete]")

if __name__ == "__main__":
    main()
