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
import sys
import select
import termios
import tty
from datetime import datetime
# Import your real environment class
from rl_envs.realworld_x_humanoid_env import RealWorld_X_Humanoid_Env
from dataset.online_buffer import RealWorldOnlineBuffer
from config.data_config import X_HUMANOID_Online_DATA_SPEC

# Import teleoperation-related modules
from xrocs.entity.tele_agent.tele_agent_loader import tele_agent_loader
from xrocs.core.app_base import RunnerBase
from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from xtele.station.tienkung.tienkung_twoarm_dm import Exit_SyncXtele_Mode

# ========== Keyboard listener module (uses native terminal mode, no X11 needed) ==========
get_lock = threading.Lock()
stop_event = threading.Event()
listen_keyboard_input = None
intervention_mode = False  # Intervention mode flag
left_gripper_closed = False  # Left gripper state: False=open, True=closed
right_gripper_closed = False  # Right gripper state: False=open, True=closed

def start_listener():
    """
    Non-blocking keyboard key listener (uses native terminal mode, no GUI needed)
    """
    global listen_keyboard_input, intervention_mode, left_gripper_closed, right_gripper_closed

    # Save the old terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set to cbreak mode (read keys without pressing Enter)
        tty.setcbreak(fd)
        while not stop_event.is_set():
            # Check for input, timeout 0.1s
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                
                with get_lock:
                    if key == 'y':
                        listen_keyboard_input = "y"
                    elif key == 'n':
                        listen_keyboard_input = "n"
                    elif key == 's':
                        listen_keyboard_input = "s"
                    elif key == 'q':
                        stop_event.set()
                        break
                    elif key == 'x':  # Press 'x' to toggle intervention mode
                        intervention_mode = not intervention_mode
                        mode_str = "Intervention mode (leader arm controls follower arm)" if intervention_mode else "Sync mode (leader and follower execute action together)"
                        print(f"\n[Mode switch] Current mode: {mode_str}")
                    elif key == 'r':  # Press 'r' to force restore sync mode
                        intervention_mode = False
                        print(f"\n[Mode switch] Restored sync mode (leader and follower execute action together)")
                    elif key == 'c':  # Press 'c' to toggle left gripper
                        left_gripper_closed = not left_gripper_closed
                        state_str = "closed" if left_gripper_closed else "open"
                        print(f"\n[Gripper control] Left gripper: {state_str}")
                    elif key == 'z':  # Press 'z' to toggle right gripper
                        right_gripper_closed = not right_gripper_closed
                        state_str = "closed" if right_gripper_closed else "open"
                        print(f"\n[Gripper control] Right gripper: {state_str}")
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    with get_lock:
        print("Keyboard listener stopped.", flush=True)


class TeleController:
    """
    Teleoperation controller: manages sync/intervention control between the leader and follower arms
    Based on the TeleServer implementation in test_xtele_back.py
    """
    def __init__(self, config_path, xtele_init_pose=None, skip_sync_check=False):
        self.skip_sync_check = skip_sync_check
        # 1. Load configuration
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        
        try:
            self.robot_station = station_loader.generate_station_handle()
            print("[TeleController] Load Station END ==== ")
        except Exception as e:
            print("[TeleController] Error loading station:", e)
            os._exit(1)

        # 2. Agent initialization
        tele_agent_name = self.cfg_dict.get("basic", {}).get("tele_agent_type", "ServoTeleCore")
        self.tele_agent = tele_agent_loader.instantiate_tele_agent(name=tele_agent_name)

        self.freq = 30
        self.step = 'sync_xtele'  # Initially in sync mode
        self.xtele_init_pose = xtele_init_pose
        if xtele_init_pose is None:
            self.tar_left_arm = np.array([0.0] * 7)
            self.tar_right_arm = np.array([0.0] * 7)
        else:
            self.tar_left_arm = xtele_init_pose[:7]
            self.tar_right_arm = xtele_init_pose[7:]

        # 3. Initial alignment
        self.init_x_tele()
        time.sleep(3)
        print("[TeleController] cur data is:", self.tele_agent.act_dict())

        # 4. Runner preparation
        self.runner = RunnerBase(self.tele_agent, self.robot_station)
        print("[TeleController] after------------------ data is:", self.tele_agent.act_dict())
        self.robot_station.connect()
        self.runner.reset_to_home(self.cfg_dict)

        # Check the position difference between leader and follower arms; retry sync multiple times if it's too large
        if self.skip_sync_check:
            print("[TeleController] Warning: leader/follower arm position check skipped (--skip_sync_check enabled)")
        else:
            max_sync_attempts = 3
            for attempt in range(max_sync_attempts):
                try:
                    self.runner.check_pre_sync_status()
                    print(f"[TeleController] Leader/follower arm position check passed")
                    break
                except AssertionError as e:
                    if attempt < max_sync_attempts - 1:
                        print(f"[TeleController] Warning: large leader/follower arm position difference (attempt {attempt + 1}/{max_sync_attempts}), re-syncing...")
                        # Re-fetch the follower arm position and sync the leader arm
                        robot_obs = self.robot_station.get_observation()
                        if robot_obs is not None:
                            # Try to sync the leader arm to the follower arm's position
                            left_pos = robot_obs.get("arm", {}).get("position", {}).get("left", self.tar_left_arm)
                            right_pos = robot_obs.get("arm", {}).get("position", {}).get("right", self.tar_right_arm)
                            self.tar_left_arm = np.array(left_pos) if isinstance(left_pos, list) else left_pos
                            self.tar_right_arm = np.array(right_pos) if isinstance(right_pos, list) else right_pos
                            self.init_x_tele()
                            time.sleep(2)
                    else:
                        print(f"[TeleController] Error: leader/follower arm position difference too large, cannot complete sync")
                        print(f"[TeleController] Please manually move the leader arm close to the follower arm's position and retry")
                        print(f"[TeleController] Or use the --skip_sync_check argument to skip the check (not recommended)")
                        raise e

        self.runner.sync_agent_to_env()

        # 5. Thread and sync control
        self._data_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start the background service thread
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def prepare(self):
        self.tar_left_arm = self.xtele_init_pose[:7]
        self.tar_right_arm = self.xtele_init_pose[7:]
        self.step = 'sync_xtele'
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        self.init_x_tele()

        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()
        
    def init_x_tele(self):
        """Initial trajectory smooth alignment"""
        xtele_data = self.tele_agent.act_dict()
        print("[TeleController] self target left arm is:", self.tar_left_arm)
        print("[TeleController] self target right arm is:", self.tar_right_arm)
        cur_left_arm = np.array(xtele_data["arm"]["position"]["left"])
        cur_right_arm = np.array(xtele_data["arm"]["position"]["right"])

        tele_cur_joints = np.concatenate((cur_left_arm, cur_right_arm))
        tele_tar_joints = np.concatenate((self.tar_left_arm, self.tar_right_arm))

        max_delta = np.max(np.abs(tele_tar_joints - tele_cur_joints))
        normalized = np.clip(max_delta * 32 / np.pi, 0, 1)
        num = int(np.clip(2 ** (normalized * np.log2(81)), 2, 81))
        path = np.linspace(tele_cur_joints, tele_tar_joints, num)

        for p in path:
            self.tele_agent.sync_position_torque(p)
            time.sleep(1.0 / self.freq)

    def close(self):
        """Exit the program and clean up resources"""
        print("[TeleController] >>> Closing all connections and cleaning up resources...")
        self._stop_event.set()

        try:
            if hasattr(self, 'robot_station'):
                self.robot_station.close()
            if hasattr(self, 'tele_agent'):
                self.tele_agent.close()
        except Exception as e:
            print(f"[TeleController] Exception while cleaning up resources: {e}")

        print("[TeleController] Resources cleaned up.")

    def set_intervention_mode(self):
        """Set to intervention mode: leader arm controls follower arm"""
        with self._data_lock:
            self.runner.sync_agent_to_env()
            self.step = 'get_xtele'
            print("[TeleController] Switched to intervention mode (leader arm controls follower arm)")

    def set_sync_mode(self, tar_left_arm, tar_right_arm):
        """Set to sync mode: leader and follower arms sync to the target position"""
        with self._data_lock:
            self.step = 'sync_xtele'
            assert len(tar_left_arm) == 7 and len(tar_right_arm) == 7
            self.tar_left_arm = np.array(tar_left_arm)
            self.tar_right_arm = np.array(tar_right_arm)

    def get_xtele_action(self):
        """Get the leader arm's current position as the action (used in intervention mode)"""
        with self._data_lock:
            xtele_data = self.tele_agent.act_dict()
            left_arm = np.array(xtele_data["arm"]["position"]["left"])
            right_arm = np.array(xtele_data["arm"]["position"]["right"])
            
            # Try several possible gripper data paths
            left_gripper = None
            right_gripper = None

            # Path 1: gripper -> position -> left/right
            if left_gripper is None:
                left_gripper = xtele_data.get("gripper", {}).get("position", {}).get("left", None)
            if right_gripper is None:
                right_gripper = xtele_data.get("gripper", {}).get("position", {}).get("right", None)

            # Path 2: hand -> position -> left/right
            if left_gripper is None:
                left_gripper = xtele_data.get("hand", {}).get("position", {}).get("left", None)
            if right_gripper is None:
                right_gripper = xtele_data.get("hand", {}).get("position", {}).get("right", None)

            # Path 3: gripper -> left/right (direct value)
            if left_gripper is None:
                left_gripper = xtele_data.get("gripper", {}).get("left", None)
            if right_gripper is None:
                right_gripper = xtele_data.get("gripper", {}).get("right", None)

            # Path 4: hand -> left/right (direct value)
            if left_gripper is None:
                left_gripper = xtele_data.get("hand", {}).get("left", None)
            if right_gripper is None:
                right_gripper = xtele_data.get("hand", {}).get("right", None)

            # If still None, use default value 0.0 (gripper open)
            if left_gripper is None:
                left_gripper = 0.0
            if right_gripper is None:
                right_gripper = 0.0

            # Ensure it's a scalar value
            if isinstance(left_gripper, (list, np.ndarray)):
                left_gripper = float(left_gripper[0]) if len(left_gripper) > 0 else 0.0
            if isinstance(right_gripper, (list, np.ndarray)):
                right_gripper = float(right_gripper[0]) if len(right_gripper) > 0 else 0.0

            # Return the full 16-dim action (7 arm + 1 gripper) x2
            action = np.concatenate([
                left_arm, [float(left_gripper)],
                right_arm, [float(right_gripper)]
            ])
            return action
    
    def print_xtele_data_structure(self):
        """Print the structure of xtele_data, for debugging the gripper data path"""
        with self._data_lock:
            xtele_data = self.tele_agent.act_dict()
            print("\n" + "=" * 50)
            print("[DEBUG] xtele_data structure:")
            self._print_dict_structure(xtele_data, indent=0)
            print("=" * 50 + "\n")

    def _print_dict_structure(self, d, indent=0):
        """Recursively print dict structure"""
        prefix = "  " * indent
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}: {{")
                    self._print_dict_structure(value, indent + 1)
                    print(f"{prefix}}}")
                elif isinstance(value, (list, np.ndarray)):
                    arr = np.array(value) if isinstance(value, list) else value
                    print(f"{prefix}{key}: array shape={arr.shape}, value={arr}")
                else:
                    print(f"{prefix}{key}: {type(value).__name__} = {value}")

    def _serve(self):
        """Core control loop"""
        while not self._stop_event.is_set():
            loop_start = time.time()

            try:
                if self.step == 'get_xtele':
                    # Intervention mode: only exit the leader arm's force-feedback sync
                    # Follower arm control is handled uniformly by the main loop via env.step()
                    # This avoids conflicts between TeleController.robot_station and env.robot_station
                    if self.tele_agent is not None:
                        self.tele_agent.exit_any_sync(Exit_SyncXtele_Mode.NotBack)

                elif self.step == 'sync_xtele':
                    # Sync mode: leader arm syncs to the target position
                    xtele_data = self.tele_agent.act_dict()
                    cur_left_arm = np.array(xtele_data["arm"]["position"]["left"])
                    cur_right_arm = np.array(xtele_data["arm"]["position"]["right"])

                    tele_cur_joints = np.concatenate((cur_left_arm, cur_right_arm))
                    tele_tar_joints = np.concatenate((self.tar_left_arm, self.tar_right_arm))

                    # Dynamic path generation
                    max_delta = np.max(np.abs(tele_tar_joints - tele_cur_joints))
                    if max_delta > 0.8:
                        normalized = np.clip(max_delta * 32 / np.pi, 0, 1)
                        num = int(np.clip(2 ** (normalized * np.log2(81)), 2, 81))
                        path = np.linspace(tele_cur_joints, tele_tar_joints, num)
                    else:
                        path = [tele_tar_joints]

                    for p in path:
                        start = time.time()
                        if self._stop_event.is_set():
                            break
                        self.tele_agent.sync_position_torque(p)
                        end = time.time()
                        if len(path) > 1 and end - start < 1.0 / self.freq / len(path):
                            time.sleep(1.0 / self.freq / len(path) - (end - start))

            except Exception as e:
                print(f"[TeleController] Serve Loop Error: {e}")

            loop_end = time.time()
            sleep_time = (1.0 / self.freq) - (loop_end - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


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


class RTCEnvClientWithIntervention:
    """
    RTC (Real-Time Consistency) client - supports intervention mode

    Core parameters:
    - d (delay): fixed estimated inference delay in steps, used for RTC guidance constraints
    - s (execution_steps): number of steps executed after each inference, determines when the next inference is triggered

    New features:
    - Intervention mode: toggled by keypress, leader arm controls follower arm
    - Sync mode: leader and follower arms simultaneously execute the action from large-model inference
    """
    def __init__(
        self,
        remote_agent: Any,
        env: Any,
        tele_controller: TeleController,
        chunk_size: int = 100,
        control_freq: float = 30.0,
        d: int = 5,
        s: int = 5,
    ):
        self.remote_agent = remote_agent
        self.env = env
        self.tele_controller = tele_controller
        self.H = chunk_size  
        self.d = d
        self.s = s
        self.control_freq = control_freq
        
        self.lock = threading.Lock()
        self.running = False
        self.latest_obs = None
        
        # Shared state
        self.t = 0
        self.A_cur = None

        self.inference_thread = None
        self.inference_condition = threading.Condition(self.lock)
        self.is_first_inference = True

        self.A_pending = None
        self.t_switch_at = None

        # Intervention mode related
        self.intervention_active = False
        self.last_intervention_state = False

        print(f"[RTC] Initialization complete: d={self.d}, s={self.s}, H={self.H}, freq={self.control_freq}Hz")

    def start(self):
        """Start the client"""
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
        """Reset state"""
        with self.lock:
            self.is_first_inference = True
            self.A_cur = None
            self.A_pending = None
            self.t = 0
            self.t_switch_at = None
            self.latest_obs = None
            self.intervention_active = False
            self.last_intervention_state = False
        print(f"[RTC] State reset")

    def check_intervention_mode(self):
        """Check and handle intervention mode switching"""
        global intervention_mode
        
        with get_lock:
            current_intervention = intervention_mode
        
        if current_intervention != self.last_intervention_state:
            self.last_intervention_state = current_intervention
            
            if current_intervention:
                # Switch to intervention mode
                self.intervention_active = True
                self.tele_controller.set_intervention_mode()
                global left_gripper_closed, right_gripper_closed
                action = self.A_cur[self.t]
                left_gripper_closed = True if action[7] > 0.3 else False
                right_gripper_closed = True if action[-1] > 0.3 else False
                print("[RTC] Switched to intervention mode: leader arm controls follower arm")
            else:
                # Switch back to sync mode - clear the old action buffer, re-infer from the current state
                self.intervention_active = False

                # **Directly clear the buffer, infer from scratch**
                print("[RTC] Ending intervention, clearing the action buffer, re-inferring from the current state (no history guidance)...")
                print(f"[DEBUG] Before ending intervention: A_cur={self.A_cur is not None}, t={self.t}, A_pending={self.A_pending is not None}")

                # Get the position at the end of intervention (for debugging)
                end_position = self.tele_controller.get_xtele_action()
                print(f"[DEBUG] Position at end of intervention: left={end_position[:7]}, right={end_position[8:15]}")

                self.A_cur = None
                self.A_pending = None
                self.t = 0
                self.t_switch_at = None

                # Set to first-inference state, so the model starts from scratch
                self.is_first_inference = True

                print(f"[DEBUG] After ending intervention: A_cur=None, t={self.t}, is_first_inference={self.is_first_inference}")

                # Notify the inference thread to start new inference
                with self.inference_condition:
                    self.inference_condition.notify_all()

                print("[RTC] Switched to sync mode: waiting for the model to re-infer")

        return self.intervention_active

    def get_action(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get the action for the current time step"""
        # **Key: latest_obs must be updated regardless of mode**
        # This way, after intervention ends, inference is based on the current position rather than the position when intervention started
        with self.lock:
            self.latest_obs = obs

        # Check intervention mode
        is_intervention = self.check_intervention_mode()

        if is_intervention:
            # Intervention mode: return the leader arm's current position as the action, gripper controlled by keyboard
            action = self.tele_controller.get_xtele_action()

            # Override the leader arm's gripper value with the keyboard-controlled gripper value
            global left_gripper_closed, right_gripper_closed
            with get_lock:
                # Gripper value: 0.0 means open, 1.0 means closed
                action[7] = 1.0 if left_gripper_closed else 0.0
                action[15] = 1.0 if right_gripper_closed else 0.0

            return action

        # Sync mode: use the action from large-model inference
        with self.lock:
            # latest_obs was already updated at the start of the function, no need to repeat

            if self.is_first_inference:
                self.inference_condition.notify_all()

            # Check whether we need to switch to the pending new chunk
            if self.A_pending is not None and self.t >= self.t_switch_at:
                self.A_cur = self.A_pending
                self.A_pending = None
                print(f"the t_switch_at is: {self.t_switch_at}")
                print(f"the t is: {self.t}")
                self.t = self.d
                self.t_switch_at = None
                print(f"[RTC] Switched to new chunk! Starting execution from t=0")

            if self.A_cur is None:
                return None

            if self.t >= len(self.A_cur):
                return self.A_cur[-1]

            action = self.A_cur[self.t]
            self.t += 1

            # In sync mode, also control the leader arm
            tar_left_arm = action[:7]
            tar_right_arm = action[8:15]
            
            self.tele_controller.set_sync_mode(tar_left_arm, tar_right_arm)
            if self.t >= self.s:
                self.inference_condition.notify_all()
                
            return action

    def _inference_loop(self):
        """Inference thread"""
        print(f"[RTC inference] Background thread started (fixed mode: d={self.d}, s={self.s})")

        while self.running:
            with self.inference_condition:
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
                t_trigger = self.t
                a_prev_absolute = self.A_cur[t_trigger:] if self.A_cur is not None else None
                current_state = obs.get('state', None)

            try:
                # Delta conversion logic
                a_prev_mixed = None
                if a_prev_absolute is not None and current_state is not None:
                    if not isinstance(current_state, np.ndarray):
                        current_state = np.array(current_state)

                    if current_state.ndim == 1:
                        current_state = current_state[None, :]

                    if a_prev_absolute.ndim == 1:
                        a_prev_absolute = a_prev_absolute[None, :]

                    a_prev_mixed = a_prev_absolute.copy()
                    action_dim = a_prev_absolute.shape[-1]
                    gripper_indices = [7, action_dim - 1]

                    delta_mask = np.ones(action_dim, dtype=bool)
                    for idx in gripper_indices:
                        delta_mask[idx] = False

                    a_prev_mixed[:, delta_mask] = a_prev_absolute[:, delta_mask] - current_state[0, delta_mask][None, :]

                # Call remote inference
                mixed_actions = self.remote_agent.predict_rtc_action(
                    obs,
                    a_prev_mixed,
                    d=self.d,
                    s=self.s
                )

                if mixed_actions.ndim == 1:
                    mixed_actions = mixed_actions[None, :]

                # Convert back to absolute action
                absolute_actions = None
                if current_state is not None:
                    action_dim = mixed_actions.shape[-1]
                    gripper_indices = [7, action_dim - 1]

                    delta_mask = np.ones(action_dim, dtype=bool)
                    for idx in gripper_indices:
                        delta_mask[idx] = False

                    absolute_actions = mixed_actions.copy()
                    absolute_actions[:, delta_mask] = mixed_actions[:, delta_mask] + current_state[0, delta_mask][None, :]
                else:
                    absolute_actions = mixed_actions

                with self.lock:
                    if self.is_first_inference:
                        self.A_cur = absolute_actions
                        self.t = 0
                        self.is_first_inference = False
                        print(f"[RTC] First-frame action chunk ready! Starting execution from t=0")
                        print(f"[DEBUG] First action of the new inference: left={absolute_actions[0][:7]}, right={absolute_actions[0][8:15]}")
                    else:
                        self.A_pending = absolute_actions
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
    parser.add_argument('--tele_config', type=str, default='./x_humanoid_configuration.toml', help='Teleoperation config file path')
    parser.add_argument('--task', type=str, default='separate the blocks and sort by color into different plates')
    parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size H')
    parser.add_argument('--d', type=int, default=3, help='Fixed inference delay estimate (delay)')
    parser.add_argument('--s', type=int, default=8, help='Fixed execution step length (execution steps)')
    parser.add_argument('--freq', type=float, default=30.0, help='Control frequency Hz')
    parser.add_argument('--save_dir', type=str, default='./tmp/rtc_data_with_intervention', help='Trajectory save directory')
    parser.add_argument('--episodes', type=int, default=10, help='Maximum number of episodes to run')
    parser.add_argument('--skip_sync_check', action='store_true', help='Skip leader/follower arm position check (not recommended, may be unsafe)')
    args = parser.parse_args()

    # Add a timestamp to the save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_with_timestamp = os.path.join(args.save_dir, f"{args.task}_{timestamp}")
    save_dir_with_timestamp = save_dir_with_timestamp.replace(" ", "_")

    # 1. Initialize the hardware environment
    env = RealWorld_X_Humanoid_Env(real_world_config_path=args.config, task_name=args.task)

    # First let the follower arm return to its home position
    print("[Main] Homing the follower arm to its initial position...")
    env.prepare()
    print("[Main] Follower arm homing complete")

    # 2. Initialize the teleoperation controller (leader arm control)
    # Get the follower arm's home position from env's config; the leader arm uses the same initial position
    robot_home_pose = env.cfg_dict.get('robot', {}).get('arm', {}).get('home', {}).get('robot', None)
    if robot_home_pose is not None:
        robot_home_pose = np.array(robot_home_pose)
        # robot_home_pose is usually 14-dim (7 left arm + 7 right arm); convert to the format xtele needs
        if len(robot_home_pose) >= 14:
            xtele_init_pose = robot_home_pose[:14]  # 7 left arm + 7 right arm
            print(f"[Main] Got the follower arm's initial position from the config file, using it as the leader arm's initial position:")
            print(f"  Left arm: {xtele_init_pose[:7]}")
            print(f"  Right arm: {xtele_init_pose[7:14]}")
        else:
            xtele_init_pose = None
            print(f"[Main] Warning: follower arm home position dimension insufficient ({len(robot_home_pose)}), using default zero position")
    else:
        xtele_init_pose = None
        print("[Main] Warning: follower arm home position config not found, leader arm uses default zero position")

    tele_controller = TeleController(args.tele_config, xtele_init_pose, skip_sync_check=args.skip_sync_check)

    # Print xtele_data structure for debugging the gripper data path
    print("[Main] Printing leader arm data structure to confirm the gripper data path:")
    tele_controller.print_xtele_data_structure()

    # 3. Initialize the remote Agent
    agent = RemoteAgent(host=args.host, port=args.port)

    # 4. Initialize the online data buffer
    online_buffer = RealWorldOnlineBuffer(
        temp_dir=save_dir_with_timestamp,
        data_spec=X_HUMANOID_Online_DATA_SPEC,
        buffer_capacity=100000,
        fps=args.freq
    )

    # 5. Create the RTC client with intervention support
    client = RTCEnvClientWithIntervention(
        remote_agent=agent,
        env=env,
        tele_controller=tele_controller,
        chunk_size=args.chunk_size,
        control_freq=args.freq,
        d=args.d,
        s=args.s
    )

    control_period = 1.0 / client.control_freq

    print("=" * 60)
    print("Keyboard control instructions:")
    print("  [x] - Toggle intervention mode (leader arm controls follower arm) / sync mode")
    print("  [r] - Force restore sync mode (leader and follower execute action together)")
    print("  [c] - Toggle left gripper open/closed (active in intervention mode)")
    print("  [z] - Toggle right gripper open/closed (active in intervention mode)")
    print("  [y] - Save current trajectory as success")
    print("  [n] - Save current trajectory as failure")
    print("  [s] - Skip current trajectory (do not save)")
    print("  [q] - Quit the program")
    print("=" * 60)
    print(f"Control frequency: {args.freq} Hz")
    print(f"Chunk size: {args.chunk_size}")
    print(f"RTC parameters: d={args.d}, s={args.s}")
    print("=" * 60)

    # Start the RTC client
    client.start()

    # After all initialization is complete, start the keyboard listener thread
    print("\n[Main] Starting keyboard listener...")
    keyboard_thread = threading.Thread(target=start_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    print("[Main] Keyboard listener started, you can now use keyboard control\n")

    try:
        for episode in range(args.episodes):
            # Reset keyboard input state
            global listen_keyboard_input, intervention_mode, left_gripper_closed, right_gripper_closed
            with get_lock:
                listen_keyboard_input = None
                intervention_mode = False
                left_gripper_closed = False  # Reset left gripper to open state
                right_gripper_closed = False  # Reset right gripper to open state

            print(f"\n[Episode {episode + 1}/{args.episodes}] Preparing environment...")
            env.prepare()
            tele_controller.prepare()
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
            print("[*] Press [x] to toggle intervention mode, [r] to force restore sync mode")
            print("[*] In intervention mode: [c] toggle left gripper, [z] toggle right gripper")
            
            prev_obs = env.get_obs()
            step_count = 0
            intervention_step_count = 0

            while True:
                if stop_event.is_set():
                    break

                # If just recovered from intervention mode, wait for the first inference to complete
                if client.is_first_inference and not client.intervention_active:
                    print("[*] Intervention ended, waiting for the model to re-infer...")
                    print(f"[DEBUG] Observation when waiting for inference to start: {env.get_obs().get('state', 'N/A')[:16]}")
                    while client.is_first_inference:
                        obs = env.get_obs()
                        action = client.get_action(obs)
                        print(f"[DEBUG] action during wait ={action}")
                        time.sleep(0.05)
                    print("[+] Inference complete, resuming automatic control")
                    print(f"[DEBUG] Observation when inference completed: {env.get_obs().get('state', 'N/A')[:16]}")

                t_cycle_start = time.perf_counter()

                obs = env.get_obs()
                action = client.get_action(obs)

                # Check whether currently in intervention mode
                is_intervention = client.intervention_active

                if action is not None:
                    # Special gripper handling
                    save_action = action.copy()
                    action[7] = 0.9 if action[7] > 0.4 else action[7]
                    action[15] = 0.9 if action[15] > 0.4 else action[15]
                    action[7] = np.clip(action[7], 0, 1)
                    action[15] = np.clip(action[15], 0, 1)

                    # Uniformly control the follower arm via env.step() (whether in intervention or automatic mode)
                    # In intervention mode, action comes from the leader arm (tele_agent)
                    # In automatic mode, action comes from large-model inference
                    next_obs = env.step(action)
                    if next_obs is False:
                        print(f"[Episode {episode + 1}] [✗] Action execution failed")
                        online_buffer.save_trajectory(success=False)
                        print(f"[Episode {episode + 1}] [✗] Saved failed trajectory (total {step_count} steps)")
                        raise ValueError("Action execution failed")

                    # Save data for every step (including data during intervention), and mark whether it was human intervention
                    online_buffer._save_step(prev_obs, save_action, next_obs, is_intervention=is_intervention)
                    prev_obs = next_obs
                    step_count += 1

                    if is_intervention:
                        intervention_step_count += 1

                    if step_count % 50 == 0:
                        mode_str = "intervention" if is_intervention else "automatic"
                        print(f"[Episode {episode + 1}] Executed {step_count} steps (intervention: {intervention_step_count} steps) [{mode_str} mode]")

                # Check keyboard input
                with get_lock:
                    if listen_keyboard_input == "y":
                        online_buffer.save_trajectory(success=True)
                        print(f"[Episode {episode + 1}] [✓] Saved successful trajectory (total {step_count} steps, intervention {intervention_step_count} steps)")
                        break
                    elif listen_keyboard_input == "n":
                        online_buffer.save_trajectory(success=False)
                        print(f"[Episode {episode + 1}] [✗] Saved failed trajectory (total {step_count} steps, intervention {intervention_step_count} steps)")
                        break
                    elif listen_keyboard_input == "s":
                        online_buffer.clear_single_traj_buffer()
                        print(f"[Episode {episode + 1}] [~] Skipped trajectory (total {step_count} steps, not saved)")
                        break

                # Frequency compensation
                t_used = time.perf_counter() - t_cycle_start
                if t_used < control_period:
                    time.sleep(control_period - t_used)

            if stop_event.is_set():
                print("[!] Exiting program")
                break

            # Ask whether to continue
            with get_lock:
                listen_keyboard_input = None
                intervention_mode = False
                left_gripper_closed = False
                right_gripper_closed = False
            time.sleep(0.1)

            print(f"[Episode {episode + 1}/{args.episodes}] Completed {episode + 1} tasks so far")
            continue_flag = input("Continue to the next task? (Y/n) ")

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
        tele_controller.close()
        agent.close()
        print("[Cleanup complete]")


if __name__ == "__main__":
    main()

