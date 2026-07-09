"""
Agent inference server (supports RTC-guided inference)
Responsible for loading the model and handling RTC inference requests containing A_prev, d, s parameters
"""
import socket
import pickle
import torch
import numpy as np
from agents.openpi_x_humanoid_agent_rtc import OpenPi_X_Humanoid_Agent_RTC
import openpi.training.config as _config
import struct
import time
import traceback
import transforms.transforms as _transforms
import albumentations as A
import openpi.models.model as _model
from transforms.tokenizer import PaligemmaTokenizerWithQuality
import json

ModelType = _model.ModelType

class AgentServer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.host = "0.0.0.0"
        self.port = 8888
        self.chunk_size = config.chunk_size
        
        print(f"Initializing RTC-capable Agent model...")

        # 1. Configure data transform pipeline (Transforms)
        prompt_inject_transform = _transforms.InjectDefaultPrompt(prompt="pick and place the apple on the basket")
        
        tokenize_transform = _transforms.TokenizePrompt(
            PaligemmaTokenizerWithQuality(max_len=200, use_quality=True),
            discrete_state_input=False,
            drop_rate=0.
        )
        
        norm_stats = json.load(open(config.data.norm_stats_path))
        denorm_stats = norm_stats.copy()
        
        # Process action space statistics
        def filter_stats(stats, dim):
            for key in stats.keys():
                stats[key] = {k: np.array(v[:dim]) if isinstance(v, list) else v for k, v in stats[key].items()}
            return stats

        action_dim = 16 if config.model.action_dim != 32 else 32
        norm_stats = filter_stats(norm_stats, action_dim)
        denorm_stats = filter_stats(denorm_stats, 16) # Final output is usually 16-dim
        
        
        norm_state_action_maxmin = _transforms.NormalizeStatesActions(
        norm_stats, use_quantile_norm= False
        )

        self.unnorm_state_action = _transforms.UnnormalizeStatesActions(denorm_stats, use_quantile_norm=False)
        
        data_transforms = _transforms.compose([
            _transforms.PadStatesAndActions(config.model.action_dim),
            norm_state_action_maxmin,
            prompt_inject_transform,
            tokenize_transform,
            _transforms.ResizeImages(256, 256),
            _transforms.NormalizeImages(),
        ])
        image_transform = A.Compose([
            A.CenterCrop(height=224, width=224, p=1.0),
        ])
        
        # 2. Load RTC-capable Agent
        self.agent = OpenPi_X_Humanoid_Agent_RTC(
            config, 
            chunk_size=config.chunk_size, 
            data_transforms=data_transforms, 
            image_transforms=image_transform
        )
        
        print(f"Agent model loaded! Using device: {self.device}")
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
    def send_data(self, conn, data):
        serialized = pickle.dumps(data)
        length = struct.pack('>I', len(serialized))
        conn.sendall(length)
        conn.sendall(serialized)
        
    def recv_data(self, conn):
        raw_length = self.recvall(conn, 4)
        if not raw_length: return None
        length = struct.unpack('>I', raw_length)[0]
        data = self.recvall(conn, length)
        return pickle.loads(data) if data else None
    
    def recvall(self, conn, n):
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return bytes(data)
    
    def process_request(self, request):
        """
        Core logic change: handle inference requests, dispatching to the standard interface or the RTC-guided interface
        """
        try:
            obs = request.get('obs')
            action_quality = np.array([2] * 50)
            action_quality = action_quality.reshape(1, -1)
            obs['action_quality'] = action_quality
            a_prev = request.get('a_prev') # Previous action chunk from the client (physical space)
            d = request.get('d', 0)         # Expected inference latency
            s = request.get('s', 0)         # Current execution progress

            # RTC logic branch
            if a_prev is not None:
                # Guided inference: cannot use no_grad, since VJP gradients must be computed internally [cite: 12, 13, 186]
                print(f"[RTC] Running guided inference: d={d}, s={s}")
                actions = self.agent.predict_rtc_action_with_noise(
                    obs=obs,
                    a_prev=a_prev,
                    d=d
                )
                # Denormalize
                result = {"actions": actions}
                actions = self.unnorm_state_action(result)['actions']
            else:
                # Initial frame / standard inference
                print("[RTC] Running standard inference (First Frame)")
                with torch.no_grad():
                    # Normalize actions
                    actions_norm = self.agent.predict_action(obs)
                    # Denormalize
                    result = {"actions": actions_norm}
                    actions = self.unnorm_state_action(result)['actions']
                
            return {
                'status': 'success',
                'actions': actions
            }
        except Exception as e:
            print(f"Error during inference: {e}")
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def run(self):
        self.server_socket.listen(1)
        print(f"Agent server started successfully, listening on {self.host}:{self.port}")
        
        while True:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Client connected: {addr}")
                while True:
                    request = self.recv_data(conn)
                    if request is None: break
                    
                    if request.get('type') == 'inference':
                        start_time = time.time()
                        # Pass the entire request payload to extract RTC parameters
                        response = self.process_request(request)
                        inference_time = time.time() - start_time
                        print(f"Inference complete, took {inference_time:.3f}s")
                        self.send_data(conn, response)
                    
                    elif request.get('type') == 'ping':
                        self.send_data(conn, {'status': 'pong'})
                    
                    elif request.get('type') == 'shutdown':
                        break
                conn.close()
            except KeyboardInterrupt: break
            except Exception as e:
                traceback.print_exc()
        self.server_socket.close()

def main():
    config = _config.cli()
    server = AgentServer(config)
    server.run()

if __name__ == "__main__":
    main()