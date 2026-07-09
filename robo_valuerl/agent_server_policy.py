"""
Agent inference server
Responsible for loading the model and handling inference requests
"""
import socket
import pickle
import torch
import numpy as np
from agents.openpi_x_humanoid_agent import OpenPi_X_Humanoid_Agent
# import config.train_config as _config
import openpi.training.config as _config
import struct
import time
import traceback
import transforms.transforms as _transforms
from transforms.tokenizer import PaligemmaTokenizer
import openpi.models.model as _model
import json

ModelType = _model.ModelType

class AgentServer:
    def __init__(self, config):
        """
        Initialize the Agent server

        Args:
            host: server address
            port: server port
            chunk_size: action sequence length
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.host = "0.0.0.0"
        self.port = 8888
        self.chunk_size = config.chunk_size
        
        print(f"Initializing Agent model...")

        # Set up prompt
        prompt_inject_transform = _transforms.InjectDefaultPrompt(prompt="pick and place the apple on the basket")

        # Set up tokenizer; set max_len and discrete_state_input based on model type
        tokenize_transform = _transforms.TokenizePrompt(
            PaligemmaTokenizer(max_len=200 if config.model.model_type == ModelType.PI05 else 48),
            discrete_state_input=config.model.model_type != ModelType.PI0
        )
        print(f"discrete state input set to: {config.model.model_type != ModelType.PI0}")

        # Load normalization stats
        norm_stats = json.load(open(config.data.norm_stats_path))
        denorm_stats = norm_stats.copy()
        
        if config.model.action_dim == 32:
            for key in norm_stats.keys():
                norm_stats[key] = {
                    "mean": np.array(norm_stats[key]["mean"]),
                    "std": np.array(norm_stats[key]["std"]),
                    # "q01": np.array(norm_stats[key]["q01"]),
                    # "q99": np.array(norm_stats[key]["q99"]),
                }
        else:
            for key in norm_stats.keys():
                norm_stats[key] = {
                    "mean": np.array(norm_stats[key]["mean"][:16]),
                    "std": np.array(norm_stats[key]["std"][:16]),
                    # "q01": np.array(norm_stats[key]["q01"][:16]),
                    # "q99": np.array(norm_stats[key]["q99"][:16]),
                }
        
        for key in denorm_stats.keys():
            denorm_stats[key] = {
                "mean": np.array(denorm_stats[key]["mean"][:16]),
                "std": np.array(denorm_stats[key]["std"][:16]),
                # "q01": np.array(denorm_stats[key]["q01"][:16]),
                # "q99": np.array(denorm_stats[key]["q99"][:16]),
            }
        
        # Set up normalize and unnormalize transforms
        norm_state_action = _transforms.NormalizeStatesActions(norm_stats, use_quantile_norm=config.model.model_type != ModelType.PI0)
        self.unnorm_state_action = _transforms.UnnormalizeStatesActions(denorm_stats, use_quantile_norm=config.model.model_type != ModelType.PI0)
        
        # Set up image transforms
        resize_transform = _transforms.ResizeImages(224, 224)
        normalize_images = _transforms.NormalizeImages()
        pad_states_and_actions = _transforms.PadStatesAndActions(config.model.action_dim)
        
        # Compose all transforms (order matters)
        data_transforms = _transforms.compose([
            norm_state_action,
            pad_states_and_actions,
            prompt_inject_transform,
            tokenize_transform,
            resize_transform,
            normalize_images,
        ])
        
        # Set to None, consistent with the test file
        image_transform = None

        print(f"Loading Agent model...")

        self.agent = OpenPi_X_Humanoid_Agent(config, chunk_size = config.chunk_size, data_transforms = data_transforms, image_transforms = image_transform)

        print(f"Agent model loaded! Using device: {self.device}")

        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
    def send_data(self, conn, data):
        """Send data to the client"""
        serialized = pickle.dumps(data)
        # First send the data length
        length = struct.pack('>I', len(serialized))
        conn.sendall(length)
        # Then send the actual data
        conn.sendall(serialized)

    def recv_data(self, conn):
        """Receive data from the client"""
        # First receive the data length
        raw_length = self.recvall(conn, 4)
        if not raw_length:
            return None
        length = struct.unpack('>I', raw_length)[0]
        # Then receive the actual data
        data = self.recvall(conn, length)
        if not data:
            return None
        return pickle.loads(data)

    def recvall(self, conn, n):
        """Receive n bytes of data"""
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def process_request(self, obs):
        """Handle an inference request"""
        try:
            with torch.no_grad():
                # Get normalized actions
                actions = self.agent.predict_action(obs)

                # Unnormalize actions
                result = {
                    "actions": actions,
                }
                actions = self.unnorm_state_action(result)['actions']
                
            return {
                'status': 'success',
                'actions': actions
            }
        except Exception as e:
            print(f"Error during inference: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }

    def run(self):
        """Start the server"""
        self.server_socket.listen(1)
        print(f"Agent server started successfully, listening on {self.host}:{self.port}")
        print("Waiting for client connection...")
        
        while True:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Client connected: {addr}")

                # Handle client requests
                while True:
                    try:
                        # Receive observation data
                        request = self.recv_data(conn)

                        if request is None:
                            print("Client disconnected")
                            break

                        if request.get('type') == 'inference':
                            obs = request['obs']
                            print(f"Received inference request...")

                            # Run inference
                            start_time = time.time()
                            response = self.process_request(obs)
                            inference_time = time.time() - start_time
                            print(f"Inference complete, took: {inference_time:.3f}s")

                            # Send the result
                            self.send_data(conn, response)

                        elif request.get('type') == 'ping':
                            # Heartbeat check
                            self.send_data(conn, {'status': 'pong'})

                        elif request.get('type') == 'shutdown':
                            print("Received shutdown request")
                            self.send_data(conn, {'status': 'ok'})
                            break

                    except Exception as e:
                        print(f"Error while handling request: {e}")
                        traceback.print_exc()
                        try:
                            self.send_data(conn, {
                                'status': 'error',
                                'message': str(e)
                            })
                        except:
                            break

                conn.close()
                print("Client connection closed")

            except KeyboardInterrupt:
                print("\nShutting down server...")
                break
            except Exception as e:
                print(f"Server error: {e}")
                traceback.print_exc()

        self.server_socket.close()
        print("Server shut down")


def main():

    config = _config.cli()
    server = AgentServer(config)

    server.run()
if __name__ == "__main__":
    main()

