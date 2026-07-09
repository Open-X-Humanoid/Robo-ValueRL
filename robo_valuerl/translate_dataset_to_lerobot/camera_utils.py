import cv2
import numpy as np
from typing import Literal


class CameraUtils:
    
    @staticmethod
    def encode_rgb_image(rgb_image):
        _, encoded_rgb_image = cv2.imencode(".jpg", rgb_image)
        return encoded_rgb_image
    
    @staticmethod
    def encode_depth_image(depth_image):
        """Unit of depth_image is mm."""
        depth_uint16_image = depth_image.clip(0.0, 65535).astype(np.uint16)
        _, encoded_depth_uint16_image = cv2.imencode(".png", depth_uint16_image)
        return encoded_depth_uint16_image

    @staticmethod
    def decode_color_image(
        encoded_color_image: np.ndarray, 
        input_format: Literal["rgb", "bgr"] = "rgb", 
        output_format: Literal["rgb", "bgr"] = "rgb"
    ) -> np.ndarray:
        """Decode the color image.

        Args:
            encoded_color_image: The encoded color image.
            input_format: The input format of the color image.
            output_format: The output format of the color image.

        Returns:
            The decoded color image.
        """
        decoded_color_image = cv2.imdecode(encoded_color_image, cv2.IMREAD_COLOR)
        
        if input_format == "rgb" and output_format == "rgb":
            return decoded_color_image
        elif input_format == "rgb" and output_format == "bgr":
            return cv2.cvtColor(decoded_color_image, cv2.COLOR_RGB2BGR)
        elif input_format == "bgr" and output_format == "rgb":
            return cv2.cvtColor(decoded_color_image, cv2.COLOR_BGR2RGB)
        elif input_format == "bgr" and output_format == "bgr":
            return decoded_color_image
        else:
            raise ValueError("Invalid format.")

    @staticmethod
    def decode_depth_image(encode_depth_uint16_image):
        depth_uint16_image = cv2.imdecode(encode_depth_uint16_image, cv2.IMREAD_UNCHANGED)
        return depth_uint16_image

    @staticmethod
    def apply_depth_colormap(
        depth_image: np.ndarray,
        color_map: int = cv2.COLORMAP_JET,
        normalize: bool = True
    ) -> np.ndarray:
        """Apply a colormap to a depth image.

        Args:
            depth_image: The depth image to apply the colormap to.
            color_map: The colormap to apply.
            normalize: Whether to normalize the depth image.

        Returns:
            The depth image with the colormap applied.
        """
        if normalize:
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_uint8 = depth_normalized.astype(np.uint8)
        else:
            depth_uint8 = depth_image.astype(np.uint8)
        
        depth_colormap = cv2.applyColorMap(depth_uint8, color_map)
        return depth_colormap
