import json
import numpy as np
import tensorflow as tf

class KeypointProcessor:
    def __init__(self, max_seq_length=370):
        self.max_seq_length = max_seq_length

    def skip_confidence_values(self, keypoints):
        """Skips every third value in the keypoints list (removing confidence values)."""
        return [value for index, value in enumerate(keypoints) if (index % 3) != 2]

    def extend_keypoints(self, keypoints):
        """Extends keypoints by repeating frames to match the max sequence length."""
        num_frames = len(keypoints) // 272 
        if num_frames < self.max_seq_length:
            # Repeat frames to extend the sequence length
            repeat_factor = (self.max_seq_length + num_frames - 1) // num_frames  # Ceiling division to ensure length
            extended_keypoints = (keypoints * repeat_factor)[:self.max_seq_length * 272]  # Trim excess frames
        else:
            # Truncate if the sequence is too long
            extended_keypoints = keypoints[:self.max_seq_length * 272]
        return extended_keypoints

    def standardize_keypoints(self, keypoints):
        """Standardizes keypoints (e.g., mean centering or scaling)."""
        keypoints_array = np.array(keypoints).reshape(-1, 272)
        mean = np.mean(keypoints_array, axis=0)
        std = np.std(keypoints_array, axis=0)
        standardized_keypoints = (keypoints_array - mean) / std
        return standardized_keypoints.flatten().tolist()

    def build_tensors(self, keypoints):
        """Converts processed keypoints into a tensor suitable for model input."""
        keypoints_array = np.array(keypoints).reshape(1, self.max_seq_length, 272)
        return tf.convert_to_tensor(keypoints_array, dtype=tf.float32)