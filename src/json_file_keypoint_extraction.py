import json
import numpy as np
import tensorflow as tf

class KeypointProcessor:
    """
    Preprocesses keypoint data for use in a machine learning model.

    This class provides methods for skipping confidence values, extending keypoint sequences to a
    fixed length, standardizing keypoints, and converting them to TensorFlow tensors.

    Attributes:
        max_seq_length (int): The desired maximum sequence length for the keypoints.
            Sequences shorter than this will be extended, while longer sequences will be truncated.

    """

    def __init__(self, max_seq_length=370):
        """
        Initializes the KeypointProcessor with the given maximum sequence length.

        Args:
            max_seq_length (int): The desired maximum sequence length for the keypoints.
        """

        self.max_seq_length = max_seq_length

    def skip_confidence_values(self, keypoints):
        """
        Removes confidence values from the keypoint sequence.

        Assumes that keypoints are organized as [x1, y1, c1, x2, y2, c2, ...],
        where (xi, yi) are the coordinates of the i-th keypoint and ci is its confidence score.

        Args:
            keypoints (list): The input sequence of keypoints with confidence values.

        Returns:
            list: A new list containing only the x and y coordinates of the keypoints.
        """

        return [value for index, value in enumerate(keypoints) if (index % 3) != 2]

    def extend_keypoints(self, keypoints):
        """
        Extends or truncates the keypoint sequence to match the `max_seq_length`.

        If the input sequence is shorter than `max_seq_length`, it's extended by repeating
        the frames. If it's longer, it gets truncated.

        Args:
            keypoints (list): The input sequence of keypoints.

        Returns:
            list: The keypoint sequence extended or truncated to `max_seq_length`.
        """

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
        """
        Standardizes the keypoints by subtracting the mean and dividing by the standard deviation.

        Args:
            keypoints (list): The input sequence of keypoints.

        Returns:
            list: The standardized keypoints.
        """

        keypoints_array = np.array(keypoints).reshape(-1, 272)
        mean = np.mean(keypoints_array, axis=0)
        std = np.std(keypoints_array, axis=0)
        standardized_keypoints = (keypoints_array - mean) / std
        return standardized_keypoints.flatten().tolist()

    def build_tensors(self, keypoints):
        """
        Converts the keypoints to a TensorFlow tensor.

        Args:
            keypoints (list): The input sequence of keypoints.

        Returns:
            tf.Tensor: A tensor containing the keypoints.
        """

        keypoints_array = np.array(keypoints).reshape(1, self.max_seq_length, 272)
        return tf.convert_to_tensor(keypoints_array, dtype=tf.float32)