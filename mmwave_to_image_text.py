import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
import os
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor

def range_fft(data: np.ndarray, N: int) -> np.ndarray:
    """
    Perform range FFT with zero-padding and a Hann window.

    Parameters:
        data (np.ndarray): Input complex data of shape (n_sample, n_channel).
        N (int): Interpolation factor for zero-padding.

    Returns:
        np.ndarray: Range profile after FFT, of shape (n_sample * N, n_channel).
    """
    # Validate inputs
    if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
        raise ValueError("Input data must be a complex-valued NumPy array.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Interpolation factor N must be a positive integer.")

    # Extract dimensions of the input data
    n_sample, n_channel = data.shape

    # Zero-padding the data
    interpolated_data = np.zeros((n_sample * N, n_channel), dtype='complex')
    interpolated_data[0:n_sample, :] = data

    # Apply a Hann window
    window = np.hanning(n_sample * N)

    # Perform FFT for each channel
    range_profile = np.zeros((n_sample * N, n_channel), dtype='complex')
    for m in range(n_channel):
        range_profile[:, m] = np.fft.fft(interpolated_data[:, m] * window, n_sample * N)

    return range_profile

def azimuth_fft(range_profile: np.ndarray) -> np.ndarray:
    """
    Perform azimuth FFT with specified channels and 120-point FFT.

    Parameters:
        range_profile (np.ndarray): Input complex data of shape (n_sample, n_channel).

    Returns:
        np.ndarray: Azimuth profile after FFT, of shape (n_sample, 120).
    """
    # Validate input
    if not isinstance(range_profile, np.ndarray) or not np.iscomplexobj(range_profile):
        raise ValueError("Input range_profile must be a complex-valued NumPy array.")

    n_sample, n_channel = range_profile.shape

    # Define channels to use for azimuth FFT
    selected_channels = [0, 1, 2, 3, 8, 9, 10, 11]
    if max(selected_channels) >= n_channel:
        raise ValueError("Selected channels exceed the available range in range_profile.")

    # Initialize azimuth profile
    azimuth_profile = np.zeros((n_sample, 120), dtype='complex')

    # Compute azimuth FFT for each sample
    for m in range(n_sample):
        # Extract selected channels for current sample
        temp = range_profile[m, selected_channels]
        # Perform FFT and apply FFT shift
        azimuth_profile[m, :] = np.fft.fftshift(np.fft.fft(temp, 120))

    return azimuth_profile

def elevation_fft(range_profile: np.ndarray) -> np.ndarray:
    """
    Perform elevation FFT using selected channels and a 60-point FFT.

    Parameters:
        range_profile (np.ndarray): Input complex data of shape (n_sample, n_channel).

    Returns:
        np.ndarray: Elevation profile after FFT, of shape (n_sample, 60).
    """
    # Validate input
    if not isinstance(range_profile, np.ndarray) or not np.iscomplexobj(range_profile):
        raise ValueError("Input range_profile must be a complex-valued NumPy array.")

    n_sample, n_channel = range_profile.shape

    # Define channels to use for elevation FFT
    selected_channels = [9, 7]
    if max(selected_channels) >= n_channel:
        raise ValueError("Selected channels exceed the available range in range_profile.")

    # Initialize elevation profile
    elevation_profile = np.zeros((n_sample, 60), dtype='complex')

    # Compute elevation FFT for each sample
    for m in range(n_sample):
        # Extract selected channels for current sample
        temp = range_profile[m, selected_channels]
        # Perform FFT and apply FFT shift
        elevation_profile[m, :] = np.fft.fftshift(np.fft.fft(temp, 60))

    return elevation_profile

def find_target_point(map_data: np.ndarray, threshold_ratio: float) -> np.ndarray:
    """
    Identify the target point in a map by thresholding and clustering.

    Parameters:
        map_data (np.ndarray): 2D array representing the map data.
        threshold_ratio (float): Ratio to determine the threshold value relative to the maximum value in the map.

    Returns:
        np.ndarray: The cluster center coordinates as a 1D array [x, y].
    """
    # Validate inputs
    if not isinstance(map_data, np.ndarray) or map_data.ndim != 2:
        raise ValueError("map_data must be a 2D NumPy array.")
    if not (0 < threshold_ratio <= 1):
        raise ValueError("threshold_ratio must be a float in the range (0, 1].")

    # Calculate the threshold value
    threshold = threshold_ratio * np.max(map_data)

    # Find indices of points that meet the threshold
    indices = np.where(map_data >= threshold)
    if len(indices[0]) == 0:
        raise ValueError("No points found above the specified threshold.")

    # Convert indices to an array of data points
    data_points = np.array(indices).T

    # Calculate the cluster center (mean of points)
    cluster_center = np.mean(data_points, axis=0)

    return cluster_center

def process_frame(transposed_data: np.ndarray,
                  sample_rate: float,
                  c: float,
                  slope: float,
                  n_sample: int,
                  N: int,
                  noise: np.ndarray,
                  target_frames: int) -> np.ndarray:
    """
    Process radar frames to compute the filtered trajectory of a target in range, azimuth, and elevation.

    Parameters:
        transposed_data (np.ndarray): Transposed input radar data of shape (target_frames, n_sample, n_channel).
        sample_rate (float): Sampling rate of the radar.
        c (float): Speed of light in m/s.
        slope (float): Slope of the radar chirp in Hz/s.
        n_sample (int): Number of samples per frame.
        N (int): Interpolation factor.
        noise (np.ndarray): Noise matrix of the same shape as a single frame in transposed_data.
        target_frames (int): Total number of frames to process.

    Returns:
        np.ndarray: Filtered target trajectory data of shape (filtered_frames, 3) with columns [range, azimuth, elevation].
    """
    # Compute range resolution
    range_resolution = sample_rate * c / (2 * slope * n_sample * N)

    # Initialize storage for processed data
    ra_map = np.zeros((target_frames, 30 * N, 120), dtype=np.float32)
    re_map = np.zeros((target_frames, 30 * N, 60), dtype=np.float32)
    trajectory_rae = np.zeros((target_frames, 3), dtype=np.float32)

    # Process each frame
    for i in range(target_frames):
        # Range-azimuth and range-elevation processing
        current_data = transposed_data[i, :, :] - noise
        range_fft_result = range_fft(current_data, N)[:30 * N, :]

        ra_map[i, :, :] = abs(azimuth_fft(range_fft_result))
        re_map[i, :, :] = abs(elevation_fft(range_fft_result))

        # Find cluster centers for RA and RE maps
        cluster_center_ra = find_target_point(ra_map[i, :, :], 0.75)
        cluster_center_re = find_target_point(re_map[i, :, :], 0.75)

        # Compute target trajectory in range, azimuth, and elevation
        trajectory_rae[i, 0] = cluster_center_ra[0] * range_resolution  # Range in meters
        trajectory_rae[i, 1] = cluster_center_ra[1] - 60  # Azimuth angle
        trajectory_rae[i, 2] = cluster_center_re[1] - 30  # Elevation angle

    # Filter outliers from trajectory data
    mean_values = np.mean(trajectory_rae, axis=0)
    std_values = np.std(trajectory_rae, axis=0)
    threshold = 3  # Threshold for standard deviation filtering

    filtered_data = trajectory_rae[
        (np.abs(trajectory_rae[:, 0] - mean_values[0]) <= threshold * std_values[0]) &
        (np.abs(trajectory_rae[:, 1] - mean_values[1]) <= threshold * std_values[1]) &
        (np.abs(trajectory_rae[:, 2] - mean_values[2]) <= threshold * std_values[2])
        ]

    return filtered_data

def smooth_and_reconstruct_trajectory(trajectory: np.ndarray,
                                      use_kalman: bool = True,
                                      kalman_observation_covariance: float = 1.0) -> np.ndarray:
    """
    Smooth and reconstruct a 3D trajectory using Kalman filtering.

    Parameters:
        trajectory (np.ndarray): Input trajectory of shape (n_frames, 3) with columns [x, y, z].
        use_kalman (bool): Whether to use Kalman filtering for smoothing.
        kalman_observation_covariance (float): Covariance value for Kalman filter observations.

    Returns:
        np.ndarray: Smoothed trajectory of shape (n_frames, 3).
    """
    if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("Input 'trajectory' must be a NumPy array of shape (n_frames, 3).")

    if not isinstance(kalman_observation_covariance, (int, float)) or kalman_observation_covariance <= 0:
        raise ValueError("kalman_observation_covariance must be a positive float or integer.")

    # Kalman filter settings
    kalman_transition_covariance = 1e-3  # Small covariance for smooth transitions
    smoothed_trajectory = trajectory

    if use_kalman:
        # Initialize Kalman filter
        kf = KalmanFilter(
            initial_state_mean=[trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 0, 0, 0],
            transition_matrices=[
                [1, 0, 0, 1, 0, 0],  # Position x depends on velocity x
                [0, 1, 0, 0, 1, 0],  # Position y depends on velocity y
                [0, 0, 1, 0, 0, 1],  # Position z depends on velocity z
                [0, 0, 0, 1, 0, 0],  # Velocity x remains constant
                [0, 0, 0, 0, 1, 0],  # Velocity y remains constant
                [0, 0, 0, 0, 0, 1]  # Velocity z remains constant
            ],
            observation_matrices=[
                [1, 0, 0, 0, 0, 0],  # Observing x
                [0, 1, 0, 0, 0, 0],  # Observing y
                [0, 0, 1, 0, 0, 0]  # Observing z
            ],
            transition_covariance=kalman_transition_covariance * np.eye(6),
            observation_covariance=kalman_observation_covariance * np.eye(3)
        )

        # Apply Kalman smoothing
        kalman_smoothed, _ = kf.smooth(trajectory)
        smoothed_trajectory = kalman_smoothed[:, :3]  # Extract position components (x, y, z)

    return smoothed_trajectory

def project_3d_trajectory_to_2d_pca(trajectory_rae: np.ndarray) -> np.ndarray:
    """
    Project a 3D trajectory (range, azimuth, elevation) to 2D using PCA.

    Parameters:
        trajectory_rae (np.ndarray): Input trajectory in spherical coordinates (r, azimuth, elevation),
                                  of shape (n_frames, 3).

    Returns:
        np.ndarray: 2D projection of the trajectory after applying PCA, of shape (n_frames, 2).
    """
    # Validate input
    if not isinstance(trajectory_rae, np.ndarray) or trajectory_rae.ndim != 2 or trajectory_rae.shape[1] != 3:
        raise ValueError("Input 'trajectory_rae' must be a NumPy array of shape (n_frames, 3).")

    # Smooth and reconstruct the trajectory
    smoothed_ra = smooth_and_reconstruct_trajectory(trajectory_rae)

    # Convert spherical coordinates (r, azimuth, elevation) to Cartesian coordinates (x, y, z)
    x_ra = smoothed_ra[:, 0] * np.sin(np.radians(smoothed_ra[:, 1])) * np.cos(np.radians(smoothed_ra[:, 2]))
    y_ra = smoothed_ra[:, 0] * np.cos(np.radians(smoothed_ra[:, 1])) * np.cos(np.radians(smoothed_ra[:, 2]))
    z_ra = smoothed_ra[:, 0] * np.sin(np.radians(smoothed_ra[:, 2]))

    # Stack the Cartesian coordinates
    cartesian_trajectory = np.vstack((x_ra, y_ra, z_ra)).T

    # Apply PCA to reduce 3D trajectory to 2D
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(cartesian_trajectory)

    # Determine the direction vector (start to end point) in 2D
    start_point = trajectory_2d[0]
    end_point = trajectory_2d[-1]
    direction_vector = end_point - start_point

    # Adjust direction based on principal writing directions
    if direction_vector[0] < 0:  # Flip X-axis if the primary direction is negative
        trajectory_2d[:, 0] = -trajectory_2d[:, 0]
    if direction_vector[1] > 0:  # Flip Y-axis if the secondary direction is positive
        trajectory_2d[:, 1] = -trajectory_2d[:, 1]

    return trajectory_2d

def save_and_label_trajectory(trajectory_2d: np.ndarray, file_path: str) -> None:
    """
    Save a 2D trajectory as a PNG image and record the corresponding label in a JSONL file.

    The label is extracted from the second last directory in the input file path.
    The image is saved in mmPencil_dataset/vision/, and the JSONL entry is stored in mmPencil_dataset/text/.

    Parameters:
        trajectory_2d (np.ndarray): Array of shape (n_frames, 2), representing 2D trajectory points.
        file_path (str): Path to the original .npy file. Used to determine save locations and label.
    """
    # Extract path components
    path_parts = Path(file_path).parts

    # Use the second last folder name as the label (e.g., 'cat' in '.../cat/filename.npy')
    label = path_parts[-2]

    # Define the root directory for output
    base_dir = "mmPencil_dataset"

    # Derive relative path to the .npy file, used to structure vision/ output
    relative_path = os.path.relpath(os.path.dirname(file_path), os.path.join(base_dir, "mmWave"))

    # Define full path to save the output PNG image
    png_save_path = os.path.join(base_dir, "vision", relative_path,
                                 os.path.basename(file_path).replace(".npy", ".png"))

    # Define JSONL path: e.g., mmPencil_dataset/text/session-id.jsonl
    json_file_name = f"{path_parts[-3]}.jsonl"
    json_save_dir = os.path.join(base_dir, "text", path_parts[-4])
    json_save_path = os.path.join(json_save_dir, json_file_name)

    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(png_save_path), exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)

    # Compute bounding box of the trajectory
    x_min, x_max = np.min(trajectory_2d[:, 0]), np.max(trajectory_2d[:, 0])
    y_min, y_max = np.min(trajectory_2d[:, 1]), np.max(trajectory_2d[:, 1])

    # Add small padding around the bounding box
    padding = 0.005 * max(x_max - x_min, y_max - y_min)
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Set figure size while preserving aspect ratio
    fig_width = x_max - x_min
    fig_height = y_max - y_min
    aspect_ratio = fig_width / fig_height if fig_height != 0 else 1.0
    fig_size = (8, 8 / aspect_ratio)

    # Plot the trajectory and save as image
    plt.figure(figsize=fig_size)
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], color='black', linewidth=7)
    plt.axis('off')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(png_save_path, bbox_inches='tight', dpi=100)
    plt.close()

    # Prepare the JSONL entry with the prompt and label
    new_message = {
        "messages": [
            {
                "role": "user",
                "content": "<image>Here is a picture containing a handwriting word. Please identify the word in the image. Ensure that your output is a correctly spelled English word. Note: Only provide the word itself, without any additional content or explanation."
            },
            {
                "role": "assistant",
                "content": label
            }
        ],
        "images": [png_save_path]
    }

    # Append the entry to the JSONL file (one JSON object per line)
    with open(json_save_path, 'a') as f:
        f.write(json.dumps(new_message) + '\n')

def process_single_file(file_path: str, n_sample: int, n_channel: int, slope: float, sample_rate: float, N: int, c: float) -> None:
    """
    Process a single .npy file to extract and project the trajectory.
    Parameters:
        file_path (str): Path to the .npy file.
        n_sample (int): Number of samples per frame.
        n_channel (int): Number of channels per frame.
        slope (float): Radar chirp slope (Hz/s).
        sample_rate (float): Sampling rate (Hz).
        N (int): FFT interpolation factor.
        c (float): Speed of light (m/s).
    """
    try:
        data_cube = np.load(file_path)
    except Exception as e:
        print(f"[Error] Failed to load: {file_path}\n  {e}")
        return None
    if data_cube.ndim != 3 or data_cube.shape[1:] != (n_sample, n_channel):
        print(f"[Warning] Skipping due to unexpected shape: {file_path}")
        return None

    target_frames = data_cube.shape[0]
    noise = data_cube.mean(axis=0)

    try:
        trajectory_rae = process_frame(data_cube, sample_rate, c, slope, n_sample, N, noise, target_frames)
        trajectory_2d = project_3d_trajectory_to_2d_pca(trajectory_rae)

        # Define the root directory for output
        base_dir = "mmPencil_dataset"
        # Derive relative path to the .npy file, used to structure vision/ output
        relative_path = os.path.relpath(os.path.dirname(file_path), os.path.join(base_dir, "mmWave"))
        # Define full path to save the output PNG image
        png_save_path = os.path.join(base_dir, "vision", relative_path,
                                     os.path.basename(file_path).replace(".npy", ".png"))

        # Check if the PNG file already exists
        if os.path.exists(png_save_path):
            print(f"[Info] Skipping {file_path} as {png_save_path} already exists.")
            return None

        save_and_label_trajectory(trajectory_2d, file_path)

    except Exception as e:
        print(f"[Error] Processing failed: {file_path}\n  {e}")


if __name__ == '__main__':
    # Define key parameters
    n_sample = 108  # Number of samples per frame
    n_channel = 12  # Number of channels per frame
    slope = 66.0105e12  # Radar chirp slope (Hz/s)
    sample_rate = 2e6  # Sampling rate (Hz)
    N = 4  # FFT interpolation factor
    c = 3e8  # Speed of light (m/s)

    root_dir = "mmPencil_dataset/mmWave"

    # Collect all .npy file paths under the directory
    npy_file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_file_list.append(os.path.join(dirpath, filename))
    print(f"Found {len(npy_file_list)} .npy files. Starting processing...\n")

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(
            process_single_file,
            npy_file_list,
            [n_sample] * len(npy_file_list),
            [n_channel] * len(npy_file_list),
            [slope] * len(npy_file_list),
            [sample_rate] * len(npy_file_list),
            [N] * len(npy_file_list),
            [c] * len(npy_file_list)),
            total=len(npy_file_list), desc="Processing Trajectories", unit="file"))