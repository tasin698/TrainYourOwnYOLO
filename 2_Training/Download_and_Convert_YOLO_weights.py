import os
import subprocess
import time
import argparse

FLAGS = None

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_folder = os.path.join(root_folder, "2_Training", "src", "keras_yolo3")
data_folder = os.path.join(root_folder, "Data")
model_folder = os.path.join(data_folder, "Model_Weights")

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--download_folder",
        type=str,
        default=download_folder,
        help="Folder to download weights to. Default is " + download_folder,
    )

    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )

    FLAGS = parser.parse_args()

    if not FLAGS.is_tiny:
        weights_file = "yolov3.weights"
        h5_file = "yolo.h5"
        cfg_file = "yolov3.cfg"
        # Original URL: https://pjreddie.com/media/files/yolov3.weights
        gdrive_id = "1ENKguLZbkgvM8unU3Hq1BoFzoLeGWvE_"

    else:
        weights_file = "yolov3-tiny.weights"
        h5_file = "yolo-tiny.h5"
        cfg_file = "yolov3-tiny.cfg"
        # Original URL: https://pjreddie.com/media/files/yolov3-tiny.weights
        gdrive_id = "1mIEZthXBcEguMvuVAHKLXQX3mA1oZUuC"

    weights_path = os.path.join(download_folder, weights_file)
    h5_path = os.path.join(download_folder, h5_file)
    
    # Download weights if not present
    if not os.path.isfile(weights_path):
        print(f"\nDownloading Raw {weights_file}\n")
        start = time.time()
        call_string = " ".join(
            [
                "cd ..",
                "\n",
                "cd",
                download_folder,
                "\n",
                "gdown",
                f"https://drive.google.com/uc?id={gdrive_id}",
            ]
        )

        subprocess.call(call_string, shell=True)

        end = time.time()
        print(f"Downloaded Raw {weights_file} in {end - start:.1f} seconds\n")

        # Convert weights to H5 if H5 file doesn't exist
    if not os.path.isfile(h5_path):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Weights file {weights_file} not found at {weights_path}. "
                f"Please run the download script first."
            )
        print(f"\nConverting {weights_file} to {h5_file}\n")
        print(f"Weights file: {weights_path}")
        print(f"Output file: {h5_path}")
        print(f"Working directory: {download_folder}\n")

        call_string = f"python convert.py {cfg_file} {weights_file} {h5_file}"
        result = subprocess.call(call_string, shell=True, cwd=download_folder)

        # Verify conversion was successful
        if result == 0 and os.path.isfile(h5_path):
            file_size = os.path.getsize(h5_path) / (1024 * 1024)  # Size in MB
            print(f"\n✓ Successfully converted {weights_file} to {h5_file}")
            print(f"  File size: {file_size:.2f} MB")
            print(f"  Location: {h5_path}\n")
        else:
            error_msg = f"Conversion failed with exit code {result}"
            if not os.path.isfile(h5_path):
                error_msg += f" - {h5_file} was not created"
            print(f"\n✗ Error: {error_msg}\n")
            raise RuntimeError(f"Failed to convert {weights_file} to {h5_file}")
        else:
            file_size = os.path.getsize(h5_path) / (1024 * 1024)  # Size in MB
        print(f"\n✓ {h5_file} already exists at {h5_path}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Skipped conversion.\n")
