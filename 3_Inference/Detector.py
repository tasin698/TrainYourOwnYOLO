import os
import cv2
import numpy as np
import sys
import argparse
from PIL import Image
from timeit import default_timer as timer
import pandas as pd

# --- TTA Function ---
def apply_tta(image):
    """Applies Test-Time Augmentation (horizontal flip)."""
    flipped = cv2.flip(image, 1)
    return [image, flipped]

# --- Path Setup ---
def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    # Use os.path.realpath to get the actual file path in Colab
    current_path = os.path.dirname(os.path.realpath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

# Note: In Colab, __file__ isn't defined. This script is intended to be run
# from the command line, so we'll assume a standard structure.
# For Colab, paths are often absolute, e.g., /content/TrainYourOwnYOLO/
base_path = "/content/TrainYourOwnYOLO" # Adjust if your repo clone is named differently

src_path = os.path.join(base_path, "2_Training", "src")
utils_path = os.path.join(base_path, "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
from utils import load_extractor_model, load_features, parse_input, detect_object
# import test # Not used
# import utils # Already imported detect_object
from Get_File_Paths import GetFileList
# import random # Not used
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(base_path, "Data")
image_folder = os.path.join(data_folder, "Source_Images")
image_test_folder = os.path.join(image_folder, "MyTest_Images")
detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")
model_folder = os.path.join(data_folder, "Model_Weights")

# Check for .weights.h5 first (TensorFlow 2.10+ format), fallback to .h5 for backward compatibility
model_weights = os.path.join(model_folder, "trained_weights_final.weights.h5")
if not os.path.isfile(model_weights):
    model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

# This path might be different if you are in 3_Inference
# Let's use the base_path
anchors_path = os.path.join(base_path, "2_Training", "src", "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    parser.add_argument(
        "--input_path",
        type=str,
        default=image_test_folder,
        help="Path to image/video directory. All subdirectories will be included. Default is "
        + image_test_folder,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=detection_results_folder,
        help="Output path for detection results. Default is "
        + detection_results_folder,
    )

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=model_classes,
        help="Path to YOLO class specifications. Default is " + model_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )

    parser.add_argument(
        "--box_file",
        type=str,
        dest="box",
        default=detection_results_file,
        help="File to save bounding box results to. Default is "
        + detection_results_file,
    )

    parser.add_argument(
        "--postfix",
        type=str,
        dest="postfix",
        default="_catface",
        help='Specify the postfix for images with bounding boxes. Default is "_catface"',
    )

    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )

    parser.add_argument(
        "--webcam",
        default=False,
        action="store_true",
        help="Use webcam for real-time detection. Default is False.",
    )
    
    # --- NEW ARGUMENTS ---
    
    parser.add_argument(
        "--nms",
        type=float,
        dest="nms_threshold",
        default=0.45,
        help="Non-Max Suppression threshold. Default is 0.45.",
    )
    
    parser.add_argument(
        "--tta",
        default=False,
        action="store_true",
        help="Use Test-Time Augmentation (horizontal flip). Default is False.",
    )

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img
    file_types = FLAGS.file_types
    webcam_active = FLAGS.webcam

    if file_types:
        input_paths = GetFileList(FLAGS.input_path, endings=file_types)
    else:
        input_paths = GetFileList(FLAGS.input_path)

    # Split images and videos
    img_endings = (".jpg", ".jpeg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        item_normalized = str(item).lower()
        item_basename = os.path.basename(item)
        
        if (item_basename.startswith("._") or 
            "__macosx" in item_normalized or 
            item_basename.startswith(".DS_Store")):
            continue
            
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if FLAGS.is_tiny and FLAGS.anchors_path == anchors_path:
        anchors_path = os.path.join(
            os.path.dirname(FLAGS.anchors_path), "yolo-tiny_anchors.txt"
        )

    anchors = get_anchors(anchors_path)
    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "nms_threshold": FLAGS.nms_threshold, # <-- NEW NMS FLAG USED HERE
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image", "image_path", "xmin", "ymin", "xmax", "ymax",
            "label", "confidence", "x_size", "y_size",
        ]
    )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    if input_image_paths and not webcam_active:
        print(
            "Found {} input images: {} ...".format(
                len(input_image_paths),
                [os.path.basename(f) for f in input_image_paths[:5]],
            )
        )
        start = timer()
        text_out = ""

        # --- MODIFIED MAIN LOOP ---
        for i, img_path in enumerate(input_image_paths):
            print(f"Processing {img_path} ({i+1}/{len(input_image_paths)})")
            
            # This list will hold all predictions for this image (original + TTA)
            final_predictions_for_this_image = []
            
            # --- 1. Original Detection ---
            try:
                prediction, image = detect_object(
                    yolo,
                    img_path,
                    save_img=save_img,
                    save_img_path=FLAGS.output,
                    postfix=FLAGS.postfix,
                )
            except Exception as e:
                print(f"Skipping {img_path} - error during original detection: {e}")
                continue
            
            if image is None:
                print(f"Skipping {img_path} - file could not be opened")
                continue
                
            try:
                image_array = np.asarray(image)
                if image_array.size == 0 or len(image_array.shape) < 3:
                    print(f"Skipping {img_path} - invalid image shape: {image_array.shape}")
                    continue
                y_size, x_size, _ = image_array.shape
            except (ValueError, AttributeError, TypeError) as e:
                print(f"Skipping {img_path} - error processing image: {e}")
                continue

            if prediction is not None and len(prediction) > 0:
                final_predictions_for_this_image.extend(prediction)

            # --- 2. TTA (Flip) Detection (if enabled) ---
            if FLAGS.tta:
                temp_img_path = "temp_tta_flip.jpg"
                try:
                    # Use the loaded image array, flip, and save temporarily
                    flipped_np = cv2.flip(image_array, 1)
                    flipped_pil = Image.fromarray(flipped_np)
                    flipped_pil.save(temp_img_path)
                    
                    # Run detection on the temporary flipped image
                    prediction_flipped, _ = detect_object(
                        yolo,
                        temp_img_path,
                        save_img=False,  # Don't save the annotated TTA image
                        save_img_path=FLAGS.output,
                        postfix=FLAGS.postfix,
                    )
                    
                    if prediction_flipped is not None and len(prediction_flipped) > 0:
                        un_flipped_preds = []
                        for pred in prediction_flipped:
                            # pred = [xmin, ymin, xmax, ymax, label, confidence]
                            xmin, ymin, xmax, ymax, label, confidence = pred
                            
                            # "Un-flip" the coordinates
                            new_xmin = x_size - xmax
                            new_xmax = x_size - xmin
                            
                            un_flipped_preds.append([new_xmin, ymin, new_xmax, ymax, label, confidence])
                        
                        final_predictions_for_this_image.extend(un_flipped_preds)
                        
                except Exception as e:
                    print(f"Error during TTA processing for {img_path}: {e}")
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path) # Clean up temp file

            # --- 3. Add all collected predictions to DataFrame ---
            if len(final_predictions_for_this_image) > 0:
                for single_prediction in final_predictions_for_this_image:
                    new_row = pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image", "image_path", "xmin", "ymin", "xmax", "ymax",
                            "label", "confidence", "x_size", "y_size",
                        ],
                    )
                    out_df = pd.concat([out_df, new_row], ignore_index=True)
            
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        out_df.to_csv(FLAGS.box, index=False)

    # --- Video and Webcam processing (unchanged) ---
    if input_video_paths and not webcam_active:
        print(
            "Found {} input videos: {} ...".format(
                len(input_video_paths),
                [os.path.basename(f) for f in input_video_paths[:5]],
            )
        )
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                FLAGS.output,
                os.path.basename(vid_path).replace(".", FLAGS.postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print(
            "Processed {} videos in {:.1f}sec".format(
                len(input_video_paths), end - start
            )
        )

    if webcam_active:
        start = timer()
        detect_webcam(yolo)
        end = timer()
        print("Processed from webcam for {:.1f}sec".format(end - start))

    # Close the current yolo session
    yolo.close_session()
