# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from .yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from .yolo3.utils import letterbox_image
import os
import tensorflow as tf
from tensorflow.keras import backend as K


class YOLO(object):
    _defaults = {
        "model_path": "model_data/yolo.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "classes_path": "model_data/coco_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        # Allow both .h5 and .weights.h5 extensions
        assert model_path.endswith(".h5") or model_path.endswith(".weights.h5"), \
            "Keras model or weights must be a .h5 or .weights.h5 file."

        # Check for .weights.h5 first (TensorFlow 2.10+ format), fallback to .h5 for backward compatibility
        if not os.path.isfile(model_path):
            # Try alternate extension if file doesn't exist
            base_path = model_path.replace(".weights.h5", "").replace(".h5", "")
            weights_path = base_path + ".weights.h5"
            h5_path = base_path + ".h5"
            
            # Prefer .weights.h5 (TensorFlow 2.10+ format), then .h5
            if os.path.isfile(weights_path):
                model_path = weights_path
                self.model_path = model_path
            elif os.path.isfile(h5_path):
                model_path = h5_path
                self.model_path = model_path

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = (
                tiny_yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 2, num_classes
                )
                if is_tiny_version
                else yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 3, num_classes
                )
            )
            self.yolo_model.load_weights(
                model_path
            )  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(
                self.yolo_model.output
            ) * (
                num_classes + 5
            ), "Mismatch between model and given anchor and class sizes"

        end = timer()
        print(
            "{} model, anchors, and classes loaded in {:.2f}sec.".format(
                model_path, end - start
            )
        )

        # Generate colors for drawing bounding boxes.
        if len(self.class_names) == 1:
            self.colors = ["GreenYellow"]
        else:
            hsv_tuples = [
                (x / len(self.class_names), 1.0, 1.0)
                for x in range(len(self.class_names))
            ]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(
                    lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors,
                )
            )
            np.random.seed(10101)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(
                self.colors
            )  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        # Note: In eager execution, we compute these dynamically in detect_image
        # These are placeholders and won't be used directly
        dummy_shape = tf.constant([416, 416], dtype=tf.float32)
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            dummy_shape,  # placeholder, actual shape computed in detect_image
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes

    def detect_image(self, image, show_stats=True):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            input_shape = tf.constant(self.model_image_size, dtype=tf.float32)
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
            input_shape = tf.constant([new_image_size[1], new_image_size[0]], dtype=tf.float32)
        
        image_data = np.array(boxed_image, dtype="float32")
        if show_stats:
            print(image_data.shape)
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # Get predictions using eager execution
        model_outputs = self.yolo_model(image_data, training=False)
        
        # YOLO models return list of tensors [y1, y2, y3] for full YOLO or [y1, y2] for tiny
        # Ensure model_outputs is a list (in case it's a tuple or single tensor)
        if isinstance(model_outputs, tuple):
            model_outputs = list(model_outputs)
        elif not isinstance(model_outputs, list):
            model_outputs = [model_outputs]
        
        # Re-evaluate boxes with correct image shape
        image_shape_tensor = tf.constant([image.size[1], image.size[0]], dtype=tf.float32)
        boxes, scores, classes = yolo_eval(
            model_outputs,
            self.anchors,
            len(self.class_names),
            image_shape_tensor,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        
        # Convert tensors to numpy arrays
        out_boxes = boxes.numpy()
        out_scores = scores.numpy()
        out_classes = classes.numpy()
        if show_stats:
            print("Found {} boxes for {}".format(len(out_boxes), "img"))
        out_prediction = []

        font_path = os.path.join(os.path.dirname(__file__), "font/FiraMono-Medium.otf")
        font = ImageFont.truetype(
            font=font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32")
        )
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # textsize is deprecated in Pillow 10.0.0+, use textbbox instead
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except AttributeError:
                label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype("int32"))
            left = max(0, np.floor(left + 0.5).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
            right = min(image.size[0], np.floor(right + 0.5).astype("int32"))

            # Validate bounding box coordinates: ensure left < right and top < bottom
            # Skip invalid boxes (can happen with coordinate conversion issues)
            if top >= bottom or left >= right:
                continue
            # image was expanded to model_image_size: make sure it did not pick
            # up any box outside of original image (run into this bug when
            # lowering confidence threshold to 0.01)
            if top > image.size[1] or right > image.size[0]:
                continue
            if show_stats:
                print(label, (left, top), (right, bottom))

            # output as xmin, ymin, xmax, ymax, class_index, confidence
            out_prediction.append([left, top, right, bottom, c, score])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, bottom])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline=self.colors[c]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c],
            )

            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        if show_stats:
            print("Time spent: {:.3f}sec".format(end - start))
        return out_prediction, image

    def close_session(self):
        # No session to close in eager execution
        pass


def detect_video(yolo, video_path, output_path=""):
    import cv2

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")  # int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    isOutput = True if output_path != "" else False
    if isOutput:
        print(
            "Processing {} with frame size {} at {:.1f} FPS".format(
                os.path.basename(video_path), video_size, video_fps
            )
        )
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:, :, ::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, show_stats=False)
        result = np.asarray(image)
        # Make a writable copy to avoid readonly array error with cv2.putText
        result = result.copy()
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(
            result,
            text=fps,
            org=(3, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50,
            color=(255, 0, 0),
            thickness=2,
        )
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result[:, :, ::-1])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    vid.release()
    out.release()
    # yolo.close_session()


# TO PROCESS VIDEOS DIRECTLY FROM WEBCAM
def detect_webcam(yolo):
    import cv2

    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, show_stats=False)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(
            result,
            text=fps,
            org=(3, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50,
            color=(255, 0, 0),
            thickness=2,
        )
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result)
        cv2.waitKey(1)
        if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            break
    vid.release()
    yolo.close_session()
