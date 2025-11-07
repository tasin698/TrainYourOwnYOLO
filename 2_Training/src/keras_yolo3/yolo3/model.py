"""YOLO_v3 Model Defined in Keras."""

from functools import wraps
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
    Lambda,
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

from ..yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {"kernel_regularizer": l2(5e-4)}
    darknet_conv_kwargs["padding"] = (
        "valid" if kwargs.get("strides") == (2, 2) else "same"
    )
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {"use_bias": False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)),
        )(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
    )(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)),
    )(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
    )(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)),
    )(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)),
    )(x2)

    x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)),
    )([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    # Get grid_shape - handle KerasTensors during graph construction
    # During model initialization, KerasTensors don't have fully defined shapes
    # We need to compute grid_shape from input_shape or use a placeholder
    from tensorflow.keras import ops
    # Use ops.reshape for KerasTensor compatibility
    anchors_tensor = ops.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    
    # Try to get static shape first (works during graph construction if shape is known)
    grid_h = None
    grid_w = None
    
    if hasattr(feats, 'shape') and feats.shape is not None and len(feats.shape) >= 3:
        try:
            shape_list = feats.shape.as_list() if hasattr(feats.shape, 'as_list') else list(feats.shape)
            if (len(shape_list) >= 3 and 
                shape_list[1] is not None and shape_list[2] is not None and 
                isinstance(shape_list[1], int) and isinstance(shape_list[2], int)):
                # Static shape available - both are integers
                grid_h = int(shape_list[1])
                grid_w = int(shape_list[2])
        except (AttributeError, IndexError, TypeError):
            pass
    
    # If static shape not available, compute from input_shape
    # This is the most reliable approach during graph construction
    # Always set fallback values first to ensure we never have None
    if grid_h is None or grid_w is None:
        # Compute approximate grid_shape from input_shape
        # YOLOv3 uses different strides for different layers, but we'll use a default
        # The actual shape will be computed correctly during inference
        if isinstance(input_shape, tf.Tensor):
            # Use input_shape to compute grid (will be refined during execution)
            # Default to 32x downsampling (largest grid)
            grid_shape_fallback = tf.cast(input_shape / 32, tf.int32)
            grid_h = tf.gather(grid_shape_fallback, 0)  # Use tf.gather for safe indexing
            grid_w = tf.gather(grid_shape_fallback, 1)
        else:
            # Static computation from input_shape - ensure we get integers
            try:
                grid_h = int(input_shape[0] // 32) if input_shape[0] is not None else 13
                grid_w = int(input_shape[1] // 32) if input_shape[1] is not None else 13
            except (TypeError, IndexError):
                # Ultimate fallback - use default YOLOv3 grid size
                grid_h = 13
                grid_w = 13
        
        # Try to refine using ops.shape if possible (works in eager mode)
        # This will override the fallback during execution if shape is available
        try:
            shape_tensor = ops.shape(feats)
            # Use tf.gather to safely extract dimensions
            grid_h_dynamic = tf.cast(tf.gather(shape_tensor, 1), tf.int32)
            grid_w_dynamic = tf.cast(tf.gather(shape_tensor, 2), tf.int32)
            # Use dynamic shape (will override static during execution)
            grid_h = grid_h_dynamic
            grid_w = grid_w_dynamic
        except (ValueError, TypeError, AttributeError):
            # Keep the fallback values computed from input_shape
            pass
    
    # Create grid_shape tensor
    if isinstance(grid_h, (int, np.integer)) and isinstance(grid_w, (int, np.integer)):
        grid_shape = tf.constant([grid_h, grid_w], dtype=tf.int32)
    else:
        grid_shape = tf.stack([tf.cast(grid_h, tf.int32), tf.cast(grid_w, tf.int32)])
    
    # Extract grid dimensions for use in tile operations
    # Ensure we use tensor operations, not Python lists with mixed types
    if isinstance(grid_h, (int, np.integer)):
        grid_h_tensor = tf.constant(grid_h, dtype=tf.int32)
    else:
        grid_h_tensor = tf.cast(grid_h, tf.int32)
    
    if isinstance(grid_w, (int, np.integer)):
        grid_w_tensor = tf.constant(grid_w, dtype=tf.int32)
    else:
        grid_w_tensor = tf.cast(grid_w, tf.int32)
    
    # Use ops operations for KerasTensor compatibility
    arange_h = K.arange(0, stop=grid_h_tensor)  # Creates new tensor, OK
    arange_w = K.arange(0, stop=grid_w_tensor)  # Creates new tensor, OK
    grid_y = ops.tile(
        ops.reshape(arange_h, [-1, 1, 1, 1]),
        ops.stack([1, grid_w_tensor, 1, 1]),
    )
    grid_x = ops.tile(
        ops.reshape(arange_w, [1, -1, 1, 1]),
        ops.stack([grid_h_tensor, 1, 1, 1]),
    )
    # Concatenate along the last dimension to create [grid_h, grid_w, 1, 2]
    grid = ops.concatenate([grid_x, grid_y], axis=-1)
    # Get dtype from tensor attribute (works with KerasTensors)
    # Convert to string if it's a DType object
    if hasattr(feats, 'dtype'):
        feats_dtype = feats.dtype
        if hasattr(feats_dtype, 'name'):
            feats_dtype = feats_dtype.name
        elif not isinstance(feats_dtype, str):
            feats_dtype = str(feats_dtype)
    else:
        feats_dtype = 'float32'
    # Use ops operations for KerasTensor compatibility
    grid = ops.cast(grid, feats_dtype)
    # Reshape grid to [1, grid_h, grid_w, 1, 2] for proper broadcasting with [batch, grid_h, grid_w, num_anchors, 2]
    # Handle both static and dynamic shapes
    if isinstance(grid_h, (int, np.integer)) and isinstance(grid_w, (int, np.integer)):
        grid = ops.reshape(grid, [1, int(grid_h), int(grid_w), 1, 2])
    else:
        # Use Lambda for dynamic reshape when dimensions are tensors (similar to feats reshape)
        def reshape_grid(x):
            """Reshape grid using grid dimensions computed from input tensor."""
            # x has shape [grid_h, grid_w, 1, 2], extract dimensions and add batch dimension
            grid_shape_tf = tf.shape(x)
            grid_h_val = grid_shape_tf[0]
            grid_w_val = grid_shape_tf[1]
            grid_shape = tf.stack([1, grid_h_val, grid_w_val, 1, 2])
            return tf.reshape(x, grid_shape)
        grid = Lambda(reshape_grid)(grid)

    # Reshape feats: use static shape when available, otherwise compute dynamically
    # If grid_h and grid_w are static integers, use a list shape
    if isinstance(grid_h, (int, np.integer)) and isinstance(grid_w, (int, np.integer)):
        # Use static shape - ops.reshape works with list of ints
        feats = ops.reshape(feats, [-1, int(grid_h), int(grid_w), num_anchors, num_classes + 5])
    else:
        # Use dynamic shape - when grid_h/grid_w are tensors, we need to defer reshape until execution
        # CRITICAL FIX: The grid is computed using grid_h_tensor and grid_w_tensor from ops.shape(feats)
        # The Lambda must compute dimensions from the input tensor in the SAME way to ensure they match
        # Instead of capturing the tensors (which may evaluate incorrectly), compute from input shape
        def dynamic_reshape(x):
            """Reshape feats using grid dimensions computed from input tensor (same method as grid)."""
            # Compute shape the same way as grid computation (using ops.shape equivalent)
            feats_shape = tf.shape(x)  # This works during execution
            batch_size = feats_shape[0]
            # Extract grid dimensions the same way as grid computation
            # grid_h_tensor and grid_w_tensor were computed from ops.shape(feats)[1:3]
            # So we compute from tf.shape(x)[1:3] here
            grid_h_val = tf.cast(feats_shape[1], tf.int32)  # Height dimension
            grid_w_val = tf.cast(feats_shape[2], tf.int32)  # Width dimension
            num_anchors_tensor = tf.constant(num_anchors, dtype=tf.int32)
            num_classes_tensor = tf.constant(num_classes + 5, dtype=tf.int32)
            reshape_shape = tf.stack([batch_size, grid_h_val, grid_w_val, num_anchors_tensor, num_classes_tensor])
            return tf.reshape(x, reshape_shape)
        
        feats = Lambda(dynamic_reshape)(feats)

    # Adjust preditions to each spatial grid point and anchor size.
    # Reverse grid_shape: [w, h] instead of [h, w]
    grid_shape_reversed = tf.stack([grid_w_tensor, grid_h_tensor])
    # Get dtype from tensor attribute (works with KerasTensors)
    # Reuse the dtype we already computed above
    # Use ops operations for KerasTensor compatibility
    box_xy = (ops.sigmoid(feats[..., :2]) + grid) / ops.cast(
        grid_shape_reversed, feats_dtype
    )
    # Handle input_shape reversal safely
    if isinstance(input_shape, tf.Tensor):
        input_shape_reversed = tf.reverse(input_shape, [0])
    else:
        input_shape_reversed = input_shape[::-1]
    box_wh = (
        ops.exp(feats[..., 2:4])
        * anchors_tensor
        / ops.cast(input_shape_reversed, feats_dtype)
    )
    box_confidence = ops.sigmoid(feats[..., 4:5])
    box_class_probs = ops.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


from tensorflow.keras import backend as K
from tensorflow.keras import ops  # <-- NEW: Use keras.ops
import tensorflow as tf

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes - Keras 3 compatible"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    # Safe dtype
    dtype = box_yx.dtype if hasattr(box_yx, 'dtype') else K.floatx()
    input_shape = K.cast(input_shape, dtype)
    image_shape = K.cast(image_shape, dtype)

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)

    # Use ops.concatenate
    boxes = ops.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1], # y_max
            box_maxes[..., 1:2], # x_max
        ],
        axis=-1
    )

    # Use ops.repeat
    image_shape_repeated = ops.repeat(image_shape, 2)
    boxes *= image_shape_repeated

    return boxes    


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    from tensorflow.keras import ops
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, num_classes, input_shape
    )
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # Use ops.reshape for KerasTensor compatibility
    boxes = ops.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = ops.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


from tensorflow.keras import ops
from tensorflow.keras import backend as K
import tensorflow as tf


from tensorflow.keras import ops, backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf


from tensorflow.keras import ops, backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf


from tensorflow.keras import ops, backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf


from tensorflow.keras import ops, backend as K
import tensorflow as tf


from tensorflow.keras import ops, backend as K
import tensorflow as tf


from tensorflow.keras import ops, backend as K
import tensorflow as tf


from tensorflow.keras import ops, backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf


from tensorflow.keras import ops, backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf


def yolo_eval(
    yolo_outputs,
    anchors,
    num_classes,
    image_shape,
    max_boxes=20,
    score_threshold=0.6,
    iou_threshold=0.5,
):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )

    # Safe input_shape
    if isinstance(image_shape, tf.Tensor):
        input_shape = ops.cast(image_shape, "float32")
    elif isinstance(image_shape, (list, tuple)):
        input_shape = ops.convert_to_tensor([float(image_shape[0]), float(image_shape[1])])
    else:
        try:
            output_shape = ops.shape(yolo_outputs[0])
            input_shape = ops.cast(output_shape[1:3], "float32") * 32
        except:
            input_shape = ops.convert_to_tensor([416.0, 416.0])

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            image_shape,
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = ops.concatenate(boxes, axis=0)
    box_scores = ops.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = ops.convert_to_tensor(max_boxes, dtype="int32")

    # === Lambda layer with output_shape ===
    def nms_per_class(inputs):
        boxes, box_scores, mask, c = inputs
        mask_c = tf.gather(mask, c, axis=1)
        scores_c = tf.gather(box_scores, c, axis=1)
        boxes_c = tf.boolean_mask(boxes, mask_c)
        scores_c = tf.boolean_mask(scores_c, mask_c)
        nms_idx = tf.image.non_max_suppression(
            boxes_c, scores_c, max_boxes_tensor, iou_threshold=iou_threshold
        )
        boxes_c = tf.gather(boxes_c, nms_idx)
        scores_c = tf.gather(scores_c, nms_idx)
        classes_c = tf.fill(tf.shape(scores_c), c)
        return boxes_c, scores_c, classes_c

    nms_layer = Lambda(
        nms_per_class,
        output_shape=((None, 4), (None,), (None,))  # boxes, scores, classes
    )
    # =====================================

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        result = nms_layer([boxes, box_scores, mask, tf.constant(c, dtype=tf.int32)])
        boxes_.append(result[0])
        scores_.append(result[1])
        classes_.append(result[2])

    boxes_ = ops.concatenate(boxes_, axis=0)
    scores_ = ops.concatenate(scores_, axis=0)
    classes_ = ops.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_



def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    assert (
        true_boxes[..., 4] < num_classes
    ).all(), "class id must be less than num_classes"
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )

    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [
        np.zeros(
            (
                m,
                grid_shapes[l][0],
                grid_shapes[l][1],
                len(anchor_mask[l]),
                5 + num_classes,
            ),
            dtype="float32",
        )
        for l in range(num_layers)
    ]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.0
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.0
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(
                        "int32"
                    )
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(
                        "int32"
                    )
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype("int32")
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [
        K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0]))
        for l in range(num_layers)
    ]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(
            yolo_outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            calc_loss=True,
        )
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(
            y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]
        )
        raw_true_wh = K.switch(
            object_mask, raw_true_wh, K.zeros_like(raw_true_wh)
        )  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        # For TensorFlow 2.x, use a simpler approach with tf.map_fn
        object_mask_bool = K.cast(object_mask, "bool")
        
        # Get batch size from pred_box
        pred_shape = K.shape(pred_box)
        batch_size = pred_shape[0]
        
        def compute_ignore_mask(b):
            """Compute ignore mask for a single batch item"""
            # Get true boxes for this batch item
            true_box = tf.boolean_mask(
                y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0]
            )
            
            # Get the shape of pred_box for this batch item
            pred_box_b = pred_box[b]  # shape: (grid_h, grid_w, num_anchors, 4)
            pred_box_shape = K.shape(pred_box_b)
            gh = pred_box_shape[0]
            gw = pred_box_shape[1]
            na = pred_box_shape[2]
            
            # Check if there are any true boxes using TensorFlow operations
            num_true_boxes = tf.shape(true_box)[0]
            has_boxes = tf.greater(num_true_boxes, 0)
            
            # If no true boxes, return zeros; otherwise compute IoU
            def compute_with_boxes():
                # Reshape pred_box to (grid_h * grid_w * num_anchors, 4)
                pred_boxes_flat = K.reshape(pred_box_b, [-1, 4])
                
                # Compute IoU: (grid_h * grid_w * num_anchors, num_true_boxes)
                iou = box_iou(pred_boxes_flat, true_box)  # Returns (grid_h * grid_w * num_anchors, num_true_boxes)
                best_iou = K.max(iou, axis=-1)  # (grid_h * grid_w * num_anchors,)
                
                # Reshape back to (grid_h, grid_w, num_anchors)
                best_iou = K.reshape(best_iou, [gh, gw, na])
                
                # Return mask: 1 if should ignore (best_iou < threshold), 0 otherwise
                return K.cast(best_iou < ignore_thresh, K.dtype(y_true[0]))
            
            def return_zeros():
                return tf.zeros([gh, gw, na], dtype=K.dtype(y_true[0]))
            
            # Use tf.cond for conditional execution
            return tf.cond(has_boxes, compute_with_boxes, return_zeros)
        
        ignore_mask = tf.map_fn(
            compute_ignore_mask,
            tf.range(batch_size),
            fn_output_signature=tf.TensorSpec(
                shape=[None, None, None],  # [grid_h, grid_w, num_anchors]
                dtype=K.dtype(y_true[0])
            )
        )
        ignore_mask = K.expand_dims(ignore_mask, -1)  # Add last dimension: (batch, grid_h, grid_w, num_anchors, 1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = (
            object_mask
            * box_loss_scale
            * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        )
        wh_loss = (
            object_mask
            * box_loss_scale
            * 0.5
            * K.square(raw_true_wh - raw_pred[..., 2:4])
        )
        confidence_loss = (
            object_mask
            * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            + (1 - object_mask)
            * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            * ignore_mask
        )
        class_loss = object_mask * K.binary_crossentropy(
            true_class_probs, raw_pred[..., 5:], from_logits=True
        )

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            # Use tf.py_function to print in TensorFlow 2.x
            def print_loss_fn(loss_val, xy, wh, conf, cls, ign):
                print(f"loss: {loss_val:.4f} xy: {xy:.4f} wh: {wh:.4f} conf: {conf:.4f} cls: {cls:.4f} ignore: {ign:.4f}")
                return loss_val
            
            loss = tf.py_function(
                print_loss_fn,
                [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                tf.float32
            )
            loss.set_shape([])  # Ensure shape is preserved
    return loss
