import sys
import os
import numpy as np
import datetime
import subprocess
from distutils import dir_util
import _sys_init

from lib.functions.anchor_projector import project_to_image_space

def get_kitti_predictions(score_threshold, globl_epoch=-1):
    # Get available prediction folders
    root_dir = _sys_init.root_dir()
    predictions_root_dir = os.path.join(root_dir, 'experiments', 'predictions')

    # 3D prediction directories
    kitti_predictions_3d_dir = predictions_root_dir + \
        '/kitti_native_eval/' + \
        str(score_threshold) + '/' + \
        str(globl_epoch) + '/data'
    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)
    return kitti_predictions_3d_dir

def save_predictions_in_kitti_format(dataset, all_predictions, img_id, kitti_predictions_3d_dir, score_threshold=0.1):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.       
        all_predictions # n*(x,y,z,l,w,h,ry,scores)
    """

    # Do conversion
    num_samples = dataset.num

    prediction_file = '%06d.txt'%(img_id)

    kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
        '/' + prediction_file

    # # Swap l, w for predictions where w > l
    # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
    # fixed_predictions = np.copy(all_predictions)
    # fixed_predictions[swapped_indices, 3] = all_predictions[
    #     swapped_indices, 4]
    # fixed_predictions[swapped_indices, 4] = all_predictions[
    #     swapped_indices, 3]

    score_filter = all_predictions[:, 7] >= score_threshold
    all_predictions = all_predictions[score_filter]

    # If no predictions, skip to next file
    if len(all_predictions) == 0:
        np.savetxt(kitti_predictions_3d_file_path, [])
        return 0, num_samples

    # Load image for truncation
    image = dataset.kitti.get_image(img_id)

    calib = dataset.kitti.get_calibration(img_id)

    boxes = []
    image_filter = []
    for i in range(len(all_predictions)):
        box_3d = all_predictions[i, 0:7]
        img_box, _ = project_to_image_space(box_3d, calib.P, image.shape)

        # Skip invalid boxes (outside image space)
        if img_box is None:
            image_filter.append(False)
            continue

        image_filter.append(True)
        boxes.append(img_box)

    boxes = np.asarray(boxes)
    all_predictions = all_predictions[image_filter]

    # If no predictions, skip to next file
    if len(boxes) == 0:
        np.savetxt(kitti_predictions_3d_file_path, [])
        return 0, num_samples

    # To keep each value in its appropriate position, an array of zeros
    # (N, 16) is allocated but only values [4:16] are used
    kitti_predictions = np.zeros([len(boxes), 16])

    # Get object types
    all_pred_classes = np.ones(len(boxes)).astype(np.int32)
    obj_types = [dataset.id2names[class_idx]
                 for class_idx in all_pred_classes]

    # Truncation and Occlusion are always empty (see below)

    # Alpha (Not computed)
    kitti_predictions[:, 3] = -10 * np.ones((len(kitti_predictions)),
                                            dtype=np.int32)

    # 2D predictions
    kitti_predictions[:, 4:8] = boxes[:, 0:4]

    # 3D predictions
    # (l, w, h)
    kitti_predictions[:, 8] = all_predictions[:, 5]
    kitti_predictions[:, 9] = all_predictions[:, 4]
    kitti_predictions[:, 10] = all_predictions[:, 3]
    # (x, y, z)
    kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
    # (ry, score)
    kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

    # Round detections to 3 decimal places
    kitti_predictions = np.round(kitti_predictions, 3)

    # Empty Truncation, Occlusion
    kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                 dtype=np.int32)

    # Stack 3D predictions text
    kitti_text_3d = np.column_stack([obj_types,
                                     kitti_empty_1,
                                     kitti_predictions[:, 3:16]])

    # Save to text files
    np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
               newline='\r\n', fmt='%s')

    return 1, num_samples

def copy_kitti_native_code(kitti_native_code_copy):
    """Copies and compiles kitti native code.

    It also creates neccessary directories for storing the results
    of the kitti native evaluation code.
    """

    # Only copy if the code has not been already copied over
    if not os.path.exists(kitti_native_code_copy+'/run_make.sh'):

        os.makedirs(kitti_native_code_copy)
        original_kitti_native_code = _sys_init.root_dir() + \
            '/lib/evaluator/offline_eval/kitti_native_eval/'

        # create dir for it first
        dir_util.copy_tree(original_kitti_native_code,
                           kitti_native_code_copy)
        # run the script to compile the c++ code
        script_folder = kitti_native_code_copy
        make_script = script_folder + 'run_make.sh'
        subprocess.call([make_script, script_folder])

    # Set up the results folders if they don't exist
    results_dir = _sys_init.root_dir() + '/experiments/results'
    results_05_dir = _sys_init.root_dir() + '/experiments/results_05_iou'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(results_05_dir):
        os.makedirs(results_05_dir)


def run_kitti_native_script(kitti_native_code_copy, checkpoint_name, score_threshold, global_epoch):
    """Runs the kitti native code script."""

    make_script = kitti_native_code_copy + '/run_eval.sh'
    script_folder = kitti_native_code_copy

    results_dir = _sys_init.root_dir() + '/experiments/results/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_epoch),
                     str(checkpoint_name),
                     str(results_dir)])


def run_kitti_native_script_with_05_iou(kitti_native_code_copy, checkpoint_name, score_threshold,
                                        global_step):
    """Runs the kitti native code script."""

    make_script = kitti_native_code_copy + '/run_eval_05_iou.sh'
    script_folder = kitti_native_code_copy

    results_dir = _sys_init.root_dir() + '/experiments/results/'

    results_dir = _sys_init.root_dir() + '/experiments/results_05_iou/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])