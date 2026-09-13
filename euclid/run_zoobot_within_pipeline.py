from typing import List, Dict
import os

import numpy as np
import pandas as pd
from skimage.transform import resize
import onnxruntime


# if os.path.isdir('/home/walml') or os.path.isdir('/Users/user'):  # local import
#     import morphology_utils  # local import with custom path, for isolated testing/debugging
# else:  # normal eden import
#     from . import morphology_utils

import morphology_utils

# import ElementsKernel.Logging as log
# logger = log.getLogger('MER_Zoobot')

# Setting multithreading CPU count
if 'PIPELINE_CPU_CORES' in os.environ:
    # This is a IAL pipeline run
    THREAD_NUMBER = int(os.environ['PIPELINE_CPU_CORES'])
else:
    # This is NOT a IAL pipeline run
    THREAD_NUMBER = 4

# Euclid non-ortho schema (but manually written as zoobot package itself is not in EDEN)
ZOOBOT_OUTPUT_COLUMNS = ['SMOOTH_OR_FEATURED_SMOOTH',
                         'SMOOTH_OR_FEATURED_FEATURED_OR_DISK',
                         'SMOOTH_OR_FEATURED_PROBLEM',
                         'DISK_EDGE_ON_YES',
                         'DISK_EDGE_ON_NO',
                         'HAS_SPIRAL_ARMS_YES',
                         'HAS_SPIRAL_ARMS_NO',
                         'BAR_STRONG',
                         'BAR_WEAK',
                         'BAR_NO',
                         'BULGE_SIZE_DOMINANT',
                         'BULGE_SIZE_LARGE',
                         'BULGE_SIZE_MODERATE',
                         'BULGE_SIZE_SMALL',
                         'BULGE_SIZE_NONE',
                         'HOW_ROUNDED_ROUND',
                         'HOW_ROUNDED_IN_BETWEEN',
                         'HOW_ROUNDED_CIGAR_SHAPED',
                         'EDGE_ON_BULGE_BOXY',
                         'EDGE_ON_BULGE_NONE',
                         'EDGE_ON_BULGE_ROUNDED',
                         'SPIRAL_WINDING_TIGHT',
                         'SPIRAL_WINDING_MEDIUM',
                         'SPIRAL_WINDING_LOOSE',
                         'SPIRAL_ARM_COUNT_1',
                         'SPIRAL_ARM_COUNT_2',
                         'SPIRAL_ARM_COUNT_3',
                         'SPIRAL_ARM_COUNT_4',
                         'SPIRAL_ARM_COUNT_MORE_THAN_4',
                         'SPIRAL_ARM_COUNT_CANT_TELL',
                         'MERGING_NONE',
                         'MERGING_MINOR_DISTURBANCE',
                         'MERGING_MAJOR_DISTURBANCE',
                         'MERGING_MERGER',
                         'CLUMPS_YES',
                         'CLUMPS_NO',
                         'PROBLEM_STAR',
                         'PROBLEM_ARTIFACT',
                         'PROBLEM_ZOOM',
                         'ARTIFACT_SATELLITE',
                         'ARTIFACT_SCATTERED',
                         'ARTIFACT_DIFFRACTION',
                         'ARTIFACT_RAY',
                         'ARTIFACT_SATURATION',
                         'ARTIFACT_OTHER',
                         'ARTIFACT_GHOST']

def zoobot_predictions_to_morphology_table(predictions: List[Dict]) -> pd.DataFrame:

    predictions = pd.DataFrame(data=predictions)
    # some output columns are not in the schema and so will be dropped:
    # "problem" columns, clumps yes/no
    # however for now I'm passing them out in the same table, if maybe we will extend the schema later
    # https://docs.google.com/spreadsheets/d/1UrIsgenUFmrxeFn4w-j818O2iCkFyiN0nm6wlR-B50c/edit?usp=sharing
    predictions = predictions[ZOOBOT_OUTPUT_COLUMNS + ['SOURCE_ID']]
    return predictions
    

"""
Assumes we have a large image (mosaic) in-memory

Assumes we have a catalog listing the discrete sources in the segmap, with columns:
- x0, x1, y0, y3 denoting the corner pixels of each source, anti-clockwise from bottom right
(this detail can be changed)
- 
(also have code where the catalog lists WCS coordinates instead)

"""

#####################################modified functions###############################################


def measure_zoobot_morphology_on_table(sources: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    For a given mosaic:
    - Select the segmented sources relevant to Zoobot (i.e. reasonably extended)
    - Create square cutouts of fixed pixel size, using the segmap corners recorded in the catalog
    - Group these cutouts into batches
    - Apply Zoobot to measure the morphology of the galaxies in each batch

    Args:
        sources (pd.DataFrame): table with id of sources and the location of their cutouts

    Returns:
        pd.DataFrame: with columns SOURCE_ID (identifying each segmented source) and Zoobot outputs (e.g. smooth, featured, etc.)
    """

    # zoobot config example

    # config = {
    #     'model_path': path/some_model.onnx
    #     'resized_cutout_size': 224,  # changed from 300
    #     'buff': 10
    # }
    """ 
    model_path (str): path to saved zoobot model
    resized_cutout_size (int, optional): Resample the (square) cutouts to this size before making predictions. Likely 224.
    """

    if config['model_path'].lower() == 'debug':
        # TODO add logging
        model = DummyZoobotModel()
    else:
        model = ONNXModel(config['model_path'])  # will need significant memory

    # filter sources to zoobot-relevant only
    # this would be more efficient with a dataframe, but it works on dicts
    print(f'Source count (no selection function): {len(sources)}')
    # sources = [s for s in sources if morphology_utils.passes_module_zoobot_selection_cuts(s)]
    # print(f'Zoobot-relevant source count: {len(sources)}')
    # exit()
    predictions = [measure_morphology_on_presaved_cutout(config, model, source) for source in sources]

    # some elements will be None, drop them
    predictions = [p for p in predictions if p is not None]

    if predictions:
        predictions = zoobot_predictions_to_morphology_table(predictions)
    else:
        predictions = None

    return predictions

def measure_morphology_on_presaved_cutout(config, model, source):
        try:
            # cutout = prepare_cutout(mosaic, source, config)  # crop, adjust dynamic range (0-1 float)
            cutout = load_and_resize_cutout(source, config)
        except Exception as e:
            print(f'Error preparing cutout for source {source["SOURCE_ID"]}: {e}')
            return None
        prediction_array = model.predict(np.expand_dims(cutout, 0))  # add batch dimension
        prediction_dict = dict(zip(ZOOBOT_OUTPUT_COLUMNS, prediction_array))
        prediction_dict.update({'SOURCE_ID': source['SOURCE_ID']})
        return prediction_dict

# def load_and_resize_cutout(source, config):
    


#########################################################################

def measure_zoobot_morphology_on_mosaic(mosaic: np.ndarray, sources: List[Dict], config: Dict) -> pd.DataFrame:
    """
    For a given mosaic:
    - Select the segmented sources relevant to Zoobot (i.e. reasonably extended)
    - Create square cutouts of fixed pixel size, using the segmap corners recorded in the catalog
    - Group these cutouts into batches
    - Apply Zoobot to measure the morphology of the galaxies in each batch

    Args:
        mosaic (np.ndarray): M x M large image from which cutouts will be extracted
        sources (List(Dict)): of sources, each with keys X_CENTER, Y_CENTER, and R_MAX (all in pixel coordinates)

    Returns:
        pd.DataFrame: with columns SOURCE_ID (identifying each segmented source) and Zoobot outputs (e.g. smooth, featured, etc.)
    """

    # zoobot config example

    # config = {
    #     'model_path': path/some_model.onnx
    #     'resized_cutout_size': 224,  # changed from 300
    #     'buff': 10
    # }
    """ 
    model_path (str): path to saved zoobot model
    resized_cutout_size (int, optional): Resample the (square) cutouts to this size before making predictions. Likely 224.
    """

    if config['model_path'].lower() == 'debug':
        # TODO add logging
        model = DummyZoobotModel()
    else:
        model = ONNXModel(config['model_path'])  # will need significant memory

    # filter sources to zoobot-relevant only
    # this would be more efficient with a dataframe, but it works on dicts
    print(f'Initial source count: {len(sources)}')
    # sources = [s for s in sources if morphology_utils.passes_module_zoobot_selection_cuts(s)]
    print(f'Zoobot-relevant source count: {len(sources)}')
    # exit()
    predictions = [measure_morphology(mosaic, config, model, source) for source in sources]

    # some elements will be None, drop them
    predictions = [p for p in predictions if p is not None]

    if predictions:
        predictions = zoobot_predictions_to_morphology_table(predictions)
    else:
        predictions = None

    return predictions

def measure_morphology(mosaic, config, model, source):
        # print(source)
        try:
            cutout = prepare_cutout(mosaic, source, config)  # crop, adjust dynamic range (0-1 float)
        except Exception as e:
            print(f'Error preparing cutout for source {source["SOURCE_ID"]}: {e}')
            return None
        prediction_array = model.predict(np.expand_dims(cutout, 0))  # add batch dimension
        prediction_dict = dict(zip(ZOOBOT_OUTPUT_COLUMNS, prediction_array))
        prediction_dict.update({'SOURCE_ID': source['SOURCE_ID']})
        return prediction_dict


def prepare_cutout(mosaic: np.ndarray, source: Dict, config, radius_estimate = False) -> np.ndarray:
    """
    Prepares cutout image of single galaxy, either by using the segmentation map or the major axis

    Args:
        mosaic (np.ndarray): M x M large image from which the cutout will be extracted
        source (dict): The catalog containing the parameters of the sources (crucially including the segmap corners)
        buff (int): see extract_cutout
        resized_cutout_size (int): Resample the (square) cutout to this size.

    Returns:
        np.ndarray: R x R x 1 small cutout image of single galaxy, resampled from native pixels to RxR
    """
    cutout = morphology_utils.make_vis_only_cutout_from_tiles(source, mosaic, radius_estimate)

    # any further ML-friendly transforms
    # resize to desired output size
    # during training we use torchvision but I'm not sure that's available in EDEN so using skimage here

    cutout = resize(
        cutout,
        output_shape=(
            config['resized_cutout_size'], config['resized_cutout_size']
        ),
        order=3,  # bicubic interpolation
        anti_aliasing=True  # true by default in torchvision
    )

    # convert to torch shape (1CHW)
    cutout = np.expand_dims(cutout, axis=0)
    return cutout.astype(np.float32)



class DummyZoobotModel():

    def __init__(self) -> None:
        pass        

    def predict(self, x):
        assert len(x.shape) == 4  # (1, H, W, C)
        assert x.shape[0] == 1
        return (np.random.rand(len(ZOOBOT_OUTPUT_COLUMNS)) + 1).squeeze()
    
class ONNXModel():

    def __init__(self, model_loc: str) -> None:        

        ## Forcing ONNX to use one thread only as per PPO requests
        ort_options = onnxruntime.SessionOptions()
        ort_options.intra_op_num_threads = THREAD_NUMBER
        ort_options.inter_op_num_threads = THREAD_NUMBER

        self.ort_session = onnxruntime.InferenceSession(model_loc, sess_options = ort_options)

    def predict(self, x: np.array, validate=False) -> np.array:
        if validate:
            assert x.min() > 0
            assert x.max() <= 1
            # onnx will check shapes and dtypes match

        outputs = self.ort_session.run(
            None,
            {"input.1": x},
        )
        # outputs is a list, because model can in general return several arrays
        return outputs[0].squeeze()  # the actual predictions, shape (n_answers)
