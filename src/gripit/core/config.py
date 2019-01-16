from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import collections
## TODO move this information to datastore config
SCENE_CONFIG = collections.OrderedDict()

SCENE_CONFIG["auto_canny_sigma_depth"] = {"label": "Depth Auto-Canny Sigma", "type": "INT_RANGE", "hidden":False, "min":1, "max":100, "value":33}
SCENE_CONFIG["auto_canny_sigma_curve"] = {"label": "Curve Auto-Canny Sigma", "type": "INT_RANGE", "hidden":False, "min":1, "max":100, "value":33}
SCENE_CONFIG["sgmnt_threshold"] = {"label": "Segmentation Threshold", "type": "INT", "hidden":True, "value":1}
SCENE_CONFIG["sgmnt_tolerance"] = {"label": "Segmentation Tolerance", "type":"INT_RANGE", "hidden":False, "max":50, "min":0, "value":10}
SCENE_CONFIG["min_sgmnt_len"] = {"label": "Minimum Segment Length", "type": "INT_RANGE", "hidden":True, "value":20, "min":1, "max":100}
SCENE_CONFIG["min_cntr_area"] = {"label": "Minimum Contour Area", "type": "INT", "hidden":True, "value":500}
SCENE_CONFIG["edge_pair_len_diff"] = {"label": "Minimum Pairing Length Ratio", "type": "INT_RANGE", "hidden":False, "min":1, "max":100, "value":50}
SCENE_CONFIG["edge_pair_min_dist"] = {"label": "Edge Pair Min Distance", "type": "INT_RANGE", "min":0, "max":100,"hidden":False, "value":1}
SCENE_CONFIG["edge_pair_max_dist"] = {"label": "Edge Pair Max Distance", "type": "INT_RANGE", "min":50, "max":1000, "hidden":False, "value":990}
SCENE_CONFIG["edge_pair_delta_angle"] = {"label": "Edge Pair Angle", "type": "INT_RANGE", "hidden":False, "min":0, "max":55, "value":30}
SCENE_CONFIG["classification_area"] = {"label": "Classification Window Size", "type": "INT", "hidden":True, "value":10}
SCENE_CONFIG["camera_focal_length"] = {"label": "Focal Length", "type": "INT", "hidden":True, "value":200}
SCENE_CONFIG["camera_pitch"] = {"label": "Camera Pitch", "type":"INT", "hidden":True, "value":44}
SCENE_CONFIG["camera_role"] = {"label": "Camera Role", "type":"INT", "hidden":True, "value":0}
SCENE_CONFIG["camera_yaw"] = {"label": "Camera Yaw", "type":"INT", "hidden":True, "value":0}
SCENE_CONFIG["camera_translate_x"] = {"label": "Camera Translation X", "type":"INT", "hidden":True, "value":0}
SCENE_CONFIG["camera_translate_y"] = {"label": "Camera Translation Y", "type":"INT", "hidden":True, "value":0} 
SCENE_CONFIG["camera_translate_z"] = {"label": "Camera Translation Z", "type":"INT", "hidden":True, "value":50}
SCENE_CONFIG["camera_location"] = {"label": "Camera Location", "type":"UI_GROUP", "hidden":False, "value":{
	"camera_location_x":{"label": "X", "type":"REAL", "hidden":False, "value":0},
	"camera_location_y":{"label": "Y", "type":"REAL", "hidden":False, "value":160.0593},
	"camera_location_z":{"label": "Z", "type":"REAL", "hidden":False, "value":142.9121}
}}
SCENE_CONFIG["camera_rotation"] = {"label": "Camera Orientation", "type":"UI_GROUP", "hidden":False, "value":{
	"camera_roftation_x":{"label": "X", "type":"REAL", "hidden":False, "value":48.6},
	"camera_rotation_y":{"label": "Y", "type":"REAL", "hidden":False, "value":0},
	"camera_rotation_z":{"label": "Z", "type":"REAL", "hidden":False, "value":46.7}
}}
SCENE_CONFIG["camera_sensor_size"] = {"label": "Sensor Size", "type":"UI_GROUP", "hidden":True, "value":{
	"camera_sensor_size_x":{"label": "Width", "type":"INT", "hidden":False, "value":32},
	"camera_sensor_size_y":{"label": "Width", "type":"INT", "hidden":False, "value":32},
	"camera_focal_length":{"label": "Focal Length", "type": "INT", "hidden":True, "value":200}
}}
# SCENE_CONFIG["parameter_comments"] = {"label": "Comments", "type":"UI_GROUP", "hidden":False, "value":{
# 	"real_edge_count":{"label": "Edge Count", "type":"INT", "hidden":False, "value":0},
# 	"processed_edge_count":{"label": "Processes Count", "type":"INT", "hidden":False, "value":0},
# 	"comment":{"label": "Comment", "type":"STRING", "hidden":False, "value":''}
# }}

