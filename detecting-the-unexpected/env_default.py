
import os
from pathlib import Path

DIR_HERE = Path(__file__).absolute().parent
DIR_ROOT = DIR_HERE

os.environ.update({
	# 'DIR_EXPERIMENTS': str(DIR_HERE.parents[1] / 'exp'),
	# 'DIR_LAF_SMALL': '/cvlabsrc1/cvlab/dataset_LostAndFound/1024x512_webp',
	# 'DIR_CITYSCAPES_SMALL': '/cvlabsrc1/cvlab/dataset_Cityscapes/1024x512',
	# 'DIR_ROAD_ANOMALY': '/cvlabsrc1/cvlab/dataset_RoadAnomaly',
    
    # where trained weights / logs live
    'DIR_EXPERIMENTS': str(DIR_ROOT / 'exp'),

    # where your compressed datasets live
    'DIR_LAF_SMALL': str(DIR_ROOT / 'datasets' / 'dataset_LostAndFound' / '1024x512'),
    'DIR_CITYSCAPES_SMALL': str(DIR_ROOT / 'datasets' / 'dataset_Cityscapes' / '1024x512'),
    'DIR_ROAD_ANOMALY': str(DIR_ROOT / 'datasets' / 'dataset_RoadAnomaly'),
})
