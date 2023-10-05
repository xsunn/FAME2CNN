import sys
sys.path.append('')

from cardio.dataset.preprocess import graph
from tqdm import tqdm
import cardio.resolvers as resolvers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cardio.dataset import SplitType
from cardio.dataset import Fame2RawDSLoader
from cardio.dataset.fame2_raw_data_loader import load_train_test_split
from cardio.utils.img import extract_lesion_patch
import os.path as op
import os

IMGS_FNAME_POSTFIX = 'imgs.pt'
LABELS_FNAME_POSTFIX = 'labels.pt'
FEATURES_FNAME_POSTFIX = 'features.pt'

def extract_lesion_wide(df_info, features, transforms):
    lesion_wide_features, names = [], []

    for i in range(len(features)):
        feature = features[i]
        transform = transforms[i]

        if transform != None:
            value = transform( getattr(df_info,feature))
            if isinstance(value, list):
                lesion_wide_features.extend(value)
                names.extend( [features[i]] * len(value) )
            else:
                lesion_wide_features.append(value)
        else:
            value = getattr(df_info,feature)
            if pd.isna(value):
                value = 0.0
            lesion_wide_features.append(value)
            names.append(features[i])

    return lesion_wide_features, names


def create_ds_split(df, is_train, features, feature_transforms, save_dir):
    # Find max size
    imgH, imgW = 512,512
    fitH, fitW = 200, 200

    # Init vectors
    ds_shape = (df.shape[0], 2, fitH, fitW)
    imgs = torch.zeros(ds_shape, dtype=torch.float32)
    labels = torch.zeros((df.shape[0], 1), dtype=torch.float32)
    lesion_wide_features = []
    lesion_wide_features_names = []

    # Loop through data
    for i, row in tqdm(enumerate(df.iterrows()), total=df.shape[0]):
        base_path = row[0][0]

        p, l = extract_lesion_patch(base_path, row[1].artery, row[1].syntaxID, imgH, imgW, fitH, fitW)
        l[l>1]=1
        imgs[i, 0, :,:] = p
        imgs[i, 1, :,:] = l
        labels[i, 0] = row[1].VOCE

        # extract lesion_wide feature
        feats, names = extract_lesion_wide(row[1], features, feature_transforms)
        lesion_wide_features.append(feats)
        if i == 0:
            lesion_wide_features_names = names

    lesion_wide_features = torch.tensor(lesion_wide_features)

    split_name = SplitType.train.name if is_train else SplitType.test.name
    print("imgsize",imgs.size(),lesion_wide_features.size())
    print("imgsavePath",op.join(save_dir, split_name, IMGS_FNAME_POSTFIX))
  
  
    torch.save(imgs, op.join(save_dir, split_name, IMGS_FNAME_POSTFIX))
    torch.save(labels, op.join(save_dir, split_name, LABELS_FNAME_POSTFIX))
    torch.save(lesion_wide_features, op.join(save_dir, split_name, FEATURES_FNAME_POSTFIX))
    # TODO

FAME2_DUMP_DIR =  '/home/sun/data/FAME2labelling'
FILEPATH_CLINICAL_EVENT_DF = '/home/sun/data/fame2_clinical_events_n_patient_data_2year.csv'


EVENT_COLUMNS = ['VOCE', 'UR_TVF', 'NUR_TVF', 'MI_TVF', 'CV_TVF']

SPLIT_PATH = "/home/sun/project/FAME2CNN/data_split/fold1"

fame2_ds = Fame2RawDSLoader(
    FAME2_DUMP_DIR,
    FILEPATH_CLINICAL_EVENT_DF,
    only_lesion_data_points=True,
    generate_data_with_2_views=False,
    duplicate_imgs_with_multiple_lesions=True,
    df_ce_event_columns=EVENT_COLUMNS,
    generate_lesion_labels=True,
    lazy_load=True,
    all_arteries_in_one_img=False)

save_dir = "/home/sun/data/cnn/patch"

features = ["DS", "FFR", "Lesion_Length",
            "Lesion_Type", "artery", "exactSegmentLocation"]
# features=["Lesion_Length",
#             "Lesion_Type", "artery", "exactSegmentLocation"]
# CLINICAL_DATA_FEATURES = ["Age", "HTN", "Hchol", "DM_Overall", "Ren_Ins", "PVD", "CVA",
#   "Prev_MI", "Prev_PCI", "Male", "CAD", "Smoker"]

feature_transforms = [None, None, None,
    graph.lesion_type_to_1hot,
    graph.artery_to_1hot,
    graph.artery_segment_to_float
]

# Generate the dataset
fame2_ds.setup(SPLIT_PATH, SplitType.train)
index, data = zip(*fame2_ds.data)
df = pd.concat([d for d in data])
df.index = index
create_ds_split(df, True, features, feature_transforms, save_dir)

fame2_ds.setup(SPLIT_PATH, SplitType.test)
index, data = zip(*fame2_ds.data)
df = pd.concat([d for d in data])
df.index = index
create_ds_split(df, False, features, feature_transforms, save_dir)
