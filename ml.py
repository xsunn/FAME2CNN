import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from radiomics import firstorder, shape2D, glcm, featureextractor
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from cardio.dataset import SplitType
from cardio.configuration import *

from cardio.dataset import Fame2RawDSLoader
from cardio.dataset.preprocess.graph import *
from cardio.dataset import Fame2GraphDatasetWrapper

def calculate_radiomix_features(extractor, radiomix_features, radiomix_categories, im, ma):
  ''' calculate features based on configuration  and fill radiomix_features dictionary '''

  featureVector = extractor.execute(im, ma)
  for featureName in featureVector.keys():

    category = None
    for c in radiomix_categories: 
      if c in featureName:
        category = c
    if not category: continue # skip unknown categories  

    feature_column_name = featureName[featureName.index(category):]
    radiomix_features[feature_column_name] = radiomix_features.get(feature_column_name, []) + [featureVector[featureName].item()]


FAME2_DUMP_DIR =  '/home/sun/data/FAME2labelling'
# FILEPATH_CLINICAL_EVENT_DF = '/home/sun/data/fame2_clinical_events_2year_data.csv'
EVENT_COLUMNS = ['VOCE', 'UR_TVF', 'NUR_TVF', 'MI_TVF', 'CV_TVF']
SPLIT_PATH = '/home/sun/data/fold1/'
CLINICAL_DATA_FEATURES = ["Age", "HTN", "Hchol", "DM_Overall", "Ren_Ins", "PVD", "CVA",
  "Prev_MI", "Prev_PCI", "Male", "CAD", "Smoker"
]

# for foldPath in foldPathList:

fame2_ds = Fame2RawDSLoader(
    FAME2_DUMP_DIR, 
    '/home/sun/data/fame2_clinical_events_n_patient_data_2year.csv', 
    only_lesion_data_points=True,
    df_ce_event_columns=EVENT_COLUMNS,
    generate_lesion_labels=True, 
    duplicate_imgs_with_multiple_lesions=True, 
    lazy_load=True,
    all_arteries_in_one_img=False)


# 
# Radiomix features
# 
radiomix_categories = ['glcm', 'glrlm', 'shape2D']
radiomix_shape2D_features = ['MeshSurface', 'Perimeter', 'MaximumDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation']

extractor = featureextractor.RadiomicsFeatureExtractor()

foldPathList=['/home/sun/data/fold1/','/home/sun/data/fold2/','/home/sun/data/fold3/','/home/sun/data/fold4/','/home/sun/data/fold5/']
for foldPath in foldPathList:
    print(foldPath)
    fame2_ds.setup(foldPath, SplitType.train)

    # Reconfigure readiomix feature extractor, based on correlation with train event
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(
        shape2D=['MeshSurface', 'Perimeter', 'MaximumDiameter', 'MajorAxisLength', 'MinorAxisLength'], 
        glcm=['Imc2'],
        glrlm=['GrayLevelNonUniformity', 'RunEntropy', 'RunLengthNonUniformity']
        )

    X_train_columns = [""] * 9
    X_train = np.zeros((len(fame2_ds), 10), dtype=np.float32)
    y_train = np.zeros((len(fame2_ds), 1), dtype=np.int32)
    ffr_train = np.zeros((len(fame2_ds), 1), dtype=np.float32)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Create Train 
    for i, item in tqdm(enumerate(fame2_ds), total=len(X_train), desc='Train'):
        label = item[1]
        # print(len(label[label != 0].unique()))
    #     assert len(label[label != 0].unique()) == 2
        label[label == 1] = 0 
        label[label == 2] = 1

        im = sitk.GetImageFromArray(item[0][0,:].numpy())
        ma = sitk.GetImageFromArray(label.numpy())

        radiomix_features = {}
        calculate_radiomix_features(extractor, radiomix_features, radiomix_categories, im, ma)

        for j, k in enumerate(sorted(radiomix_features)):
            X_train_columns[j] = k
            X_train[i,j] = radiomix_features[k][0]
        X_train[i, 9]= item[2].FFR.values[0]
        y_train[i]= item[2].Event.values[0]
        

    # Create Test
    fame2_ds.setup(foldPath, SplitType.test)
    X_test = np.zeros((len(fame2_ds), 10), dtype=np.float32)
    y_test = np.zeros((len(fame2_ds), 1), dtype=np.int32)

    for i, item in tqdm(enumerate(fame2_ds), total=len(X_test), desc='Test'):
        label = item[1]
        # assert len(label[label != 0].unique()) == 2
        label[label == 1] = 0 
        label[label == 2] = 1

        im = sitk.GetImageFromArray(item[0][0,:].numpy())
        ma = sitk.GetImageFromArray(label.numpy())

        radiomix_features = {}
        calculate_radiomix_features(extractor, radiomix_features, radiomix_categories, im, ma)

        for j, k in enumerate(sorted(radiomix_features)):
            X_test[i,j] = radiomix_features[k][0]
        X_test[i, 9]= item[2].FFR.values[0]
        y_test[i]= item[2].Event.values[0]


    # bst = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.9, objective='binary:logistic')
    # bst = svm.SVC(kernel='linear')
    bst = DecisionTreeClassifier()
    # bst = LogisticRegression()
    # bst.fit(X_train, y_train)
    # preds_test = bst.predict(X_test)   

    bst.fit(X_train[:,:9], y_train)
    preds_test = bst.predict(X_test[:,:9])
    def specificity(y_true, preds):
        tn, fp, fn, tp = confusion_matrix(y_true, preds > 0.5).ravel()
        return tn/(fp+tn) 

    test_metrics = [ accuracy_score, recall_score, precision_score, f1_score,specificity, roc_auc_score ]
    metrics_column = {f"{m.__name__}":list() for m in test_metrics }

    for m in test_metrics:
        metrics_column[f"{m.__name__}"].append(m(y_test, preds_test > 0.5 if m.__name__ != 'roc_auc_score' else preds_test).item())  

    from collections import Counter
    print("Test label distribution", Counter(y_test.reshape(-1).tolist()))

    pd.DataFrame(metrics_column) 
    print(metrics_column)