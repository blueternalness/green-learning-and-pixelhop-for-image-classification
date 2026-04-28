"""
(1) Wooyoung Kim
(2) 2717106812
(3) wkim9450@usc.edu
(4) 04/28 2026
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import gzip
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.util import view_as_windows
from skimage.measure import block_reduce

import warnings
warnings.filterwarnings("ignore")
from cwSaab import cwSaab

# manually download dataset
def download_and_extract(url, is_images, prefix):
    local_filename = prefix + "_" + url.split('/')[-1]
    if not os.path.exists(local_filename):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(local_filename, 'wb') as out_file:
            out_file.write(response.read())
            
    with gzip.open(local_filename, 'rb') as f:
        if is_images:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            return np.frombuffer(f.read(), np.uint8, offset=8)

def load_and_preprocess_data(dataset_name="mnist", subset_size=10000):
    if dataset_name == "mnist":
        base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        prefix = "mnist"
    else:
        base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
        prefix = "fashion"
        
    x_train = download_and_extract(base + "train-images-idx3-ubyte.gz", True, prefix)
    y_train = download_and_extract(base + "train-labels-idx1-ubyte.gz", False, prefix)
    x_test = download_and_extract(base + "t10k-images-idx3-ubyte.gz", True, prefix)
    y_test = download_and_extract(base + "t10k-labels-idx1-ubyte.gz", False, prefix)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), 'constant')

    if subset_size is not None:
        samples_per_class = subset_size // 10
        indices = []
        for c in range(10):
            c_idx = np.where(y_train == c)[0][:samples_per_class]
            indices.extend(c_idx)
        x_train = x_train[indices]
        y_train = y_train[indices]
    return x_train, y_train, x_test, y_test

def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pool = shrinkArg.get('pool', False)
    
    if pool:
        X = block_reduce(X, block_size=(1, 2, 2, 1), func=np.max)
    
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X

def get_pixelhop_model(use_cw, TH1, TH2):
    shrinkArgs = [
        {'func': Shrink, 'win': 5, 'stride': 1, 'pool': False},
        {'func': Shrink, 'win': 5, 'stride': 1, 'pool': True},
        {'func': Shrink, 'win': 5, 'stride': 1, 'pool': True}
    ]
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useAW': False, 'cw': use_cw}] * 3
    model = cwSaab(depth=3, TH1=TH1, TH2=TH2, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
    return model

def calculate_num_parameters(features, use_cw, window_size=5):
    total_params = 0
    input_channels = 1
    for feat in features:
        output_channels = feat.shape[-1]
        if use_cw:
            hop_params = output_channels * (window_size * window_size * 1)
        else:
            hop_params = output_channels * (window_size * window_size * input_channels)
            input_channels = output_channels 
        total_params += hop_params
        
    return total_params

def p_2a(dataset_name="mnist"):
    x_train, y_train, x_test, y_test = load_and_preprocess_data(dataset_name, subset_size=10000)
    
    TH1, TH2 = 0.005, 0.001
    start_time = time.time()
    
    ph_plus_plus = get_pixelhop_model(use_cw=True, TH1=TH1, TH2=TH2)
    ph_plus_plus.fit(x_train)
    
    train_features = ph_plus_plus.transform(x_train)
    hop3_train = train_features[-1].reshape(x_train.shape[0], -1) 
    
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1)
    clf.fit(hop3_train, y_train)
    
    train_time = time.time() - start_time
    train_acc = accuracy_score(y_train, clf.predict(hop3_train))
    model_size = calculate_num_parameters(train_features, use_cw=True)
    
    print(f"training time: {train_time}")
    print(f"train ACC: {train_acc * 100}%")
    print(f"model size: {model_size}")
    
    test_features = ph_plus_plus.transform(x_test)
    hop3_test = test_features[-1].reshape(x_test.shape[0], -1)
    test_acc = accuracy_score(y_test, clf.predict(hop3_test))
    print(f"test accuracy: {test_acc * 100}%")
    
    th1_values = [0.001, 0.003, 0.005, 0.008, 0.01]
    th1_accuracies = []
    
    for th in th1_values:
        ph_temp = get_pixelhop_model(use_cw=True, TH1=th, TH2=TH2)
        ph_temp.fit(x_train)
        
        feat_train_list = ph_temp.transform(x_train)
        feat_train = feat_train_list[-1].reshape(x_train.shape[0], -1)
        feat_test = ph_temp.transform(x_test)[-1].reshape(x_test.shape[0], -1)
        
        clf_temp = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1)
        clf_temp.fit(feat_train, y_train)
        
        acc = accuracy_score(y_test, clf_temp.predict(feat_test))
        th1_accuracies.append(acc)
        
        sz = calculate_num_parameters(feat_train_list, use_cw=True)
        print(f"  TH1: {th}, test Acc: {acc*100}%, size: {sz}")

    plt.figure(figsize=(8, 5))
    plt.plot(th1_values, th1_accuracies, marker='o', linestyle='-')
    plt.title(f'TH1 vs Test Accuracy (PixelHop++ {dataset_name.upper()})')
    plt.xlabel('TH1 Energy Threshold')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    
    plot_filename = f"{dataset_name}_task_a_th1_curve.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

def p_2b(dataset_name="mnist"):
    x_train, y_train, x_test, y_test = load_and_preprocess_data(dataset_name, subset_size=10000)
    
    results = {}
    for model_type, use_cw in [("PixelHop", False), ("PixelHop++", True)]:
        start_time = time.time()
        
        model = get_pixelhop_model(use_cw=use_cw, TH1=0.005, TH2=0.001)
        model.fit(x_train)
        
        train_features = model.transform(x_train)
        hop3_train = train_features[-1].reshape(x_train.shape[0], -1)
        
        test_features = model.transform(x_test)
        hop3_test = test_features[-1].reshape(x_test.shape[0], -1)
        
        clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1)
        clf.fit(hop3_train, y_train)
        
        runtime = time.time() - start_time
        train_acc = accuracy_score(y_train, clf.predict(hop3_train))
        test_acc = accuracy_score(y_test, clf.predict(hop3_test))
        
        params = calculate_num_parameters(train_features, use_cw=use_cw)
            
        results[model_type] = {
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Runtime": runtime,
            "Params": params
        }
        
    for m, res in results.items():
        print(f"{m}:\n  train ACC: {res['Train Acc']*100}%\n  test ACC: {res['Test Acc']*100}%")
        print(f"\n runtime: {res['Runtime']}s\n  Params: {res['Params']}\n")


def p_2c(dataset_name="fashion_mnist"):
    x_train, y_train, x_test, y_test = load_and_preprocess_data(dataset_name, subset_size=None)
    
    ph_plus_plus = get_pixelhop_model(use_cw=True, TH1=0.005, TH2=0.001)
    ph_plus_plus.fit(x_train)
    
    hop3_train = ph_plus_plus.transform(x_train)[-1].reshape(x_train.shape[0], -1)
    hop3_test = ph_plus_plus.transform(x_test)[-1].reshape(x_test.shape[0], -1)
    
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1)
    clf.fit(hop3_train, y_train)
    
    test_preds = clf.predict(hop3_test)
    
    cm = confusion_matrix(y_test, test_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - PixelHop++ ({dataset_name.upper()})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plot_filename = f"{dataset_name}_task_c_confusion_matrix.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    class_totals = cm.sum(axis=1)
    class_errors = class_totals - np.diag(cm)
    error_rates = class_errors / class_totals
    
    lowest_err_class = np.argmin(error_rates)
    highest_err_class = np.argmax(error_rates)
    
    print(f"lowest error class: {lowest_err_class} (error rate: {error_rates[lowest_err_class]})")
    print(f"highest error class: {highest_err_class} (error rate: {error_rates[highest_err_class]})")

if __name__ == "__main__":
    dataset_choice = "mnist" # or fashion-mnist
    
    print("problem 2-a")
    p_2a(dataset_choice)
    print("problem 2-b")
    p_2b(dataset_choice)
    print("problem 2-c")
    p_2c(dataset_choice)