import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch


def train_and_evaluate_svm(x, y):

    x = x.detach().cpu().numpy()
    x = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    y = np.array(y).ravel()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    n_classes = len(np.unique(y))
    is_binary = n_classes == 2

    svm_clf = svm.SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True if is_binary else False,
        random_state=42,
        decision_function_shape='ovr'
    )
    svm_clf.fit(x_train, y_train)

    y_pred = svm_clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    if is_binary:
        y_prob = svm_clf.predict_proba(x_test)[:, 1]
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    else:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")


def pca(features, labels, miss_rate, dataset_name):

    features = features.detach().cpu().numpy()
    features_np = features.numpy() if hasattr(features, 'numpy') else np.array(features)
    labels_np = np.array(labels).ravel()

    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(features_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=labels_np,
                         cmap='viridis', alpha=0.7, s=50)

    if dataset_name == 'ROSMAP':
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=['braak-0', 'braak-1'])
    else:
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=['Normal-like', 'Basal-like', 'HER2-enriched', 'Luminal_A', 'Luminal_B'])

    plt.title('Miss-Rate: {}, PCA Visualization of Samples (Explained Variance: {:.2f}%)'.format(miss_rate,
              sum(pca.explained_variance_ratio_) * 100))
    plt.xlabel('Principal Component 1 ({:.2f}%)'.format(
              pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('Principal Component 2 ({:.2f}%)'.format(
              pca.explained_variance_ratio_[1] * 100))
    plt.colorbar(scatter, label='Class')
    plt.grid(True)
    plt.show()


def evaluate_imputation(real_data, imputed_data):

    imputed_data = imputed_data.detach().cpu().numpy()
    real_data = real_data.cpu().numpy()
    real = np.array(real_data)
    imputed = np.array(imputed_data)

    metrics = {}

    # 1. 计算全局指标
    metrics['MAE'] = mean_absolute_error(real, imputed)
    metrics['MSE'] = mean_squared_error(real, imputed)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])

    sample_r2 = []
    for i in range(real.shape[0]):
        r2 = r2_score(real[i], imputed[i])
        sample_r2.append(r2)

    sample_r2 = np.array(sample_r2)

    metrics['Mean R²'] = np.mean(sample_r2)
    metrics['% R² > 0.3'] = np.mean(sample_r2 > 0.3) * 100

    # 打印结果
    print("RNA-seq imputation quality evaluation metrics:")
    print("=" * 40)
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Mean R²: {metrics['Mean R²']:.4f}")
    print(f"% R² > 0.3: {metrics['% R² > 0.3']:.1f}%")


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False


def calculate_r2_per_sample(pred, true, miss_rate):

    ss_res = torch.sum((true - pred) ** 2, dim=1)
    ss_tot = torch.sum((true - true.mean(dim=1, keepdim=True)) ** 2, dim=1)
    r2_scores = 1 - (ss_res / (ss_tot + 1e-8))

    max_idx = torch.argmax(r2_scores).item()
    best_pred = pred[max_idx]
    best_true = true[max_idx]

    plot_imputation_evaluation(best_true, best_pred, miss_rate, r2_scores[max_idx])

def plot_imputation_evaluation(real_data, imputed_data, miss_rate, r2, save_path=None):

    real_data = real_data.cpu().numpy()
    imputed_data = imputed_data.detach().cpu().numpy()

    plt.figure(figsize=(8, 8), dpi=100)

    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16
    })

    plt.scatter(real_data, imputed_data, alpha=0.6, s=50, c='purple')

    min_val = min(np.min(real_data), np.min(imputed_data))
    max_val = max(np.max(real_data), np.max(imputed_data))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='steelblue',
             alpha=0.7,
             linewidth=2)

    plt.title(f'Miss Rate：{miss_rate} [ $R^2 = {r2}$ ]', pad=20)

    plt.xlabel('Real Gene Expression Level', labelpad=10)
    plt.ylabel('Predicted Gene Expression Level', labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=14, pad=8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(False)

    plt.tight_layout()
    plt.show()
