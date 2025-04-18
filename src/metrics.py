import numpy as np
# import torch
# print(torch.__version__)

def get_metrics(real_score, predict_score):
    # Flatten both real_score and predict_score to ensure they are one-dimensional
    real_score, predict_score = real_score.flatten(), predict_score.flatten()

    # Get all unique values in predict_score and sort them
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)  # Number of unique values in sorted predict_score

    # Generate threshold values based on sorted unique values (used for binary classification thresholds)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)  # Convert thresholds into a matrix for easier computation
    thresholds_num = thresholds.shape[1]  # Number of thresholds

    # Repeat the predict_score to create a matrix for comparison with thresholds
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))

    # Find indices where predicted scores are below or above the threshold
    negative_index = np.where(predict_score_matrix < thresholds.T)  # Indices for negative samples
    positive_index = np.where(predict_score_matrix >= thresholds.T)  # Indices for positive samples

    # Assign binary values based on the threshold (0 for negative, 1 for positive)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
    TP = predict_score_matrix.dot(real_score.T)  # Multiply the predicted binary matrix by real score to get TP
    FP = predict_score_matrix.sum(axis=1) - TP  # FP is the total number of positive predictions minus TP
    FN = real_score.sum() - TP  # FN is the total number of real positives minus TP
    TN = len(real_score.T) - TP - FP - FN  # TN is the total number of samples minus TP, FP, and FN

    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr = FP / (FP + TN)  # False Positive Rate
    tpr = TP / (TP + FN)  # True Positive Rate (recall)

    # Generate ROC curve points
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]  # Add (0,0) to the ROC curve
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]  # Add (1,1) to the ROC curve

    # Compute AUC (Area Under Curve) from the ROC curve
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])  # Calculate the AUC

    # Calculate Precision and Recall for the Precision-Recall curve
    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]  # Add (0,1) to the PR curve
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]  # Add (1,0) to the PR curve

    # Compute AUPR (Area Under Precision-Recall curve)
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])  # Calculate the AUPR

    # Calculate additional metrics like F1 score, accuracy, specificity
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    # Find the index of the best F1 score and corresponding values
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    # Return the computed metrics along with the column names for easy reference
    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']
