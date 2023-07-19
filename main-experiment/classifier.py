from lightgbm import LGBMClassifier, Booster
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import json
import numpy as np
import torch

def construct_data():
    model_name = ["gpt2", "opt", "unilm", "llama", "bart", "t5", "bloom", "neo", "vicuna", "gpt2_large", "opt_3b"]
    # model_name = ["gpt2", "opt", "unilm"]
    labels_to_number = {"human": 0, "gpt2": 1, "opt": 2, "unilm": 3, "llama": 4, "bart": 5, "t5": 6, "bloom": 7,"neo": 8, "vicuna":9}
    # labels_to_number = {"human": 0, "gpt2": 1, "opt": 2, "unilm": 3}
    perplexity = []
    for model in model_name:
        perplexity_file = f'result/{model}.json'
        model_perplexity = []
        label = []
        with open(perplexity_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    for i, j in data.items():
                        if i in labels_to_number.keys():
                            model_perplexity.append(j)
                            label.append(labels_to_number[i])
                except Exception as e:
                    print(e, line)
        perplexity.append(model_perplexity)
    return perplexity, label

def LightGBM_Classification(X, y, n):
    # Load train dataset
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12343)

    model = LGBMClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        objective='multiclass',
        num_class=n,
        booster='gbtree',
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        seed=0)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100, early_stopping_rounds=50)

    y_pred = model.predict(X_test)
    # y_pred = [np.argmax(i) for i in y_pred]
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_test, y_pred, labels=[i for i in range(10)])
    print(p_class, r_class)
    print(f_class)
    # Save model
    model.booster_.save_model("nine_LightGBM_model.txt")

    return X_test,  y_test

if __name__ == "__main__":
    perplexity, label = construct_data()
    number = len(perplexity)
    lenght = len(label)
    perplexity = [[perplexity[i][j] for i in range(number)] for j in range(lenght)]


    # Train classification model and return test data
    X_test, y_test = LightGBM_Classification(perplexity, label, 10)
    for i in range(len(y_test)):
        if y_test[i] == 9:
            y_test[i] = 4

    # Load trained model
    model = Booster(model_file='nine_LightGBM_model.txt')
    # Make predictions on the test setMake predictions on the test set
    y_pred = model.predict(X_test)
    sum_num = len(y_test)
    R2 = 0
    for i in range(sum_num):
        values, indics = torch.topk(torch.tensor(y_pred[i]), 3, -1)
        if y_test[i] in indics:
            R2 += 1
    R2 = R2/sum_num
    y_pred = [np.argmax(i) for i in y_pred]
    for i in range(len(y_pred)):
        if y_pred[i] == 9:
            y_pred[i] = 4
    # Calculate accuracy, recall, f1 value
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_test, y_pred, labels=[i for i in range(9)])
    print(p_class, r_class)
    print(f_class)
    print(R2)

    np.savez("matrix.npz", y_test=np.array(y_test), y_pred=np.array(y_pred))
