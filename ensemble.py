import pandas as pd
import numpy as np
from itertools import combinations

e2i = {
            "Against": 0,
            "nan": 1,
            "Favor": 2,
        }
i2e = {v: k for k, v in e2i.items()}

def get_max_from_dict(model1, model2, model3, f1dict):
    if f1dict[model1] > f1dict[model2] and f1dict[model1] > f1dict[model3]:
        return model1
    elif f1dict[model2] > f1dict[model1] and f1dict[model2] > f1dict[model3]:
        return model2
    else:
        return model3

def get_outputs(model1, model2, model3, f1dict, length=348):
    y_pred = []
    
    for i in range(length):
        temp = np.zeros((8),dtype = int)
        temp[e2i[str(model1[i][0])]]+=1
        temp[e2i[str(model2[i][0])]]+=1
        temp[e2i[str(model3[i][0])]]+=1
        if(np.max(temp) == 1):
            max_model = get_max_from_dict(model1, model2, model3, f1dict)
            y_pred.append(str(max_model[i][0]))
        else:
            cur = np.argmax(temp)
            y_pred.append(i2e[(cur)])
    return(y_pred)

def create_new_submission(models):
    possible_combinations = []
    possible_combinations = list(combinations(models, 3))
    counter = 1
    for combination in possible_combinations:
        y_pred = get_outputs(combination[0], combination[1], combination[2], f1_scores)
        finalarr = np.array(y_pred)
        submission = pd.DataFrame(finalarr, columns=["label"])
        submission.to_csv(f"combination_{counter}.csv", index=False)
        print(f"combination_{counter} created")
        counter += 1


arabert = np.array(pd.read_csv("/kaggle/input/best-stanceeval/arabert_test_final.csv"))
marbert = np.array(pd.read_csv("/kaggle/input/best-stanceeval/marbert_test_final.csv"))
camelbert = np.array(pd.read_csv("/kaggle/input/best-stanceeval/camelbert_test_final.csv"))

arabert = tuple(map(tuple, arabert))
marbert = tuple(map(tuple, marbert))
camelbert = tuple(map(tuple, camelbert))

f1_scores = {
    arabert: 0.897341741061115,
    marbert: 0.896563290821766,
    camelbert: 0.892321644953224
}

models = [arabert, marbert, camelbert]

create_new_submission(models)
