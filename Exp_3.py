import pandas as pd
import math
data = pd.read_csv("play_tennis.csv")
def entropy(target_col):
    values = target_col.value_counts()
    total = len(target_col)
    ent = 0
    for count in values:
        p = count / total
        ent -= p * math.log2(p)
    return ent
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[feature] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy
def id3(data, features, target):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if len(features) == 0:
        return data[target].mode()[0]
    gains = {feature: information_gain(data, feature, target) for feature in features}
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(subset, remaining_features, target)
    return tree
features = list(data.columns[:-1])
target = "PlayTennis"
decision_tree = id3(data, features, target)
print("Decision Tree:")
print(decision_tree)
