import pandas as pd
data = pd.read_csv("training_data.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values
num_attributes = concepts.shape[1]
S = ['0'] * num_attributes  
G = [['?'] * num_attributes]
for i, example in enumerate(concepts):
    if target[i] == 'Yes':
        for j in range(num_attributes):
            if S[j] == '0':
                S[j] = example[j]
            elif S[j] != example[j]:
                S[j] = '?'
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(num_attributes))]

    else:
        new_G = []
        for g in G:
            for j in range(num_attributes):
                if g[j] == '?' and S[j] != example[j]:
                    new_hypothesis = g.copy()
                    new_hypothesis[j] = S[j]
                    new_G.append(new_hypothesis)
        G = new_G
    print(f"\nStep {i+1}")
    print("S =", S)
    print("G =", G)
print("\nFinal Specific Hypothesis (S):", S)
print("Final General Hypotheses (G):", G)
