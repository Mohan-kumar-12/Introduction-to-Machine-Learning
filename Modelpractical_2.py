import csv

# Read training data
data = []
with open('enjoysport.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)   # Skip header
    for row in reader:
        data.append(row)

print("Training Data:")
print(data)

print("\nTotal number of training instances:", len(data))

# Number of attributes (excluding target)
num_attributes = len(data[0]) - 1

# Initialize hypothesis
hypothesis = ['0'] * num_attributes
print("\nInitial Hypothesis:")
print(hypothesis)

# Apply Find-S algorithm
for i in range(len(data)):
    if data[i][num_attributes].lower() == 'yes':
        for j in range(num_attributes):
            if hypothesis[j] == '0':
                hypothesis[j] = data[i][j]
            elif hypothesis[j] != data[i][j]:
                hypothesis[j] = '?'
        print(f"\nHypothesis after training instance {i+1}:")
        print(hypothesis)

print("\nFinal Maximally Specific Hypothesis:")
print(hypothesis)
