from model import model

# Calculate probability for a given observation
# none rain, no maintenance, train on time, appointment attend 的 概率
probability = model.probability([["heavy", "yes", "delayed", "miss"]])

print(probability)
