import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# 1. LOAD DATASET

df = pd.read_csv("cleaned_dataset.csv")


# 7 features used for recommendation
features = [
    "Nature",
    "Adventure",
    "Historical",
    "Beach",
    "Wildlife",
    "Family_Friendly",
    "Nightlife"
]


# 2. TRAIN-TEST SPLIT

# 80% training, 20% testing
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

print("Total Dataset:", df.shape)
print("Training Dataset:", train_df.shape)
print("Testing Dataset:", test_df.shape)


# Save training and testing datasets
train_df.to_csv("training_dataset.csv", index=False)
test_df.to_csv("testing_dataset.csv", index=False)

# 3. SELECT FEATURES

X_train = train_df[features]
X_test = test_df[features]

# 4. FEATURE SCALING

scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler for testing data
X_test_scaled = scaler.transform(X_test)

# 5. CREATE KNN RECOMMENDATION MODEL


model = NearestNeighbors(
    n_neighbors=5,
    metric="euclidean"
)


# Train the model
model.fit(X_train_scaled)

print("\nModel Training Completed!")


# 6. TESTING THE MODEL

# Find 5 nearest training destinations
# for every test destination
distances, indices = model.kneighbors(X_test_scaled)

print("\nTesting Completed!")


# Display recommendations for first 5 test destinations
for i in range(5):

    test_destination = test_df.iloc[i]["Destination Name"]

    print("\nTest Destination:", test_destination)
    print("Recommended Destinations:")

    for j in indices[i]:

        recommended_destination = train_df.iloc[j]["Destination Name"]

        print("-", recommended_destination)


# 7. MODEL EVALUATION


average_distance = distances.mean()

print("\nModel Evaluation")
print("----------------")
print("Average Euclidean Distance:", average_distance)


# 8. NEW USER PREFERENCES


# Order:
# Nature, Adventure, Historical, Beach, Wildlife, Family_Friendly, Nightlife

user_preferences = [[5, 2, 1, 5, 2, 4, 5]]


# Scale new user data using the same scaler
user_scaled = scaler.transform(user_preferences)


# Find 5 nearest destinations
distances, indices = model.kneighbors(user_scaled)

# 9. DISPLAY NEW USER RECOMMENDATIONS

print("\nNew User Recommendations")
print("------------------------")


recommended_names = []
recommended_distances = []


for i in range(5):

    index = indices[0][i]

    destination = train_df.iloc[index]["Destination Name"]
    state = train_df.iloc[index]["State"]
    category = train_df.iloc[index]["Category"]
    distance = distances[0][i]

    recommended_names.append(destination)
    recommended_distances.append(distance)

    print(
        i + 1,
        destination,
        "| State:", state,
        "| Category:", category,
        "| Distance:", distance
    )


# 10. GRAPH OF TOP 5 RECOMMENDATIONS

plt.figure(figsize=(10, 6))

plt.bar(
    recommended_names,
    recommended_distances
)

plt.xlabel("Recommended Destinations")
plt.ylabel("Euclidean Distance")

plt.title(
    "Top 5 Recommended Tourist Destinations"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()