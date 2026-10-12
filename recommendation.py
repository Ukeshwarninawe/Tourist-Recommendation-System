
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
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

# Use same scaler for testing data
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

# Find 5 nearest training destinations for every test destination
distances, indices = model.kneighbors(X_test_scaled)
print("\nTesting Completed!")


# Display recommendations for first 5 test destinations
for i in range(min(5, len(test_df))):

    test_destination = test_df.iloc[i]["Destination Name"]

    print("\nTest Destination:", test_destination)
    print("Recommended Destinations:")

    for j in indices[i]:

        recommended_destination = train_df.iloc[j]["Destination Name"]

        print("-", recommended_destination)


# 7. MODEL EVALUATION
average_distance = distances.mean()
print("\nModel Evaluation")
print("Average Euclidean Distance:", average_distance)


# ACCURACY CALCULATION
# Calculate how many recommendations have a distance below the chosen threshold

threshold = 2.0
total_recommendations = distances.size
correct_recommendations = (distances <= threshold).sum()
accuracy = (
    correct_recommendations / total_recommendations
) * 100
print("Recommendation Accuracy:", round(accuracy, 2), "%")


# 7A. K-FOLD CROSS-VALIDATION

print("\n5-Fold Cross-Validation")
print("-----------------------")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

X_all = df[features]

fold_distances_list = []
fold_accuracy_list = []

fold_number = 1

for train_index, val_index in kf.split(X_all):

    # Split data for the current fold
    X_fold_train = X_all.iloc[train_index]
    X_fold_val = X_all.iloc[val_index]

    # Scale features using only the current fold's training data
    cv_scaler = StandardScaler()

    X_fold_train_scaled = cv_scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = cv_scaler.transform(X_fold_val)

    # Create and train a separate KNN model for this fold
    cv_model = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean"
    )

    cv_model.fit(X_fold_train_scaled)

    # Find the 5 nearest training destinations for each validation destination
    fold_distances, fold_indices = cv_model.kneighbors(
        X_fold_val_scaled
    )

    # Calculate average Euclidean distance for this fold
    fold_average_distance = fold_distances.mean()

    # Calculate threshold-based similarity percentage
    fold_total = fold_distances.size
    fold_correct = (fold_distances <= threshold).sum()

    fold_accuracy = (fold_correct / fold_total) * 100

    fold_distances_list.extend(fold_distances.flatten())
    fold_accuracy_list.append(fold_accuracy)

    print("\nFold", fold_number)
    print("Average Euclidean Distance:", round(fold_average_distance, 4))
    print("Threshold-Based Similarity:", round(fold_accuracy, 2), "%")

    fold_number += 1


# Overall cross-validation results
cv_average_distance = sum(fold_distances_list) / len(fold_distances_list)
cv_average_accuracy = sum(fold_accuracy_list) / len(fold_accuracy_list)

print("\nOverall Cross-Validation Results")
print("--------------------------------")
print("Number of Folds: 5")
print("Mean Euclidean Distance:", round(cv_average_distance, 4))
print(
    "Mean Threshold-Based Similarity:",
    round(cv_average_accuracy, 2),
    "%"
)


# 8. NEW USER PREFERENCES
# Order: Nature, Adventure, Historical, Beach, Wildlife, Family_Friendly, Nightlife

user_preferences = [[5, 2, 1, 5, 2, 4, 5]]

# Scale new user data using same scaler
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
