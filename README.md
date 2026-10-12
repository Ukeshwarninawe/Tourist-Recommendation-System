# Tourist-Recommendation-System
A KNN-based tourism recommendation system that recommends tourist destinations based on user preferences such as nature, adventure, history, beaches, wildlife, family-friendliness, and nightlife.

# Tourism Recommendation System

A machine learning-based tourism recommendation system that uses **K-Nearest Neighbors (KNN)** and **Euclidean distance** to recommend tourist destinations based on user preferences.

## Project Overview

This project recommends tourist destinations by comparing a user's preferences with the profiles of existing destinations.

The system considers seven tourism-related features:

- Nature
- Adventure
- Historical
- Beach
- Wildlife
- Family Friendly
- Nightlife

Based on these preferences, the system finds the **5 most similar tourist destinations**.

## Machine Learning Approach

This project uses a **Content-Based Recommendation System** with the K-Nearest Neighbors algorithm.

### Process

1. Load the tourism dataset.
2. Clean and preprocess the data.
3. Split the dataset into:
   - 80% Training Data
   - 20% Testing Data
4. Select seven recommendation features.
5. Standardize the features using `StandardScaler`.
6. Train a KNN model using Euclidean distance.
7. Test the model using the test dataset.
8. Calculate the average Euclidean distance.
9. Accept new user preferences.
10. Recommend the 5 most similar tourist destinations.
11. Visualize the recommendations using a bar graph.

## Dataset

The dataset contains **110 tourist destinations** and **15 features**.

### Recommendation Features

| Feature | Description |
|---|---|
| Nature | Preference/rating for nature-based tourism |
| Adventure | Preference/rating for adventure activities |
| Historical | Preference/rating for historical places |
| Beach | Preference/rating for beach destinations |
| Wildlife | Preference/rating for wildlife tourism |
| Family_Friendly | Suitability/preference for family travel |
| Nightlife | Preference/rating for nightlife |

Other columns contain destination information such as:

- Destination Name
- State
- Region
- Category
- Popular Attraction
- Accessibility
- Nearest Airport
- Nearest Railway Station

## Model

### K-Nearest Neighbors

The KNN model uses:

- `n_neighbors = 5`
- Distance metric = Euclidean Distance

The model finds the five destinations with the smallest distance from the user's preference profile.

### Example User Input

```text
Nature          = 5
Adventure       = 2
Historical      = 1
Beach           = 5
Wildlife        = 2
Family_Friendly = 4
Nightlife       = 5
