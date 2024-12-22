import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def plot_full_metrics(history, title, model_name):
    """
    Plot training and validation metrics for a given model's training history.

    Parameters:
    - history: Keras history object
    - title: Title for the plot
    - model_name: Name of the model (e.g., "FFNN" or "CNN")
    """
    # Metrics to plot
    metrics = [
        'accuracy', 'true_negatives', 'true_positives',
        'false_negatives', 'false_positives',
        'precision', 'recall', 'f1_score'
    ]
    val_metrics = [f'val_{metric}' for metric in metrics]

    plt.figure(figsize=(20, 16))  # Adjust figure size for better visibility
    num_metrics = len(metrics)

    # Create subplots for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)  # Create a grid of 3x3 subplots
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}', color='blue')
        plt.plot(history.history[val_metrics[i]], label=f'Validation {metric.capitalize()}', color='orange')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()

    # Set the overall title
    plt.suptitle(f'{title} ({model_name})', fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.show()



class Charts:
    def __init__(self, data):
        self.data = data


    def show_ethnicity_chart(self):
        ################## Ethnicity
        ethnicity_counts = self.data['Ethnicity'].value_counts()
        asd_yes_ethnicity_counts = self.data[self.data['ASD'] == 'Yes']['Ethnicity'].value_counts()
        categories = ethnicity_counts.index
        asd_yes_ethnicity_counts = asd_yes_ethnicity_counts.reindex(categories, fill_value=0)
        plt.figure(figsize=(10, 8))  # Width: 10, Height: 8
        bars_all = plt.bar(categories, ethnicity_counts, color='skyblue', label='All Responses')
        bars_asd = plt.bar(categories, asd_yes_ethnicity_counts, color='lightcoral', alpha=0.7, label='ASD Traits = Yes')

        for bar, category in zip(bars_all, categories):
            total_count = ethnicity_counts.sum()
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, 660.5, f'{percentage:.1f}%', ha='center', fontsize=10)

        for bar, category in zip(bars_asd, categories):
            total_count = ethnicity_counts[category]
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10, color='darkred')

        plt.title('Ethnicity Distribution with ASD Traits Comparison', fontsize=16)
        plt.xlabel('Ethnicity', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    # Function to show Age chart
    def show_age_chart(self):
        ################## Age
        age_bins = [0, 12, 24, 36, 48, 60, 72, 84]  # 1-year increments in months
        age_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '73-84']
        self.data['Age_Range'] = pd.cut(self.data['Age_Mons'], bins=age_bins, labels=age_labels, right=False)

        # Count occurrences of each age range
        age_range_counts = self.data['Age_Range'].value_counts().sort_index()

        # Count occurrences of each age range where ASD_traits is "Yes"
        asd_yes_age_range_counts = self.data[self.data['ASD'] == 'Yes']['Age_Range'].value_counts().sort_index()

        # Ensure both datasets have the same order of categories
        categories = age_range_counts.index
        asd_yes_age_range_counts = asd_yes_age_range_counts.reindex(categories, fill_value=0)
        plt.figure(figsize=(10, 8))
        bars_all = plt.bar(categories, age_range_counts, color='lightgreen', label='All Responses')
        bars_asd = plt.bar(categories, asd_yes_age_range_counts, color='orange', alpha=0.7, label='ASD Traits = Yes')

        for bar, category in zip(bars_all, categories):
            total_count = age_range_counts.sum()
            if total_count == 0:
                continue
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10)

        for bar, category in zip(bars_asd, categories):
            total_count = age_range_counts[category]
            if total_count == 0:
                continue
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10, color='darkred')

        plt.title('Age Distribution with ASD Traits Comparison', fontsize=16)
        plt.xlabel('Age Range (Months)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    # Function to show Sex chart
    def show_sex_chart(self):
        ################## Sex
        sex_counts = self.data['Sex'].value_counts()

        # Count occurrences of each sex where ASD_traits is "Yes"
        asd_yes_sex_counts = self.data[self.data['ASD'] == 'Yes']['Sex'].value_counts()

        # Ensure both datasets have the same order of categories
        categories = sex_counts.index
        asd_yes_sex_counts = asd_yes_sex_counts.reindex(categories, fill_value=0)

        plt.figure(figsize=(8, 6))
        bars_all = plt.bar(categories, sex_counts, color='lightblue', label='All Responses')
        bars_asd = plt.bar(categories, asd_yes_sex_counts, color='pink', alpha=0.7, label='ASD Traits = Yes')

        for bar, category in zip(bars_all, categories):
            total_count = sex_counts.sum()
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10)

        for bar, category in zip(bars_asd, categories):
            total_count = sex_counts[category]
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10, color='darkred')

        plt.title('Sex Distribution with ASD Traits Comparison', fontsize=16)
        plt.xlabel('Sex', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def show_family_mem_with_asd_chart(self):
        ################## Family Member with ASD
        family_counts = self.data['Family_mem_with_ASD'].value_counts()

        # Count occurrences where ASD_traits is "Yes"
        asd_yes_family_counts = self.data[self.data['ASD'] == 'Yes']['Family_mem_with_ASD'].value_counts()

        # Ensure both datasets have the same order of categories
        categories = family_counts.index
        asd_yes_family_counts = asd_yes_family_counts.reindex(categories, fill_value=0)

        plt.figure(figsize=(8, 6))
        bars_all = plt.bar(categories, family_counts, color='gold', label='All Responses')
        bars_asd = plt.bar(categories, asd_yes_family_counts, color='purple', alpha=0.7, label='ASD Traits = Yes')

        for bar, category in zip(bars_all, categories):
            total_count = family_counts.sum()
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center',
                     fontsize=10)

        for bar, category in zip(bars_asd, categories):
            total_count = family_counts[category]
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center',
                     fontsize=10, color='darkred')

        plt.title('Family Member with ASD Distribution', fontsize=16)
        plt.xlabel('Family Member with ASD', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def show_jaundice_chart(self):
        ################## Jaundice
        jaundice_counts = self.data['Jaundice'].value_counts()

        # Count occurrences where ASD_traits is "Yes"
        asd_yes_jaundice_counts = self.data[self.data['ASD'] == 'Yes']['Jaundice'].value_counts()

        # Ensure both datasets have the same order of categories
        categories = jaundice_counts.index
        asd_yes_jaundice_counts = asd_yes_jaundice_counts.reindex(categories, fill_value=0)

        plt.figure(figsize=(8, 6))
        bars_all = plt.bar(categories, jaundice_counts, color='lightblue', label='All Responses')
        bars_asd = plt.bar(categories, asd_yes_jaundice_counts, color='orange', alpha=0.7, label='ASD Traits = Yes')

        for bar, category in zip(bars_all, categories):
            total_count = jaundice_counts.sum()
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center',
                     fontsize=10)

        for bar, category in zip(bars_asd, categories):
            total_count = jaundice_counts[category]
            percentage = (bar.get_height() / total_count * 100)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center',
                     fontsize=10, color='darkred')

        plt.title('Jaundice Distribution with ASD Traits Comparison', fontsize=16)
        plt.xlabel('History of Jaundice', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(ticks=range(len(categories)), labels=['No', 'Yes'],
                   fontsize=12)  # Assuming categories are "Yes" and "No"
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def perform_pca_a1_to_a10(self):
        pca_data = self.data[[f'A{i}' for i in range(1, 11)]].copy()

        # Apply PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(pca_data)

        # Create a DataFrame for the PCA components
        pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
        pca_df['ASD'] = self.data['ASD']

        # Scatter plot of PCA components colored by ASD
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['ASD'].map({'Yes': 1, 'No': 0}), cmap='coolwarm', alpha=0.7)
        plt.title('PCA of A1-A10 with ASD Diagnosis', fontsize=16)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='ASD Diagnosis')
        plt.show()


    def perform_pca_all(self):
        features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                    'Ethnicity', 'Age_Mons', 'Sex', 'Family_mem_with_ASD', 'Jaundice']

        # Encode categorical variables and scale all features

        categorical_features = ['Ethnicity', 'Sex', 'Family_mem_with_ASD', 'Jaundice']
        numerical_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons']

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        # Apply preprocessing
        X = preprocessor.fit_transform(self.data[features])

        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)

        # Create a DataFrame for plotting
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['ASD'] = self.data['ASD']  # Add ASD diagnosis for coloring

        # Plot the PCA result
        plt.figure(figsize=(10, 8))
        colors = {'Yes': 'red', 'No': 'blue'}
        for diagnosis in colors:
            subset = pca_df[pca_df['ASD'] == diagnosis]
            plt.scatter(subset['PC1'], subset['PC2'], label=f'ASD = {diagnosis}', color=colors[diagnosis], alpha=0.6)

        # Add plot details
        plt.title('PCA: A1-A10 + Demographic Features', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% Variance)', fontsize=14)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% Variance)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def perform_feature_importance_a1_to_a10(self):
        # Prepare the data (convert ASD to binary and drop rows with missing values)
        data_clean = self.data.dropna(subset=[f'A{i}' for i in range(1, 11)])
        X = data_clean[[f'A{i}' for i in range(1, 11)]]
        y = data_clean['ASD'].map({'Yes': 1, 'No': 0})

        # Train a decision tree classifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(X.columns, clf.feature_importances_)
        plt.title('Feature Importance from Decision Tree Classifier', fontsize=16)
        plt.xlabel('Importance')
        plt.ylabel('Question')
        plt.show()


    def perform_feature_importance_all(self):
        features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                    'Ethnicity', 'Age_Mons', 'Sex', 'Family_mem_with_ASD', 'Jaundice']
        target = 'ASD'

        # Encode categorical variables and scale numerical features
        categorical_features = ['Ethnicity', 'Sex', 'Family_mem_with_ASD', 'Jaundice']
        numerical_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons']

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        # Prepare the features (X) and target (y)
        X = preprocessor.fit_transform(self.data[features])
        y = self.data[target].map({'Yes': 1, 'No': 0})  # Convert ASD to binary

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train, y_train)

        # Extract feature importance
        importance_scores = dt_classifier.feature_importances_

        # Get feature names after encoding
        feature_names = (numerical_features +
                         list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

        # Sort feature importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_importance = importance_scores[sorted_idx]
        sorted_features = np.array(feature_names)[sorted_idx]

        encoded_feature_mapping = {
            'Ethnicity': [col for col in feature_names if col.startswith('Ethnicity')],
            'Sex': [col for col in feature_names if col.startswith('Sex')],
            'Family_mem_with_ASD': [col for col in feature_names if col.startswith('Family_mem_with_ASD')],
            'Jaundice': [col for col in feature_names if col.startswith('Jaundice')],
        }

        # Initialize aggregated feature importance dictionary
        aggregated_importance = {}

        # Sum importance scores for grouped features
        for original_feature, encoded_features in encoded_feature_mapping.items():
            aggregated_importance[original_feature] = sum(
                importance_scores[feature_names.index(encoded)] for encoded in encoded_features
            )

        # Add importance of non-categorical features
        for feature in numerical_features:
            aggregated_importance[feature] = importance_scores[feature_names.index(feature)]

        # Sort aggregated importance
        sorted_aggregated_importance = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)

        # Extract sorted feature names and scores
        sorted_features = [item[0] for item in sorted_aggregated_importance]
        sorted_scores = [item[1] for item in sorted_aggregated_importance]

        # Plot aggregated feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features, sorted_scores, color='skyblue')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.title('Aggregated Feature Importance from Decision Tree Classifier', fontsize=16)
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tight_layout()
        plt.show()
