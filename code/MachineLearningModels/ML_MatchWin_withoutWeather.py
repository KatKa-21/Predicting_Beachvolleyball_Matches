# ## ML Model ohne Wetter-Variablen


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate,StratifiedKFold, cross_val_score, train_test_split
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

#---------------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('DatenML_V2_relCol_ZusatzType.csv', sep=';')	

df = data.copy()
# Erstellen der neuen Spalte "Team1":
# Wenn TeamDesignation = A, dann soll TeamAName in Team1 stehen,
# andernfalls (also bei B) TeamBName.
df["Team1"] = np.where(
    df["TeamDesignation"].str.upper() == "A",
    df["@TeamAName"],
    df["@TeamBName"]
)

# Erstellen der Spalte "Team2" als das jeweils andere Team:
df["Team2"] = np.where(
    df["TeamDesignation"].str.upper() == "A",
    df["@TeamBName"],
    df["@TeamAName"]
)
#---------------------------------------------------------------------------------------------------------------------------

data1 = df.drop(['FirstName', 'LastName', 'TeamDesignation','@PointsTeamASet1', '@PointsTeamBSet1', '@PointsTeamASet2',
       '@PointsTeamBSet2', '@PointsTeamASet3', '@PointsTeamBSet3', 
       'FederationCode_y', '@TeamAName', '@TeamBName',
       '@DurationSet1','@LocalDate', '@LocalTime','temperature_2m',
       'precipitation', 'wind_speed_10m', 'rain', 'wind_gusts_10m',
       'FirstName2', 'LastName2','Type','TournamentNo','NoPlayer1_team', 'NoPlayer2_team',#'total_A', 'total_B', 'TeamName'-> nicht mehr da
       '@DurationSet2', '@DurationSet3'], axis=1)



y = data1.pop('match_win')
X = data1.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3146)
#---------------------------------------------------------------------------------------------------------------------------


# select categorical and numerical column names
nominal_features = X.select_dtypes(include=['object']).columns.tolist()
# Define feature types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessors for different feature types
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Build the column transformer
transformers = [
    ('num', numeric_transformer, numeric_features),
    ('nom', nominal_transformer, nominal_features)
]

preprocessor = ColumnTransformer(transformers=transformers)

#---------------------------------------------------------------------------------------------------------------------------

#Model Building
# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15, 
                                         min_samples_split=5, min_samples_leaf=2),
     'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1, 
                                                  max_depth=5, subsample=0.8)
}

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Function to train and evaluate models
def evaluate_model(name, model, X, y, cv):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    print(f"{name} - Cross-validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train on full dataset for feature importance
    pipeline.fit(X, y)
    
    return pipeline


# Evaluate all models
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = evaluate_model(name, model, X, y, cv)
    trained_models[name] = pipeline

# Get feature importance from one of the tree-based models (e.g., Random Forest)
rf_model = trained_models['GradientBoosting']
preprocessed_X = rf_model.named_steps['preprocessor'].transform(X)

# Get feature names after preprocessing
feature_names = []
for name, trans, cols in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(cols)
    elif name == 'nom':
        ohe = trans.named_steps['onehot']
        feature_names.extend([f"{col}_{cat}" for col in cols 
                            for cat in ohe.categories_[cols.index(col)]])
    else:  # ordinal features
        feature_names.append(name[4:])  # Remove 'ord_' prefix

# Get and display feature importances
if hasattr(rf_model.named_steps['classifier'], 'feature_importances_'):
    importances = rf_model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],  # Ensure lengths match
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Display top 20 features
    print("\nTop 20 most important features:")
    print(importance_df.head(20))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()

#---------------------------------------------------------------------------------------------------------------------------

# ## Modelle einzeln berechnen für Streamlit app


#RandomForestClassifier
pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15, 
                                         min_samples_split=5, min_samples_leaf=2))
    ])

rf = pipeline.fit(X_train, y_train)
# Trainieren der Pipeline
pipeline.fit(X_train, y_train)

# Vorhersagen
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
#################################################################
#Gradient Boosting
pipeline2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1, 
                                                  max_depth=5, subsample=0.8))
])

GB = pipeline2.fit(X_train, y_train)
# Trainieren der Pipeline
pipeline2.fit(X_train, y_train)

# Vorhersagen
y_pred2 = pipeline2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)

print(f"Model Accuracy: {accuracy2:.4f}")

# Vorhersagen auf den Trainingsdaten
y_train_pred2 = GB.predict(X_train)

# Accuracy auf Trainingsdaten
train_acc2 = accuracy_score(y_train, y_train_pred2)
print(f"Trainings-Accuracy: {train_acc2:.2f}")



