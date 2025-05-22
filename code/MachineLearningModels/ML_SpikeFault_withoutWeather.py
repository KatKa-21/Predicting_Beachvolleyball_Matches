
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate,StratifiedKFold, cross_val_score, train_test_split
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

#---------------------------------------------------------------------------------------------------
#Import data

data = pd.read_csv('ML_Data_Teams.csv', sep=';')	


data1 = data[['@PositionInEntry',
       '@Rank', '@EarnedPointsTeam', '@EarningsTotalTeam', 'Gender_x', 'Type',
       'SpikeFault', 'SpikePoint', 'ServeFault', 'ServePoint',
       'ServeTotal', 'BlockPoint', 'BlockTotal', 'DigTotal', 'ReceptionFault',
       'SpikeTotal', '@LocalDate', '@LocalTime', 'FederationCode_y',
       'FirstName', 'LastName', 'FirstName2', 'LastName2',   '@DurationSet1',
       '@DurationSet2', '@DurationSet3', 'TeamFault_team',
       'match_win', 'Standard_Namen', 'Standard_Namen_team2']]

#Variablen umbenennen
data1 = data1.rename(columns={'Standard_Namen': 'Team1', 'Standard_Namen_team2': 'Team2'})

special_feature = '@DurationSet3'
data2 = data1.copy()
# Erzeuge den Indikator: 1, wenn in @DurationSet3 ein Wert vorhanden ist, sonst 0.
data2[special_feature + '_indicator'] = data2[special_feature].notnull().astype(int)

#-------------------------------------------------------------------------------------------------------
y = data2.pop('SpikeFault')
X = data2.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

#---------------------------------------------------------------------------------------------------------

nominal_features = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Erstelle eine Liste der allgemeinen numerischen Features, exklusive des speziellen Features und dessen Indikator
general_numeric_features = [col for col in numeric_features if col not in [special_feature, special_feature + '_indicator']]

# Erstelle separate Pipelines:
# 1. Für allgemeine numerische Features mit KNNImputer
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# 2. Für das spezielle Feature, das immer mit 0 imputiert werden soll (das signalisiert, dass kein dritter Satz stattgefunden hat)
special_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

# 3. Für die Indikatorspalte (@DurationSet3_indicator)
indicator_transformer = Pipeline(steps=[
    # Hier ist meist keine Imputation nötig; Skalierung ist optional, da 0 und 1 oft schon aussagekräftig sind.
    ('scaler', StandardScaler())
])

# 4. Für nominale Features: Imputation mit dem häufigsten Wert + OneHotEncoding
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Baue den ColumnTransformer unter Berücksichtigung aller Feature-Gruppen:
transformers = [
    ('num', numeric_transformer, general_numeric_features),
    ('spec_num', special_numeric_transformer, [special_feature]),
    ('indicator', indicator_transformer, [special_feature + '_indicator']),
    ('nom', nominal_transformer, nominal_features)
]

preprocessor = ColumnTransformer(transformers=transformers)

#------------------------------------------------------------------------------------------------------------------

model_scores = []
def score_model(model_name, y_true, y_pred):
  scores = {
      'Model': model_name,
      'MAE ($)': round(mean_absolute_error(y_true, y_pred), 2),
      'RMSE ($)': round(root_mean_squared_error(y_true, y_pred), 2),
      'MAPE (%)': round(100 * mean_absolute_percentage_error(y_true, y_pred), 2),
      'R-Squared': round(r2_score(y_true, y_pred), 3)
  }
  return scores

#---------------------------------------------------------------------------------------------------------------
#GradientBoostingRegressor

gbr_pipeline = make_pipeline(preprocessor,
                             RobustScaler(),
                             GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, 
                                                  max_depth=5, subsample=0.8))

GB = gbr_pipeline.fit(X_train, y_train)

gbr_predictions = gbr_pipeline.predict(X_test)

model_scores.append(score_model('GradientBoostigRegressor', y_test, gbr_predictions))

# 1. Extrahiere das trainierte Model (der letzte Schritt in der Pipeline)
model = gbr_pipeline.named_steps['gradientboostingregressor']

# 2. Extrahiere den Preprocessor-Schritt, der die Features formatiert hat.
# Je nachdem, wie deine Pipeline aufgebaut ist, kann der Schritt-Name variieren.
# Beim make_pipeline werden die Namen automatisch generiert:
preprocessor = gbr_pipeline.named_steps[list(gbr_pipeline.named_steps.keys())[1]]
# Alternativ, wenn du die Pipeline explizit mit Namen erstellt hast, z.B. im Pipeline-Konstruktur,
# dann wäre es etwas wie:
preprocessor = gbr_pipeline.named_steps['columntransformer']

# 3. Hole die Feature-Namen aus dem Preprocessor:
# Diese Methode funktioniert, wenn der Preprocessor und seine Zwischenschritte get_feature_names_out unterstützen.
feature_names = preprocessor.get_feature_names_out()

# 4. Extrahiere die Feature Importances aus dem Model
importances = model.feature_importances_

# 5. Erstelle einen DataFrame zum besseren Überblick
feature_importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importances_df.head(10))



train_r2 = gbr_pipeline.score(X_train, y_train)
test_r2  = gbr_pipeline.score(X_test, y_test)
print("Train R²:", train_r2)
print("Test R²:", test_r2)

#--------------------------------------------------------------------------------------------------
#RandomForestRegressor

rf_pipeline = make_pipeline(preprocessor,
                             RobustScaler(),
                             RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15, 
                                         min_samples_split=5, min_samples_leaf=2))

RF = rf_pipeline.fit(X_train, y_train)

rf_predictions = rf_pipeline.predict(X_test)

model_scores.append(score_model('RandomForest', y_test, rf_predictions))
pd.DataFrame(model_scores)

train_r2 = rf_pipeline.score(X_train, y_train)
test_r2  = rf_pipeline.score(X_test, y_test)
print("Train R²:", train_r2)
print("Test R²:", test_r2)


# 1. Extrahiere das trainierte Model (der letzte Schritt in der Pipeline)
model = rf_pipeline.named_steps['randomforestregressor']

# 2. Extrahiere den Preprocessor-Schritt, der die Features formatiert hat.
# Je nachdem, wie deine Pipeline aufgebaut ist, kann der Schritt-Name variieren.
# Beim make_pipeline werden die Namen automatisch generiert:
preprocessor = rf_pipeline.named_steps[list(rf_pipeline.named_steps.keys())[1]]
# Alternativ, wenn du die Pipeline explizit mit Namen erstellt hast, z.B. im Pipeline-Konstruktur,
# dann wäre es etwas wie:

preprocessor = rf_pipeline.named_steps['columntransformer']

# 3. Hole die Feature-Namen aus dem Preprocessor:
# Diese Methode funktioniert, wenn der Preprocessor und seine Zwischenschritte get_feature_names_out unterstützen.
feature_names = preprocessor.get_feature_names_out()

# 4. Extrahiere die Feature Importances aus dem Model
importances = model.feature_importances_

# 5. Erstelle einen DataFrame zum besseren Überblick
feature_importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importances_df.head(10))






