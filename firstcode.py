import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def remove_outliers_iqr(df, columns):
   
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
        df = df.loc[filter]
    return df


train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')


train_data = pd.merge(train_features, train_labels, on='respondent_id')


X_train_full = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_train_xyz = train_data['xyz_vaccine']
y_train_seasonal = train_data['seasonal_vaccine']

print("Shapes after merging and separating features and targets:")
print(f"X_train_full: {X_train_full.shape}")
print(f"y_train_xyz: {y_train_xyz.shape}")
print(f"y_train_seasonal: {y_train_seasonal.shape}")


numerical_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_full.select_dtypes(include=['object']).columns


X_train_full = remove_outliers_iqr(X_train_full, numerical_cols)

print("Shape after removing outliers:")
print(f"X_train_full: {X_train_full.shape}")


for col in categorical_cols:
    le = LabelEncoder()
    X_train_full[col] = le.fit_transform(X_train_full[col].astype(str))

imputer = SimpleImputer(strategy='mean')
X_train_full_imputed = imputer.fit_transform(X_train_full)
X_train_full = pd.DataFrame(X_train_full_imputed, columns=X_train_full.columns)


corr_matrix = X_train_full.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_train_full = X_train_full.drop(columns=to_drop)

print("Shape after removing highly correlated features:")
print(f"X_train_full: {X_train_full.shape}")


y_train_xyz = y_train_xyz.loc[X_train_full.index]
y_train_seasonal = y_train_seasonal.loc[X_train_full.index]

print("Shapes after aligning labels with features:")
print(f"y_train_xyz: {y_train_xyz.shape}")
print(f"y_train_seasonal: {y_train_seasonal.shape}")


X_train, X_valid, y_train_xyz, y_valid_xyz, y_train_seasonal, y_valid_seasonal = train_test_split(
    X_train_full, y_train_xyz, y_train_seasonal, test_size=0.2, random_state=42
)

print("Shapes after train_test_split:")
print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")
print(f"y_train_xyz: {y_train_xyz.shape}, y_valid_xyz: {y_valid_xyz.shape}")
print(f"y_train_seasonal: {y_train_seasonal.shape}, y_valid_seasonal: {y_valid_seasonal.shape}")


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


numerical_transformer = SimpleImputer(strategy='mean')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


model_xyz = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_seasonal = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


pipeline_xyz = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_xyz)
])

pipeline_seasonal = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_seasonal)
])


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0]
}


grid_search_xyz = GridSearchCV(pipeline_xyz, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_seasonal = GridSearchCV(pipeline_seasonal, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)


grid_search_xyz.fit(X_train, y_train_xyz)
grid_search_seasonal.fit(X_train, y_train_seasonal)


best_model_xyz = grid_search_xyz.best_estimator_
best_model_seasonal = grid_search_seasonal.best_estimator_


preds_xyz = best_model_xyz.predict_proba(X_valid)[:, 1]
preds_seasonal = best_model_seasonal.predict_proba(X_valid)[:, 1]


auc_xyz = roc_auc_score(y_valid_xyz, preds_xyz)
auc_seasonal = roc_auc_score(y_valid_seasonal, preds_seasonal)

print(f'Best ROC AUC for xyz_vaccine: {auc_xyz}')
print(f'Best ROC AUC for seasonal_vaccine: {auc_seasonal}')


test_features = test_features.drop(columns=['respondent_id'])  

for col in categorical_cols:
    le = LabelEncoder()
    test_features[col] = le.fit_transform(test_features[col].astype(str))

test_features_imputed = imputer.transform(test_features)
test_features = pd.DataFrame(test_features_imputed, columns=test_features.columns)
test_features = test_features.drop(columns=to_drop)


final_preds_xyz = best_model_xyz.predict_proba(test_features)[:, 1]
final_preds_seasonal = best_model_seasonal.predict_proba(test_features)[:, 1]


submission = pd.DataFrame({
    'respondent_id': test_features.index,  
    'xyz_vaccine': final_preds_xyz,
    'seasonal_vaccine': final_preds_seasonal
})

submission.to_csv('submission1.csv', index=False)
