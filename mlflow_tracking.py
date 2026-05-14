# MLFLOW for model tracking and deployment
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score

mlflow.set_experiment('heart-disease-risk-prediction')

# Run 1: Logging the original Logistic Regression model
with mlflow.start_run(run_name = 'Logistic Regression_original'):
    mlflow.log_param('model','LogisticRegression')
    mlflow.log_param('C',1.0) # for a base or orginal model the default value of C is 1.0, we are logging it as a parameter for reference
    mlflow.log_param('max_iter',1000)

    mlflow.log_metric('accuracy', round(accuracy_score(y_test,y_pred_lr),3))
    mlflow.log_metric('f1_score',round(f1_score(y_test,y_pred_lr),3))
    mlflow.log_metric('recall_score',round(recall_score(y_test,y_pred_lr),3))
    mlflow.log_metric('train_score', round(lr_model.score(x_train_scaled, y_train), 3))
    mlflow.log_metric('test_score',  round(lr_model.score(x_test_scaled, y_test), 3))
    mlflow.sklearn.log_model(lr_model,'model')
    print('Logistic Regression Original logged ')

# Run 2: Logging the original Random Forest model
with mlflow.start_run(run_name='Random_Forest_Original'):
    
    mlflow.log_param('model', 'RandomForest_Original')
    mlflow.log_param('max_depth', 'None')
    mlflow.log_param('n_estimators', 100)
    
    mlflow.log_metric('accuracy', round(accuracy_score(y_test, y_pred_rf), 3))
    mlflow.log_metric('f1_score', round(f1_score(y_test, y_pred_rf), 3))
    mlflow.log_metric('recall',   round(recall_score(y_test, y_pred_rf), 3))
    mlflow.log_metric('train_score', round(rf_model.score(x_train_scaled, y_train), 3))
    mlflow.log_metric('test_score',  round(rf_model.score(x_test_scaled, y_test), 3))
    
    mlflow.sklearn.log_model(rf_model, 'model')
    print('RF Original logged')

# Run 3: Logging the tuned Logistic Regression model
with mlflow.start_run(run_name = 'Logistic_Regression_Tuned'):
    mlflow.log_param('model','LogisticRegression')
    mlflow.log_param('C',grid_search_lr.best_params_['classifier__C'])
    mlflow.log_param('max_iter',1000)
    
    mlflow.log_metric('accuracy', round(accuracy_score(y_test,y_best_lr),3))
    mlflow.log_metric('f1_score',round(f1_score(y_test,y_best_lr),3))
    mlflow.log_metric('recall_score',round(recall_score(y_test,y_best_lr),3))
    mlflow.log_metric('train_score', round(best_lr.score(x_train_scaled, y_train), 3))
    mlflow.log_metric('test_score',  round(best_lr.score(x_test_scaled, y_test), 3))
    mlflow.sklearn.log_model(best_lr,'model')
    print('Logistic Regression Tuned logged')

# Run 4: Logging the tuned Random Forest model
with mlflow.start_run(run_name='Random_Forest_Tuned'):
    mlflow.log_param('model', 'RandomForest_Tuned')
    mlflow.log_param('max_depth', grid_rf.best_params_['classifier__max_depth'])
    mlflow.log_param('n_estimators', grid_rf.best_params_['classifier__n_estimators'])
    mlflow.log_param('min_samples_leaf', grid_rf.best_params_['classifier__min_samples_leaf'])
    
    mlflow.log_metric('accuracy', round(accuracy_score(y_test, y_pred_tuned_rf), 3))
    mlflow.log_metric('f1_score', round(f1_score(y_test, y_pred_tuned_rf), 3))
    mlflow.log_metric('recall',   round(recall_score(y_test, y_pred_tuned_rf), 3))
    mlflow.log_metric('train_score', round(best_rf.score(x_train_scaled, y_train), 3))
    mlflow.log_metric('test_score',  round(best_rf.score(x_test_scaled, y_test), 3))
    
    mlflow.sklearn.log_model(best_rf, 'model')
    print('RF Tuned logged')


# Register the best model in the MLflow Model Registry
# Logistic Regression Original is the best model based on balance of accuracy, f1_score, recall and overfitting/underfitting analysis, so we will register the Logistic Regression Original model in the MLflow Model Registry
from mlflow.tracking import MlflowClient
client = MlflowClient()
# Get the run ID of the best model 
run_id = "95520bfe5e6e44ca9cf8b09629153bf9"

registered = mlflow.register_model(
            model_uri = f"runs:/{run_id}/model",
            name = "heart_disease_production_model"
)
print(f'Registered model {registered.name} with version {registered.version}')

# Transition to Production stage 
client.transition_model_version_stage(
    name = registered.name,
    version = registered.version,
    stage = 'Production',
    archive_existing_versions = True
)
print('Model is in Production')
