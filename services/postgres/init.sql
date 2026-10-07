-- One PostgreSQL instance, three databases: MLflow metadata, Prefect state, and the app's
-- prediction / human-label log (tables are created by SQLAlchemy on API start-up).
CREATE DATABASE mlflow;
CREATE DATABASE prefect;
CREATE DATABASE app;
