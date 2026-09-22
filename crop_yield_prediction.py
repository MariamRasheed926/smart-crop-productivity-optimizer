import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

data = pd.DataFrame({
    "N": np.random.uniform(20, 140, 1200),
    "P": np.random.uniform(10, 100, 1200),
    "K": np.random.uniform(10, 120, 1200),
    "temperature": np.random.uniform(15, 35, 1200),
    "humidity": np.random.uniform(35, 90, 1200),
    "rainfall": np.random.uniform(200, 1000, 1200),
    "ph": np.random.uniform(5, 8, 1200)
})

optimized_yield = (
    25 + 0.18 * data["N"] +
    0.10 * data["P"] +
    0.08 * data["K"] +
    0.055 * data["rainfall"] +
    0.3 * data["humidity"] -
    0.7 * (data["temperature"] - 25) ** 2 +
    4 * np.sin(data["ph"]) +
    np.random.normal(0, 5, 1200)
)

X = data
y = optimized_yield

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
