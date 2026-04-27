import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ===== 1. Load Data =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Housing.csv")

df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== 2. Encode Categorical Columns =====
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

# ===== 3. Features & Target =====
FEATURES = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

X = df[FEATURES]
y = df["price"]

# ===== 4. Train / Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 5. Train 3 Models =====
models = {}

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['random_forest'] = rf

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['linear_regression'] = (lr, scaler)

# XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
models['xgboost'] = xgb

print("✅ All 3 models trained!")

# ===== 6. Evaluate =====
for name, model_data in models.items():
    if name == 'linear_regression':
        mdl, scl = model_data
        y_pred = mdl.predict(X_test_scaled)
    else:
        y_pred = model_data.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📊 {name:20s} | MAE: {mae:>10,.0f} | R²: {r2:.4f}")

# ===== 7. Save Models =====
MODEL_PATH = os.path.join(BASE_DIR, "house_models.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(models, f)
print(f"✅ Models saved → {MODEL_PATH}")

# ===== 8. Unsupervised Learning (K-Means Clustering) =====
# Cluster houses by price ranges
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['price_cluster'] = kmeans.fit_predict(X)

# Analyze clusters
cluster_info = {}
for i in range(3):
    cluster_prices = df[df['price_cluster'] == i]['price']
    cluster_info[i] = {
        'min': cluster_prices.min(),
        'max': cluster_prices.max(),
        'mean': cluster_prices.mean(),
        'count': len(cluster_prices),
        'label': ['Budget', 'Standard', 'Luxury'][i]
    }

with open(os.path.join(BASE_DIR, "cluster_info.pkl"), "wb") as f:
    pickle.dump({'kmeans': kmeans, 'info': cluster_info}, f)

print("\n🏷️ Price Clusters:")
for i, info in cluster_info.items():
    print(f"   Cluster {i} ({info['label']}): {info['min']:,.0f} - {info['max']:,.0f} MAD (n={info['count']})")

# ===== 9. Prediction Function =====
def predict_price(area, bedrooms, bathrooms, stories,
                  mainroad, guestroom, basement, hotwaterheating,
                  airconditioning, parking, prefarea, furnishingstatus,
                  model_name='random_forest'):
    """
    Parameters
    ----------
    model_name : 'random_forest', 'linear_regression', or 'xgboost'
    """
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_models.pkl")
    with open(MODEL_PATH, "rb") as f:
        loaded_models = pickle.load(f)

    features = pd.DataFrame([[area, bedrooms, bathrooms, stories,
                               mainroad, guestroom, basement, hotwaterheating,
                               airconditioning, parking, prefarea, furnishingstatus]],
                             columns=FEATURES)

    if model_name == 'linear_regression':
        mdl, scl = loaded_models['linear_regression']
        features_scaled = scl.transform(features)
        return float(mdl.predict(features_scaled)[0])
    else:
        return float(loaded_models[model_name].predict(features)[0])


def get_price_cluster(area, bedrooms, bathrooms, stories,
                      mainroad, guestroom, basement, hotwaterheating,
                      airconditioning, parking, prefarea, furnishingstatus):
    """Get cluster label for a house"""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_info.pkl"), "rb") as f:
        data = pickle.load(f)

    features = pd.DataFrame([[area, bedrooms, bathrooms, stories,
                               mainroad, guestroom, basement, hotwaterheating,
                               airconditioning, parking, prefarea, furnishingstatus]],
                             columns=FEATURES)
    cluster = data['kmeans'].predict(features)[0]
    return data['info'][cluster]


if __name__ == "__main__":
    # Quick test
    for mdl in ['random_forest', 'linear_regression', 'xgboost']:
        price = predict_price(
            area=7420, bedrooms=4, bathrooms=2, stories=3,
            mainroad=1, guestroom=0, basement=0, hotwaterheating=0,
            airconditioning=1, parking=2, prefarea=1, furnishingstatus=2,
            model_name=mdl
        )
        print(f"\n🏠 {mdl}: {price:,.0f} MAD")

    cluster = get_price_cluster(
        area=7420, bedrooms=4, bathrooms=2, stories=3,
        mainroad=1, guestroom=0, basement=0, hotwaterheating=0,
        airconditioning=1, parking=2, prefarea=1, furnishingstatus=2
    )
    print(f"\n🏷️ Cluster: {cluster['label']} ({cluster['min']:,.0f} - {cluster['max']:,.0f} MAD)")
