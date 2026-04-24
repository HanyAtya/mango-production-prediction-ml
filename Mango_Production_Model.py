# Mango Production Prediction Model
# Location: Ismailia, Egypt 
# Objective:
# This project aims to analyze environmental factors affecting Mango production
# and build a predictive model using machine learning.

# Note:
# The dataset used in this project is synthetically generated for learning purposes.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Data Generation
# =========================

np.random.seed(42) # constant random values every Run.
Temperature = np.random.uniform(20,50,50) /50
Soil_Quality = np.random.uniform (0,50,50) /50
Humidity = np.random.uniform (0,50,50) /50
Fertilizer = np.random.uniform(0,50,50) /50
Noise = np.random.uniform(0,20,50) /50
Production = (0.4*Soil_Quality + 0.25*Fertilizer + 0.15*(Humidity**2) - 0.25*(Temperature**2) - 0.1*Noise)
Production = np.where(Production < 0 ,0 , Production) 
# print(Production)

# =========================
# 2. Create DataFrame
# =========================

Production_Mango = pd.DataFrame({
                                 "Production" : Production ,
                                 "Soil Quality" : Soil_Quality ,
                                 "Humidity" : Humidity ,
                                 "Temperature" : Temperature  ,
                                 "Fertilizer" : Fertilizer ,
                                 "Noise" : Noise
                                    })
# print(Production_Mango)
# print(Production_Mango.describe())
# print(Production_Mango.head())

# =========================
# 3. Exploratory Data Analysis (EDA)
# =========================

print(Production_Mango.corr())

# Key Insights:
# - Soil Quality is the strongest positive factor affecting production
# - Temperature has a negative impact, indicating non-linear relationship
# - Fertilizer has moderate influence
# - Noise has negligible effect

plt.scatter(Production_Mango["Temperature"], Production_Mango["Production"])
plt.title("Temperature vs Production")
plt.xlabel("Temperature")
plt.ylabel("Production")
plt.show()

plt.scatter(Production_Mango["Humidity"], Production_Mango["Production"])
plt.title("Humidity vs Production")
plt.xlabel("Humidity")
plt.ylabel("Production")
plt.show()

# Observation:
# Temperature has a negative nonlinear relationship with production
# Humidity shows a positive nonlinear effect

# =========================
# 4. Model Training
# =========================

# Feature Engineering to train the nonlinear factor  
Production_Mango["Temperature^2"] = Production_Mango["Temperature"]**2
Production_Mango["Humidity^2"] = Production_Mango["Humidity"]**2

x = Production_Mango.drop("Production" , axis = 1 )
y = Production_Mango["Production"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient")

print(feature_importance)


# Soil Quality is the most influential positive factor affecting Mango production,
# while Temperature has a significant negative impact.
# Noise has minimal effect on the output.

# ============================================

# comparison = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": y_pred
# })

# print(comparison)

# =============================================


# error = y_pred - y_test

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# =========================
# 5. Model Evaluation
# =========================

from sklearn.metrics import r2_score
r2 = r2_score(y_test , y_pred)
print(f"Model R2 Score: {r2:.3f}")

# =========================

print("Conclusion: Soil quality has the strongest impact on Mango production.")




























