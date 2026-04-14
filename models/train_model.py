import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Veriyi Yükle
df = pd.read_csv("data/istanbul_apartment_prices_2026.csv")

# 2. Gereksiz Sütunları At ve Hedef Belirle
# Modelin öğrenmesi için en kritik özellikleri seçiyoruz
features = ['district', 'neighborhood', 'rooms', 'halls', 'gross_sqm', 'building_age', 'floor', 'total_floors']
X = df[features].copy()
y = df['price']

# 3. Metinleri Sayıya Çevir (Encoding)
le_dict = {}
for col in ['district', 'neighborhood']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le # Daha sonra app.py'da kullanmak için saklıyoruz

# 4. Veriyi Eğitime ve Teste Böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. XGBoost Modelini Eğit
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# 6. Modeli ve Encoder'ları Kaydet
joblib.dump(model, "models/xgboost_model.joblib")
joblib.dump(le_dict, "models/encoders.joblib")

print("✅ Model başarıyla eğitildi ve models/ klasörüne kaydedildi!")
