!pip install -q -U google-generativeai

# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Write a story about a magic backpack.")
print(response.text)

mport kagglehub
import os
import pandas as pd

# Last ned datasettet
path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")

# Bruker bitcoin fila
csv_path = os.path.join(path, 'coin_Bitcoin.csv')
print("Path to the CSV file:", csv_path)

# Les CSV-filen
df = pd.read_csv(csv_path)
print(df.head())


#kan renske den litt videre evt. så vi bare har data, high, low, open
heatmap
import matplotlib.pyplot as plt
import seaborn as sns

numeric_df = df.select_dtypes(include=['float64', 'int64'])

corr_matrix = numeric_df.corr()

 
# Creating a seaborn heatmap
plt.figure(figsize=(10, 8))  #adjust the figure size
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='cool', square=True)
plt.title('Correlation Heatmap')
plt.show()

# annen graf
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], color='blue')
plt.title('Bitcoin Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

#videre analyse
# Scatter plots of features vs. close price
plt.figure(figsize=(15, 5))

# High vs. Close
plt.subplot(1, 3, 1)
plt.scatter(df['High'], df['Close'], alpha=0.5, color='red')
plt.title('High vs Close Price')
plt.xlabel('High Price')
plt.ylabel('Close Price')

# Low vs. Close
plt.subplot(1, 3, 2)
plt.scatter(df['Low'], df['Close'], alpha=0.5, color='blue')
plt.title('Low vs Close Price')
plt.xlabel('Low Price')
plt.ylabel('Close Price')

# Volume vs. Close
plt.subplot(1, 3, 3)
plt.scatter(df['Volume'], df['Close'], alpha=0.5, color='green')
plt.title('Volume vs Close Price')
plt.xlabel('Volume')
plt.ylabel('Close Price')

plt.tight_layout()
plt.show()

!pip install xgboost

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import balanced_accuracy_score

# Anta at df er DataFrame med historiske Bitcoin-priser
# Lag en kolonne for prisforandringer
df['Price_Change'] = df['Close'].diff()  # Beregn differansen i lukkekurs

# Kategoriser prisforandringer
def categorize_price_change(change):
    if change > 0:
        return 1  # bullish
    elif change < 0:
        return 0  # bearish
    else:
        return 2  # neutral

df['Investment_Decision'] = df['Price_Change'].apply(categorize_price_change)

# Velg funksjoner (drop unødvendige kolonner)
X = df.drop(columns=['Investment_Decision', 'Price_Change', 'Date'])  # Juster kolonnene etter behov
y = df['Investment_Decision']

# Konvertere kategoriske variabler til numeriske med one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Splitte data i trenings- og testsett
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Steg 3: Trene modellen
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Steg 4: Evaluere modellen
y_pred = model.predict(X_test)

# Skriv ut rapport
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy score: ", balanced_accuracy_score(y_test, y_pred))

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# Initialiser XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [600],
    'learning_rate': [0.4],
    'max_depth': [4],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2)

grid_search.fit(X_train, y_train)

# Beste parametre
print("Best parameters found: ", grid_search.best_params_)
#Best parameters found:  {'learning_rate': 0.4, 'max_depth': 4, 'n_estimators': 600}
# Evaluere den beste modellen
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

!pip install gradio

#gradio interface hvor man kan legge inn vedlegg
import gradio as gr

def count_files(message, history):
    num_files = len(message["files"])
    return f"You uploaded {num_files} files"

demo = gr.ChatInterface(fn=count_files, type="messages", examples=[{"text": "Hello", "files": []}], title="Echo Bot", multimodal=True)

demo.launch()
