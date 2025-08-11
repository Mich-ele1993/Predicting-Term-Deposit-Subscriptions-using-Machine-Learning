# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:17:32 2024

@author: PAGAN
Progetto master gruppo E
"""
##OBIETTIVO
#Un istituto bancario intende migliorare le proprie attività di acquisizione clienti ottimizzando
#le campagne di marketing diretto. La banca dispone di dati storici relativi a precedenti 
#campagne di marketing, tra cui informazioni sui dati demografici dei clienti, 
#sui contatti, sui risultati delle campagne precedenti e sugli indicatori economici. 
#L'obiettivo primario è prevedere se un cliente sottoscriverà un deposito a termine in base 
#alle varie caratteristiche fornite.


#age 
#job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
#marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
#education 
#default: has credit in default? (categorical: “no”, “yes”, “unknown”)
#balance: average yearly balance
#housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
#loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
#contact: contact communication type (categorical: “cellular”, “telephone”)
#month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
#day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
#duration:  last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#previous: number of contacts performed before this campaign and for this client (numeric)
#poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
#y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)

#librerie
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


#visione di tutte le colonne
pd.set_option('display.max_columns', None)

#importo il dataset con i dati
df = pd.read_csv("data.csv")
df.head()

#copio il dataset per mantenere inalterato l'originale
df_d = df.copy()

#CAPIRE I DATI
#età è quantitativa continua
#JOB è una variabile qualitativa nominale
#MARTIAL è una variabile qualitativa nominale
#EDUCATION è una variabile qualitativa ordinale
#DEAFAULT è una variabile qualitativa nominale
#BELANCE è una variabile quantitativa continua
#HOUSING è una variabile qualitativa nominale
#LOAN è una variabile qualitativa nominale
#CONTACT è una variabile qualitativa nominale
#DAY OF WEEK è una variabile qualitativa
#MONTH variabile qualitativa nominale
#DURATION variabile quantitativa continua
#CAMPAIGN variabile quantitativa?
#PDAYS variabile quantitaiva continua
#PREVIOUS variabile quantitativa continua
#POUTCOME variabile qualitativa nominale
#Y variabile binaria

#-------------------------------------------------------------------------------------------------
#PARTE 1 DATA CLEANING

#1
#Iniziamo a vedere se ci sono dei caratteri speciali
caratteri_speciali = df_d.applymap(lambda x: isinstance(x, (str)) and not x.isalnum()).sum()
print(caratteri_speciali)
#solo la variabile job presenta dei caratteri speciali. Li sostituiamo con il trattino basso
#tolgo i caratteri speciali
df_d["job"] = df_d["job"].str.replace(".", "")
df_d["job"] = df_d["job"].str.replace("-", "_")

#2 VALORI DUPLICATI
df_d.duplicated().sum()
#non ci sono valori valori duplicati

#3
#VALORI MANCANTI
df_d.isnull().sum()
#ci alcune variabili con dei dati mancanti ovvero: job, education, contact, poutcome. Andiamo
#ad esplorare tali variabili

#JOB ha 288 dati mancanti e sostituisco i dati mancanti con la modalità più frequente ovvero 
#blue_collar
df_d['job'].value_counts(normalize=True)*100
df_d['job'].fillna('blue_collar', inplace=True)


#EDUCATION ha 1857 valori mancanti e vado a sostituirli con la madalità secondary
df_d['education'].value_counts(normalize=True)*100
df_d['education'].fillna('secondary', inplace=True)


#CONTACT abbiamo 13020 dati mancanti li sostituisco con la moda
df_d['contact'].value_counts(normalize=True)*100
df_d['contact'].fillna('cellular', inplace=True)


#DEFAULT viene eliminato perché ha solo una grande modalità
df_d["default"].value_counts(normalize=True)*100
df_d.drop(["default"], inplace=True, axis=1)
#la variabile default ha il 98% delle osservazioni nella stessa modalità. Dunque non 
#è discriminante.

#VARIABILE DAY OF WEEK
df_d["day_of_week"].value_counts().sort_index()
df_d['day_of_week'].describe()
sns.set(style="whitegrid")
sns.boxplot(df_d['day_of_week'], color='skyblue')
plt.title('giorni')
plt.xlabel('day_of_week')
plt.ylabel('Valori')
plt.show()
#osservando la distribuzione della variabile giorni della settimana, tramite il boxplot, decido 
#di dividere la variabile in tre classi:
#dal giorno 1 al 10: inizio mese
#dal giorno 11-20: metà mese
#dal giorno 21-31: fine mese
bins = [0, 11, 21, 32]  
labels = ['inizio_mese', 'meta_mese', 'fine_mese']  # Etichette per le classi
df_d['day_of_month'] = pd.cut(df_d['day_of_week'], bins=bins, labels=labels, right=False)

df_d.drop(["day_of_week"], inplace=True, axis=1)
df_d['day_of_month'] = df_d['day_of_month'].astype('object')

#faccio un grafico
# Contare il numero di occorrenze per ogni modalità di day_of_month
day_counts = df_d['day_of_month'].value_counts().reset_index()
day_counts.columns = ['day_of_month', 'count']
# Impostazioni estetiche
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
barplot = sns.barplot(x='day_of_month', y='count', data=day_counts, palette="pastel")
barplot.set_xlabel('Giorno del Mese', fontsize=14)
barplot.set_ylabel('Conteggio', fontsize=14)
barplot.set_title('Distribuzione dei Giorni del Mese', fontsize=16)
plt.show()

#ricontrollo tutte le variabili
print(df_d.dtypes)

#analisi variabile POUTCOME
# Troviamo gli indici dove poutcome è NA e pdays è diverso da -1. Con questo script, ci troviamo
#i veri valori mancanti
condition = df_d['poutcome'].isna() & ( df_d['pdays'] != -1)
result = pd.DataFrame({'pdays': df_d['pdays'][condition], 'poutcome': df_d['poutcome'][condition]})
print(result)
#queste 5 righe rappresentano i veri NA
#individuo questi 5 valori trovati nel dataset
filtered_df = df_d[condition]
print(filtered_df)

# Trasformiamo i NaN in una categoria separata per poterli visualizzare nel plot
df_d['poutcome'] = df_d['poutcome'].fillna('NaN')
#Strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(x='poutcome', y='pdays', data=df_d, jitter=True, palette='pastel')
plt.title('Strip Plot')
plt.xlabel('poutcome')
plt.ylabel('pdays')
plt.show()

#sostituisco tutti gli NA della variabile poutcome con il valore no_poutcome, perche abbiamo 
#capito che in poutcome i valori mancanti ci sono perche il cliente non ha mai partecipato 
#ad una campagna precedente. Questo ad eccezione di 5 valori NA che sono davvero tali perche 
#non coincidono con i -1 della variabile pdays. Così, creamo una nuovo modalità della variabile
#poutcome
exclude_rows = [40658, 41821, 42042, 43978, 45021]
condition = (df_d.index.isin(exclude_rows) == False) & (df_d['poutcome'] == 'NaN')
df_d.loc[condition, 'poutcome'] = 'no_poutcome'
df_d['poutcome'] = df_d['poutcome'].replace('NaN', np.nan)

#elimino direttamente i 5 valori mancanti
df_d = df_d.dropna(subset=['poutcome'])


#serve per salvare il dataset modificato in excel
#df_d.to_excel('nuovo_bank.xlsx', index=False)

df_p = df_d.copy()

#elimino duration perche dalla descrizione delle variabili mi consiglia di non utilizzarla
#in un modello predittivo
df_p.drop(["duration"], inplace=True, axis=1)

#-------------------------------------------------------------------------------------------------
#PARTE 2 STATISTICHE UNIVARIATE
#variabile Y
df_p['y'].value_counts(normalize=True)*100
conteggio_y = df_p["y"].value_counts()
plt.figure(figsize=(8,4), dpi=100)
plt.style.use("ggplot")
plt.pie(conteggio_y, labels=['no','si'], wedgeprops={'edgecolor':'#000000'},
        explode = (0.1, 0), autopct='%1.1f%%')
plt.title("Distribuzione della variabile Y")
plt.show()
#dal grafico a torto osserviamo come la nostra variabile target presenta una distribuzione
#molto asimmetrica.

#VARIABILI NUMERICHE
numeric_df = df_p.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()
#grafico della matrice
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
# Inserisci i valori della matrice di correlazione
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        plt.text(j+0.5, i+0.5, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='white')
plt.title('Matrice di Correlazione')
plt.show()
#dalla matrice di correlazione vediamo che le variabili numeriche non sono correlate.

#faccio un describe delle variabili numeriche
numeric_df.describe()
#l'età media degli individui è 41 anni
#la durata media delle telefonate è 258 secondi che sono all'incirca 4 minuti
#la media del bilancio è pari a 1362$
#in questa campagna ci sono stati mediamente quasi 3 contatti tra la banca e i clienti
#mediamente se passati quasi 40 giorni dalla campagan precedente
#mediamente prima di questa campagna, i clienti sono stai contattati 0.58 volte

#faccio i box plot delle variabili numeriche

colors = sns.color_palette("husl", len(numeric_df))
for col, color in zip(numeric_df, colors):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df_p[col], palette="husl")
    plt.title(f'Boxplot della colonna {col}')
    plt.ylabel(col)
    plt.show()
    
#VARIABILI QUALITATIVE
cat_df = df_p.select_dtypes(include='object')
# Crea un grafico a barre per ogni colonna qualitativa con colori differenti
for col, color in zip(cat_df, colors):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df_p[col], palette="husl")
    plt.title(f'Grafico a barre della colonna {col}')
    plt.xlabel(col)
    plt.ylabel('Conteggio')
    plt.show()
    
#---------------------------------------------------------------------------------------------------------
#PARTE 3: IMPLEMENTAZIONE DEL MODELLO: RETE NEURALE
#CODIFICA DELLE VARIABILI

#codifico le variabili binarie
df_p['housing'] = df_p['housing'].apply(lambda x: 1 if x == 'yes' else 0)
df_p['loan'] = df_p['loan'].apply(lambda x: 1 if x == 'yes' else 0)
df_p['contact'] = df_p['contact'].apply(lambda x: 1 if x == 'cellular' else 0)
#codifico la variabile ordinale education
df_p['education'] = df_p['education'].map({'primary': 0, 'secondary': 1, 'tertiary': 2})

#dicotomizzazione variabili
df_onehot = pd.get_dummies(df_p['job'], prefix='jo', dtype="int")
df_p = pd.concat([df_p, df_onehot], axis=1)
df_p = df_p.drop('job', axis=1)

df_onehot = pd.get_dummies(df_p['marital'], prefix='ma', dtype="int")
df_p = pd.concat([df_p, df_onehot], axis=1)
df_p = df_p.drop('marital', axis=1)

df_onehot = pd.get_dummies(df_p['month'], prefix='mo', dtype="int")
df_p = pd.concat([df_p, df_onehot], axis=1)
df_p = df_p.drop('month', axis=1)

df_onehot = pd.get_dummies(df_p['day_of_month'], prefix='da', dtype="int")
df_p = pd.concat([df_p, df_onehot], axis=1)
df_p = df_p.drop('day_of_month', axis=1)

df_onehot = pd.get_dummies(df_p['poutcome'], prefix='po', dtype="int")
df_p = pd.concat([df_p, df_onehot], axis=1)
df_p = df_p.drop('poutcome', axis=1)

#TRAIN E TEST

#dividiamo in x e y
x = df_p.drop('y', axis=1)
y = df_p['y']

#scaliamo i dati in modo tale da non avere problemi con le unita di misura
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_array = scaler.fit_transform(x)
x_sc = pd.DataFrame(scaled_array, columns=x.columns)

#divido il mio dataset in train e test utilizzando i dati scalati
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_sc, y, test_size=0.25, 
                                                    random_state=42, 
                                                    stratify=y)
#stratify mantiene le stesse percentuali di modalità della variabili target

#RETE NEURALE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


model = MLPClassifier(random_state=42, activation='relu', alpha=0.1, early_stopping=True, batch_size= 64,
                      validation_fraction=0.1, n_iter_no_change=10) 

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#MATRICE DI CONFUSIONE
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
ax = sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='coolwarm', xticklabels=['no', 'yes'], 
                 yticklabels=['no', 'yes'], vmin=0, vmax=conf_matrix.max())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# Annotazioni manuali dei valori
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        text = ax.text(j + 0.5, i + 0.5, format(conf_matrix[i, j], 'd'),
                       ha='center', va='center', color='black', fontsize=20)
plt.show()

#parametri
accuracy_score(y_test, y_pred) 
#l'accuracy è del 89.6%
precision_score(y_test, y_pred, pos_label='yes')
#la precision è del 65.2%
recall_score(y_test, y_pred, pos_label='yes')
#la recall e del 24.1%, quindi non è accettabile ma è dovuto dallo sbilanciamento della
#variabile y

from sklearn.metrics import roc_curve, auc, roc_auc_score

# Calcolo delle probabilità previste dal modello
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Calcolo della curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='yes')

# Calcolo dell'AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot della curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Stampa dell'AUC
print(f'AUC: {roc_auc:.2f}')
#l'auc score è del 78%

#CONTROLLO OVERFITTING
# Calcolo delle metriche per il set di addestramento
y_train_pred = model.predict(x_train)
y_train_proba = model.predict_proba(x_train)[:, 1]

# Precisione sul set di addestramento
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, pos_label='yes')
train_recall = recall_score(y_train, y_train_pred, pos_label='yes')
train_auc = roc_auc_score(y_train, y_train_proba)

# Precisione sul set di test
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, pos_label='yes')
test_recall = recall_score(y_test, y_pred, pos_label='yes')
test_auc = roc_auc_score(y_test, y_pred_proba)

# Stampa dei risultati
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}\n")

print(f"Training Precision: {train_precision:.2f}")
print(f"Test Precision: {test_precision:.2f}\n")

print(f"Training Recall: {train_recall:.2f}")
print(f"Test Recall: {test_recall:.2f}\n")

print(f"Training AUC: {train_auc:.2f}")
print(f"Test AUC: {test_auc:.2f}")

# Visualizzazione delle curve ROC per entrambi i set di dati
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba, pos_label='yes')
test_fpr, test_tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='yes')

plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, color='blue', lw=2, label=f'Train ROC curve (area = {train_auc:.2f})')
plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
#Basandosi sulle curve ROC e sui valori di AUC:
#La differenza minima tra le due AUC (0.79 per il train e 0.78 per il test) indica che il modello 
#generalizza bene e non sembra sovradattato.
#Le curve ROC che si sovrappongono in gran parte confermano che il modello ha prestazioni simili 
#sui dati di addestramento e sui dati di test.
#Quindi, possiamo concludere che non c'è evidenza significativa di overfitting nel modello basato su 
#queste analisi. Non c'è overfitting.

# Ottieni i coefficienti dei pesi
weights = model.coefs_

# Calcola l'importanza delle variabili considerando la media dei pesi per ciascuna variabile
feature_importance = np.mean(np.abs(weights[0]), axis=1)

# Visualizza le variabili più importanti
importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

top_10_importance = importance_df.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_10_importance, palette='viridis')
plt.title('Top 10 Variabili più importanti nel modello di rete neurale')
plt.xlabel('Importanza')
plt.ylabel('Variabile')
plt.show()
#le variabili più importanti sono:
#1)inzio mese
#2)mese febbraio
#3)housing

















