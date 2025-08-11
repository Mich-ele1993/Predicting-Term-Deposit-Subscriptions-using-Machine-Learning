# Predicting Term Deposit Subscriptions using Machine Learning

## 📌 Obiettivo
Un istituto bancario vuole migliorare le proprie campagne di marketing diretto prevedendo se un cliente sottoscriverà un **deposito a termine** in base alle sue caratteristiche demografiche, finanziarie e alle informazioni delle campagne precedenti.

## 📂 Dataset
Il dataset contiene informazioni su:
- **Dati demografici** (età, lavoro, stato civile, istruzione)
- **Informazioni finanziarie** (saldo medio, mutuo, prestiti)
- **Dettagli di contatto** (tipo, mese e giorno di contatto)
- **Risultati precedenti delle campagne**
- **Variabile target:** `y` (1 = sottoscrive, 0 = non sottoscrive)

> Il dataset originale non è incluso nel repository per motivi di dimensione/privacy.  
> Può essere scaricato dalla [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) o da fonti equivalenti.

## 🛠️ Metodologia
1. **Pulizia dati**
   - Gestione valori mancanti con imputazione e rimozione di variabili poco informative (`default`)
   - Trasformazione variabile `day_of_week` in `day_of_month` (inizio, metà, fine mese)
   - Eliminazione della variabile `duration` per evitare bias predittivo

2. **Analisi esplorativa**
   - Statistiche descrittive di variabili numeriche e categoriche
   - Analisi della distribuzione della variabile target
   - Matrice di correlazione e visualizzazioni con Seaborn/Matplotlib

3. **Preprocessing**
   - Codifica binaria per variabili `yes/no`
   - Codifica ordinale per `education`
   - One-Hot Encoding per variabili categoriche
   - Standardizzazione con `StandardScaler`

4. **Modellazione**
   - Modello: **Rete neurale MLPClassifier (Scikit-learn)**
   - Suddivisione train/test: 75% / 25% (stratificata)
   - Metriche: Accuracy, Precision, Recall, AUC-ROC
   - Controllo overfitting con confronto ROC train/test

5. **Risultati principali**
   - Accuracy: **89.6%**
   - Precision (classe positiva): **65.2%**
   - Recall (classe positiva): **24.1%** (basso a causa dello sbilanciamento della classe)
   - AUC-ROC: **0.78**
   - Variabili più importanti: `inizio_mese`, `mese_febbraio`, `housing`

## 📊 Visualizzazioni
- Grafici a barre e torte per la distribuzione delle variabili
- Boxplot per variabili numeriche
- Matrice di correlazione
- Curva ROC per train e test
- Importanza delle feature nel modello di rete neurale

## 🚀 Requisiti
Per eseguire il progetto:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## ▶️ Esecuzione
1. Scaricare il dataset e salvarlo come `data.csv` nella cartella del progetto
2. Eseguire lo script:
```bash
python progetto.py
```

## 📄 Licenza
MIT License - libera per uso e modifiche con attribuzione.
