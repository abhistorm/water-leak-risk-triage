"""Train the Water Leak Risk Triage baseline on its bundled synthetic dataset."""
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = json.loads((ROOT / 'project.json').read_text(encoding='utf-8'))
FEATURES = PROJECT['features']
TARGET = PROJECT['target']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data' / 'dataset.csv'))
    parser.add_argument('--limit', type=int, default=20000, help='Rows to use for a quick local run')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.data, nrows=args.limit)
    X = frame[FEATURES]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Project: {PROJECT['name']}")
    print(f'train_rows={len(X_train):,}  test_rows={len(X_test):,}')
    print(f'model={model.__class__.__name__}')
    # Task-appropriate holdout metrics
    print(f"accuracy={accuracy_score(y_test, pred):.4f}")
    print(f"macro_f1={f1_score(y_test, pred, average='macro'):.4f}")
    if hasattr(model, "predict_proba") and y.nunique() == 2:
        try: print(f"roc_auc={roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
        except ValueError: pass
    out = ROOT / 'models' / 'model.joblib'
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'target': TARGET, 'task_type': PROJECT['task_type'], 'project': PROJECT['name']}, out)
    print(f'Saved model: {out}')

if __name__ == '__main__':
    main()
