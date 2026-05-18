import os
import sys
import time
import pathlib

import torch
import psycopg2
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Forcer l'encodage UTF-8 ───────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Charger les variables d'environnement ─────────────────────────────────────
current_dir = pathlib.Path(__file__).parent
root_dir = current_dir.parent
dotenv_path = root_dir / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path, encoding="utf-8")
else:
    load_dotenv(encoding="utf-8")

# ── Modèle Hugging Face ───────────────────────────────────────────────────────
MODEL_ID = "Riles14m/roberta-fake-news"

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost").strip()
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "fake_news_project").strip()
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres").strip()
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "").strip()

# ── Device (GPU ou CPU) ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device utilisé : {device}")


# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
def get_connection():
    """Retourne une connexion psycopg2."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def load_posts_from_db(conn, limit=None):
    """Récupère les posts en anglais depuis la base de données."""
    query = "SELECT id, text FROM raw_posts WHERE lang = 'en' ORDER BY id DESC"
    if limit is not None:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    print(f"📥 {len(df)} posts chargés depuis PostgreSQL")
    return df


def load_model_and_tokenizer():
    """Charge le modèle fine-tuné et le tokenizer depuis Hugging Face."""
    print(f"⏳ Chargement du modèle depuis Hugging Face : {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    print("✅ Modèle et tokenizer chargés")
    return model, tokenizer


class PostDataset(Dataset):
    """Dataset pour les posts."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def classify_posts(model, tokenizer, texts, batch_size=32):
    """Classifie les posts avec le modèle fine-tuné."""
    print(f"\n🚀 Démarrage de la classification ({len(texts)} posts)...")

    dataset = PostDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classification"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            confidences.extend(probs.max(dim=-1).values.cpu().numpy())

    print("✅ Classification terminée")
    return predictions, confidences


def save_results(df, predictions, confidences, model, output_file="classification_results.csv"):
    """Sauvegarde les résultats en CSV."""
    df["prediction"] = predictions
    df["confidence"] = confidences

    # Utiliser les labels du modèle si disponibles
    if hasattr(model.config, "id2label") and model.config.id2label:
        label_map = {int(k): v for k, v in model.config.id2label.items()}
    else:
        label_map = {0: "Not Fake News", 1: "Fake News"}

    df["label"] = df["prediction"].map(lambda x: label_map.get(int(x), str(x)))

    df.to_csv(output_file, index=False)
    print(f"💾 Résultats sauvegardés : {output_file}")

    print("\n📊 Statistiques :")
    print(df["label"].value_counts())
    print(f"\n⏳ Confiance moyenne : {df['confidence'].mean():.4f}")


def main():
    """Pipeline de classification."""
    try:
        # Charger le modèle
        model, tokenizer = load_model_and_tokenizer()

        # Charger les données
        conn = get_connection()
        df = load_posts_from_db(conn, limit=None)
        conn.close()

        if len(df) == 0:
            print("⚠️  Aucun post trouvé dans la base de données!")
            return

        # Green IT - dossier de sortie
        green_it_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "green_it_results")
        os.makedirs(green_it_dir, exist_ok=True)

        tracker = EmissionsTracker(
            project_name="fake_news_classification",
            output_dir=green_it_dir,
            output_file="classification_emissions.csv",
        )

        tracker.start()
        start_time = time.time()

        # Classifier
        predictions, confidences = classify_posts(model, tokenizer, df["text"].tolist())

        end_time = time.time()
        emissions = tracker.stop()
        duration = end_time - start_time

        print(f"⏱️ Temps de classification : {duration:.2f} secondes")
        print(f"🌱 Émissions estimées : {emissions:.6f} kg CO2eq")
        print(f"⚡ Temps moyen par post : {duration / len(df):.4f} secondes/post")

        # Sauvegarder les résultats
        save_results(df, predictions, confidences, model)

        print("\n🏁 Pipeline de classification terminé avec succès!")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()