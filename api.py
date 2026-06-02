"""
Smart Reassort — API FastAPI
==============================
Endpoints :
  GET /                    → Info de l'API
  GET /stats               → KPIs (stats globales)
  GET /reassort            → Tableau complet de réassort (filtrable)
  GET /reassort/{product_id} → Détail d'un produit
  GET /alerts              → Produits en alerte uniquement
  GET /obsolescence        → Produits obsolescents

Usage :
    uvicorn api:app --reload --port 8000
    → http://localhost:8000/docs (Swagger UI)
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import time
import os
import boto3
from io import StringIO

DATA_DIR = "data"

def get_reassort_df():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    obj = s3.get_object(
        Bucket=os.environ["AWS_BUCKET_NAME"],
        Key="temp/tp-data-eng/save_pkl/reassort_output.csv"
    )
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

def get_category_list():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    obj = s3.get_object(
        Bucket=os.environ["AWS_BUCKET_NAME"],
        Key="temp/tp-data-eng/save_pkl/sales_step4_b2c.csv"
    )
    df =  pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    
    data = df[["category_id", "category"]]
    return data.drop_duplicates().to_dict(orient="records")


# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════
app = FastAPI(
    title="Smart Reassort API",
    description="API de prédiction de réassort intelligent basée sur XGBoost",
    version="1.0.0",
)

# CORS (pour que le frontend puisse appeler l'API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
# CACHE DU RÉASSORT (recalculé au démarrage)
# ══════════════════════════════════════════════
reassort_df: pd.DataFrame = pd.DataFrame()


@app.on_event("startup")
def startup():
    """Calcule le réassort au démarrage de l'API."""
    global reassort_df
    global category_list
    start_time = time.time()
    reassort_df = get_reassort_df()
    category_list = get_category_list()
    end_time = time.time()
    print(f"✓ time taken : {end_time - start_time} sec")
    print(f"✓ Réassort calculé : {len(reassort_df)} produits")


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def df_to_records(df: pd.DataFrame) -> list:
    """Convertit un DataFrame en liste de dicts pour JSON."""
    return df.replace({float("inf"): None, float("-inf"): None}).fillna("").to_dict(orient="records")


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/")
def root():
    """Info de l'API."""
    return {
        "name": "Smart Reassort API",
        "version": "1.0.0",
        "model": "XGBoost 30 jours",
        "products": len(reassort_df),
        "endpoints": ["/stats", "/reassort", "/reassort/{product_id}", "/alerts", "/obsolescence"],
    }


@app.get("/up")
def uptime():
    """Uptime de l'API."""
    return {
        "status": "ok",
        "message": "Prophecy FastAPI is running",
    }

@app.get("/categories")
def get_categories():
    """Liste des catégories."""
    return {
        "data": category_list,
    }

@app.get("/stats")
def get_stats():
    """KPIs globaux du réassort."""
    df = reassort_df

    # Alertes
    alerts = df["alert"].value_counts().to_dict()

    # Cycle de vie
    cycle = df["cycle_status"].value_counts().to_dict()

    # Chiffres clés
    n_to_order = int((df["qty_to_order"] > 0).sum())
    total_cost = float(df["estimated_cost"].sum())
    n_rupture = int((df["alert"] == "Rupture imminente").sum())
    n_forte_demande = int((df["alert"] == "Forte demande").sum())
    n_obsolete = int(df["cycle_status"].isin(["Obsolescence", "Inactif"]).sum())
    avg_coverage = float(df[df["coverage_days"] < 999]["coverage_days"].mean()) if (df["coverage_days"] < 999).any() else 0

    return {
        "total_products": len(df),
        "products_to_order": n_to_order,
        "estimated_total_cost": round(total_cost, 0),
        "rupture_imminente": n_rupture,
        "forte_demande": n_forte_demande,
        "obsolete": n_obsolete,
        "avg_coverage_days": round(avg_coverage, 1),
        "alerts": alerts,
        "cycle_status": cycle,
    }


@app.get("/reassort")
def get_reassort(
    alert: Optional[str] = Query(None, description="Filtrer par alerte (Rupture imminente, Forte demande, À commander, Stable, Stock OK, Ne pas recommander)"),
    cycle: Optional[str] = Query(None, description="Filtrer par cycle (Croissance, Maturité, Déclin, Obsolescence, Inactif)"),
    category_id: Optional[int] = Query(None, description="Filtrer par category_id"),
    min_qty: Optional[int] = Query(None, description="Quantité à commander minimum"),
    limit: int = Query(100, description="Nombre max de résultats"),
    offset: int = Query(0, description="Offset pour pagination"),
):
    """Tableau complet de réassort avec filtres."""
    df = reassort_df.copy()

    # Filtres
    if alert:
        df = df[df["alert"] == alert]
    if cycle:
        df = df[df["cycle_status"] == cycle]
    if category_id:
        df = df[df["category_id"] == category_id]
    if min_qty is not None:
        df = df[df["qty_to_order"] >= min_qty]

    total = len(df)
    df = df.iloc[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": df_to_records(df),
    }

@app.get("/grouped-reassort")
def get_grouped_reassort():
    """Réassort groupé par alerte."""
    df = reassort_df.copy()
    df = df[['alert','qty_to_order']].groupby("alert").agg({
        "qty_to_order": "sum",
    }).reset_index().sort_values("qty_to_order", ascending=False)
    return {
        "data": df_to_records(df),
    }



@app.get("/reassort/{product_id}")
def get_product_reassort(product_id: int):
    """Détail du réassort pour un produit spécifique."""
    df = reassort_df[reassort_df["product_id"] == product_id]

    if df.empty:
        return {"error": f"Produit {product_id} non trouvé"}

    row = df.iloc[0].to_dict()

    # Remplacer inf/nan
    for k, v in row.items():
        if isinstance(v, float) and (pd.isna(v) or v == float("inf")):
            row[k] = None

    return row


@app.get("/alerts")
def get_alerts(
    level: Optional[str] = Query(None, description="Filtrer: Rupture imminente, Forte demande, À commander"),
):
    """Produits en alerte uniquement (à commander)."""
    df = reassort_df[reassort_df["qty_to_order"] > 0].copy()

    if level:
        df = df[df["alert"] == level]

    return {
        "total": len(df),
        "data": df_to_records(df),
    }


@app.get("/obsolescence")
def get_obsolescence(
    min_score: float = Query(0.5, description="Score minimum d'obsolescence (0-1)"),
):
    """Produits à risque d'obsolescence."""
    df = reassort_df[reassort_df["obsolescence_score"] >= min_score].copy()
    df = df.sort_values("obsolescence_score", ascending=False)

    return {
        "total": len(df),
        "threshold": min_score,
        "data": df_to_records(df),
    }


@app.post("/refresh")
def refresh_reassort():
    """Recalcule le réassort (après mise à jour des données)."""
    global reassort_df
    reassort_df = get_reassort_df()
    return {
        "status": "ok",
        "products": len(reassort_df),
        "message": "Réassort recalculé",
    }