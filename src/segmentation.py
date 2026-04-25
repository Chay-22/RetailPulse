import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------- RFM ----------------
def create_rfm(df):
    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm


# ---------------- FIND OPTIMAL K ----------------
def find_optimal_clusters(data, max_k=6):
    distortions = []

    K = range(2, min(max_k, len(data)) + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    if len(distortions) < 2:
        return 1

    # Elbow logic (simple)
    diffs = np.diff(distortions)
    optimal_k = K[np.argmin(diffs)]

    return optimal_k


# ---------------- APPLY KMEANS ----------------
def apply_kmeans(rfm):
    if len(rfm) < 2:
        rfm['Cluster'] = 0
        rfm['Segment'] = "Single Customer"
        return rfm

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    # 🔥 Smart cluster selection
    k = find_optimal_clusters(scaled)

    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    # 🔥 Label clusters
    rfm = label_clusters(rfm)

    return rfm


# ---------------- LABEL SEGMENTS ----------------
def label_clusters(rfm):
    cluster_summary = rfm.groupby('Cluster').mean()

    labels = {}

    for cluster in cluster_summary.index:
        r = cluster_summary.loc[cluster, 'Recency']
        f = cluster_summary.loc[cluster, 'Frequency']
        m = cluster_summary.loc[cluster, 'Monetary']

        if m > cluster_summary['Monetary'].mean() and f > cluster_summary['Frequency'].mean():
            labels[cluster] = "💎 High Value"
        elif r < cluster_summary['Recency'].mean():
            labels[cluster] = "🟢 Active"
        elif r > cluster_summary['Recency'].mean():
            labels[cluster] = "🔴 At Risk"
        else:
            labels[cluster] = "🟡 Regular"

    rfm['Segment'] = rfm['Cluster'].map(labels)

    return rfm

# ---------------- RFM ----------------
def create_rfm(df):
    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm


# ---------------- RFM SCORING ----------------
def calculate_rfm_scores(rfm):
    rfm = rfm.copy()

    # Recency (lower is better → reverse score)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')

    # Frequency & Monetary (higher is better)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

    # Combine
    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str)
    )

    return rfm


# ---------------- SEGMENT BASED ON SCORE ----------------
def assign_segments(rfm):
    def segment(row):
        r = int(row['R_Score'])
        f = int(row['F_Score'])
        m = int(row['M_Score'])

        if r >= 4 and f >= 4:
            return "💎 Champions"
        elif r >= 3 and f >= 3:
            return "🟢 Loyal Customers"
        elif r >= 4:
            return "🆕 New Customers"
        elif r <= 2 and f >= 3:
            return "⚠️ At Risk"
        else:
            return "🟡 Regular"

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm


# ---------------- OPTIONAL CLUSTERING ----------------
def apply_kmeans(rfm):
    if len(rfm) < 2:
        rfm['Cluster'] = 0
        return rfm

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

    k = min(4, len(rfm))
    kmeans = KMeans(n_clusters=k, random_state=42)

    rfm['Cluster'] = kmeans.fit_predict(scaled)
    return rfm


# ---------------- ACTIONS ----------------
def segment_actions(segment):
    actions = {
        "💎 Champions": "Reward them, offer VIP perks, retain strongly.",
        "🟢 Loyal Customers": "Upsell and cross-sell products.",
        "🆕 New Customers": "Provide onboarding offers and engagement.",
        "⚠️ At Risk": "Send re-engagement campaigns and discounts.",
        "🟡 Regular": "Nurture with personalized marketing."
    }
    return actions.get(segment, "General engagement strategy.")