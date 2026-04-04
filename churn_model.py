import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import shap
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. LOAD
# =============================================================================
print("=" * 60)
print("The Retention Architect")
print("=" * 60)
print("\n[1/7] Loading datasets...")

# ---------------------------------------------------------------------------
# Auto-detect CSV files by their column signatures — not by filename.
# This works regardless of what the files are named.
# ---------------------------------------------------------------------------

# Signature: unique set of columns that identifies each dataset
SIGNATURES = {
    "users":   {"churn_status"},
    "gens":    {"generation_id", "credit_cost", "generation_type"},
    "props":   {"subscription_start_date", "subscription_plan"},
    "purch":   {"purchase_type", "purchase_amount_dollars"},
    "txn":     {"failure_code", "card_brand", "card_funding"},
    "quizzes": {"frustration", "first_feature", "flow_type"},
}

import glob, os

def detect_datasets(folder="."):
    """Scan all CSVs in folder and match each to a dataset role by column signature."""
    found = {}
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for path in csv_files:
        try:
            cols = set(pd.read_csv(path, nrows=0, low_memory=False).columns)
        except Exception:
            continue
        for role, sig in SIGNATURES.items():
            if sig.issubset(cols) and role not in found:
                found[role] = path
                break
    return found

detected = detect_datasets(".")
print("  Detected files:")
for role, path in detected.items():
    print(f"    {role:10s} -> {os.path.basename(path)}")

missing = [r for r in SIGNATURES if r not in detected]
if missing:
    print(f"\n  WARNING: Could not detect: {missing}")
    print("  Make sure all dataset CSV files are in the same folder as this script.")
    raise FileNotFoundError(f"Missing datasets: {missing}")

users   = pd.read_csv(detected["users"],   low_memory=False)
gens    = pd.read_csv(detected["gens"],    low_memory=False)
props   = pd.read_csv(detected["props"],   low_memory=False)
purch   = pd.read_csv(detected["purch"],   low_memory=False)
txn     = pd.read_csv(detected["txn"],     low_memory=False)
quizzes = pd.read_csv(detected["quizzes"], low_memory=False)

print(f"\n  users={users.shape} props={props.shape} purch={purch.shape}")
print(f"  txn={txn.shape} quizzes={quizzes.shape} gens={gens.shape}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[2/7] Engineering features...")

# ---------------------------------------------------------------------------
# GENERATIONS (days 0-14)
# Each failing/nsfw generation adds frustration — snowball effect
# ---------------------------------------------------------------------------
gens["created_at"] = pd.to_datetime(gens["created_at"], errors="coerce", utc=True)
gens["credit_cost"] = pd.to_numeric(gens["credit_cost"], errors="coerce").fillna(0)

# Resolution quality rank: 1080p > 720p > 480p > unknown
RESOLUTION_RANK = {"1080": 3, "1080p": 3, "720": 2, "720p": 2, "480": 1, "480p": 1}
gens["resolution_rank"] = gens["resolution"].map(RESOLUTION_RANK).fillna(0)

gen_agg = gens.groupby("user_id").agg(
    gen_count        = ("generation_id",   "count"),
    completed_count  = ("status",          lambda x: (x == "completed").sum()),
    failed_count     = ("status",          lambda x: (x == "failed").sum()),
    nsfw_count       = ("status",          lambda x: (x == "nsfw").sum()),
    avg_credit_cost  = ("credit_cost",     "mean"),
    total_credits    = ("credit_cost",     "sum"),
    unique_models    = ("generation_type", "nunique"),
    avg_resolution   = ("resolution_rank", "mean"),   # higher = uses better quality
    max_resolution   = ("resolution_rank", "max"),
).reset_index()

gen_agg["completion_rate"] = gen_agg["completed_count"] / gen_agg["gen_count"].clip(1)
gen_agg["nsfw_rate"]       = gen_agg["nsfw_count"]      / gen_agg["gen_count"].clip(1)
gen_agg["fail_rate_gen"]   = gen_agg["failed_count"]    / gen_agg["gen_count"].clip(1)

# Frustration score: each nsfw/failed adds to a running "snowball"
# Higher = more accumulated frustration in first 14 days
gen_agg["frustration_score"] = (
    gen_agg["nsfw_count"] * 1.5 + gen_agg["failed_count"] * 1.0
) / gen_agg["gen_count"].clip(1)

# Activity trend: only reliable when gen_count >= 6
def compute_trend(group):
    if len(group) < 6:
        return 1.0   # neutral for sparse data
    group = group.sort_values("created_at")
    mid   = len(group) // 2
    return len(group.iloc[mid:]) / max(len(group.iloc[:mid]), 1)

gen_trend = (
    gens.groupby("user_id")
    .apply(compute_trend)
    .reset_index(name="gen_trend")
)
gen_agg = gen_agg.merge(gen_trend, on="user_id", how="left")

# Days before first generation (activation speed)
# Long delay = bad onboarding = higher churn risk
props_dates = props[["user_id", "subscription_start_date"]].copy()
props_dates["subscription_start_date"] = pd.to_datetime(
    props_dates["subscription_start_date"], errors="coerce", utc=True
)
first_gen = gens.groupby("user_id")["created_at"].min().reset_index(name="first_gen_date")
activation = props_dates.merge(first_gen, on="user_id", how="left")
activation["days_to_first_gen"] = (
    activation["first_gen_date"] - activation["subscription_start_date"]
).dt.days.fillna(14).clip(0, 14)   # cap at 14 (observation window)
gen_agg = gen_agg.merge(activation[["user_id", "days_to_first_gen"]], on="user_id", how="left")

# Used generation types per user (for vol offer logic)
used_types_map = (
    gens.groupby("user_id")["generation_type"]
    .apply(lambda x: list(x.dropna().unique()))
    .to_dict()
)

# ---------------------------------------------------------------------------
# PURCHASES (days 0-14)
# More spending = more invested in platform = retention signal
# credit_packs > 0 in first 14 days = active, spending = POSITIVE signal
# ---------------------------------------------------------------------------
PLAN_TIER = {
    "Higgsfield Basic": 1, "Higgsfield Pro": 2,
    "Higgsfield Creator": 3, "Higgsfield Ultimate": 4,
    "Higgsfield Teams": 3,
}
props["plan_tier"] = props["subscription_plan"].map(PLAN_TIER).fillna(1)

purch_agg = purch.groupby("user_id").agg(
    total_purchases = ("transaction_id",        "count"),
    lifetime_spend  = ("purchase_amount_dollars","sum"),
    avg_purchase    = ("purchase_amount_dollars","mean"),
    sub_creates     = ("purchase_type", lambda x: (x == "Subscription Create").sum()),
    sub_updates     = ("purchase_type", lambda x: (x == "Subscription Update").sum()),
    credit_packs    = ("purchase_type", lambda x: (x == "Credits package").sum()),
).reset_index()
# credit_packs = positive engagement signal (buying more = using more)
# higher plan = more committed

# ---------------------------------------------------------------------------
# TRANSACTIONS (days 0-14)
# Payment health + card metadata
# ---------------------------------------------------------------------------

# Bank country development tier
# Developed regions → more reliable payments → higher retention
DEVELOPED   = {"US","GB","DE","FR","NL","SE","NO","DK","FI","CH","AT","BE","AU","NZ","CA","JP","SG","KR","HK","IE","LU","IS"}
AVERAGE     = {"BR","MX","AR","CL","CO","ZA","TR","PL","CZ","HU","RO","GR","PT","IL","AE","SA","MY","TH","PH","ID","IN","CN"}
# Everything else = lower tier

def country_tier(c):
    if c in DEVELOPED:
        return 2
    if c in AVERAGE:
        return 1
    return 0

txn["bank_country_tier"]    = txn["bank_country"].apply(country_tier)
txn["billing_country_tier"] = txn["billing_address_country"].apply(country_tier)

txn_agg = txn.groupby("user_id").agg(
    txn_count            = ("transaction_id",  "count"),
    fail_count           = ("failure_code",    lambda x: x.notna().sum()),
    total_spend_txn      = ("amount_in_usd",   "sum"),
    avg_spend_txn        = ("amount_in_usd",   "mean"),
    bank_country_tier    = ("bank_country_tier",    "max"),
    billing_country_tier = ("billing_country_tier", "max"),
    is_business          = ("is_business",     lambda x: x.astype(str).str.lower().eq("true").any()),
    is_virtual           = ("is_virtual",      lambda x: x.astype(str).str.lower().eq("true").any()),
    has_prepaid          = ("is_prepaid",      lambda x: x.astype(str).str.lower().eq("true").any()),
    has_debit            = ("card_funding",    lambda x: x.eq("debit").any()),
).reset_index()

txn_agg["payment_fail_rate"] = txn_agg["fail_count"] / txn_agg["txn_count"].clip(1)
txn_agg["is_business"]  = txn_agg["is_business"].astype(int)
txn_agg["is_virtual"]   = txn_agg["is_virtual"].astype(int)
txn_agg["has_prepaid"]  = txn_agg["has_prepaid"].astype(int)
txn_agg["has_debit"]    = txn_agg["has_debit"].astype(int)

# Card country mismatch
txn["mismatch"] = (txn["billing_address_country"] != txn["card_country"]).astype(int)
mismatch_agg = txn.groupby("user_id")["mismatch"].max().reset_index(name="card_country_mismatch")
txn_agg = txn_agg.merge(mismatch_agg, on="user_id", how="left")

# High-risk country flag (countries with above-average failure rate in your data)
country_fail = (
    txn.groupby("billing_address_country")["failure_code"]
    .apply(lambda x: x.notna().mean())
    .reset_index(name="country_fail_rate")
)
high_risk_countries = set(
    country_fail.loc[
        country_fail["country_fail_rate"] > country_fail["country_fail_rate"].mean(),
        "billing_address_country"
    ]
)
txn["high_risk_country"] = txn["billing_address_country"].isin(high_risk_countries).astype(int)
hr_agg = txn.groupby("user_id")["high_risk_country"].max().reset_index()
txn_agg = txn_agg.merge(hr_agg, on="user_id", how="left")

# ---------------------------------------------------------------------------
# QUIZZES (day 0 — onboarding)
# ---------------------------------------------------------------------------
quizzes_dedup = quizzes.drop_duplicates(subset="user_id", keep="first")

HIGH_COST_VALUES    = {"high-cost", "High cost of top models"}
HARD_PROMPT_VALUES  = {"hard-prompt", "Hard to prompt", "confusing", "AI is confusing to me"}
INCONSISTENT_VALUES = {"inconsistent", "Inconsistent results"}
LIMITED_VALUES      = {"limited", "Limited generations"}

quiz_agg = quizzes_dedup[["user_id"]].copy()
quiz_agg["quiz_high_cost"]    = quizzes_dedup["frustration"].isin(HIGH_COST_VALUES).astype(int)
quiz_agg["quiz_hard_prompt"]  = quizzes_dedup["frustration"].isin(HARD_PROMPT_VALUES).astype(int)
quiz_agg["quiz_inconsistent"] = quizzes_dedup["frustration"].isin(INCONSISTENT_VALUES).astype(int)
quiz_agg["quiz_limited"]      = quizzes_dedup["frustration"].isin(LIMITED_VALUES).astype(int)
quiz_agg["is_beginner"]       = (quizzes_dedup["experience"] == "beginner").astype(int)
quiz_agg["is_expert"]         = (quizzes_dedup["experience"].isin(["advanced", "expert"])).astype(int)
quiz_agg["is_invited"]        = (quizzes_dedup["flow_type"]  == "invited").astype(int)  # invited > personal
quiz_agg["is_personal"]       = (quizzes_dedup["flow_type"]  == "personal").astype(int)
quiz_agg["wants_video"]       = (quizzes_dedup["first_feature"].str.contains(
    "video|Video|cinema|Cinema|commercial|Commercial|viral|Viral", na=False)).astype(int)
quiz_agg["is_filmmaker_role"] = (quizzes_dedup["role"].str.contains(
    "film|Film|cinema|Cinema|creator|Creator|director|Director", na=False)).astype(int)
team_map = {"1": 1, "small": 3, "growing": 10, "midsize": 30, "enterprise": 100}
quiz_agg["team_size_num"]     = quizzes_dedup["team_size"].map(team_map).fillna(1)

# Raw quiz for offer logic
quiz_raw = quizzes_dedup.set_index("user_id")[
    ["frustration", "first_feature", "role", "experience"]
].to_dict("index")

# ---------------------------------------------------------------------------
# PROPERTIES
# ---------------------------------------------------------------------------
props_f = props[["user_id", "subscription_start_date", "plan_tier"]].copy()
props_f["subscription_start_date"] = pd.to_datetime(
    props_f["subscription_start_date"], errors="coerce", utc=True
)

# =============================================================================
# 3. MERGE
# =============================================================================
print("\n[3/7] Merging feature table...")

df = users.copy()
for frame in [
    purch_agg,
    txn_agg,
    quiz_agg,
    props_f[["user_id", "plan_tier"]],
    gen_agg.drop(columns=["gen_trend"], errors="ignore"),
    gen_trend,
]:
    df = df.merge(frame, on="user_id", how="left")

df = df.reset_index(drop=True).fillna(0)
print(f"  Shape: {df.shape}")
print(f"  Label distribution:\n{df['churn_status'].value_counts().to_string()}")

# =============================================================================
# 4. TRAIN
# =============================================================================
print("\n[4/7] Training models...")

FEATURES = [
    # Payment health — invol signals
    "payment_fail_rate", "fail_count", "txn_count",
    "card_country_mismatch", "has_prepaid", "has_debit",
    "high_risk_country", "bank_country_tier", "billing_country_tier",
    "total_spend_txn", "avg_spend_txn",
    # Card type — retention signals
    "is_business", "is_virtual",
    # Purchase behaviour — retention signals
    "total_purchases", "lifetime_spend", "avg_purchase",
    "sub_creates", "sub_updates", "credit_packs",
    "plan_tier",
    # Generation activity — vol signals
    "gen_count", "completion_rate", "nsfw_rate", "fail_rate_gen",
    "avg_credit_cost", "total_credits", "unique_models",
    "gen_trend", "frustration_score",
    "avg_resolution", "max_resolution",
    "days_to_first_gen",
    # Quiz — intent signals
    "quiz_high_cost", "quiz_hard_prompt", "quiz_inconsistent", "quiz_limited",
    "is_beginner", "is_expert", "is_invited", "is_personal",
    "wants_video", "is_filmmaker_role", "team_size_num",
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Features: {len(FEATURES)}")

df["churned"]  = (df["churn_status"] != "not_churned").astype(int)
df["is_invol"] = (df["churn_status"] == "invol_churn").astype(int)

X  = df[FEATURES]
y1 = df["churned"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y1, test_size=0.2, random_state=42, stratify=y1
)

# SMOTE — balance classes before training
print("  Applying SMOTE...")
X_tr_bal, y_tr_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr, y_tr)
print(f"  Class counts after SMOTE: {pd.Series(y_tr_bal).value_counts().to_dict()}")

# Optuna — automated hyperparameter search
print("  Running Optuna (50 trials)...")

def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 800),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        eval_metric      = "aucpr",
        random_state     = 42,
        verbosity        = 0,
    )
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr_bal, y_tr_bal)
    return f1_score(y_te, m.predict(X_te), average="macro")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best = {**study.best_params, "eval_metric": "aucpr", "random_state": 42, "verbosity": 0}
print(f"  Best params: {study.best_params}")
print(f"  Best Macro F1: {study.best_value:.4f}")

# Stage 1: churn vs not_churned
model1 = xgb.XGBClassifier(**best)
model1.fit(X_tr_bal, y_tr_bal)

y_pred1 = model1.predict(X_te)
y_prob1 = model1.predict_proba(X_te)[:, 1]
print("\n  --- Stage 1: Churn Detection ---")
print(classification_report(y_te, y_pred1, target_names=["Not Churned", "Churned"]))
print(f"  ROC-AUC : {roc_auc_score(y_te, y_prob1):.4f}")
print(f"  PR-AUC  : {average_precision_score(y_te, y_prob1):.4f}")
print(f"  Macro F1: {f1_score(y_te, y_pred1, average='macro'):.4f}")

# Stage 2: voluntary vs involuntary (on churned users only)
ch_mask = df["churned"] == 1
X2, y2  = df.loc[ch_mask, FEATURES], df.loc[ch_mask, "is_invol"]

X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)
X2_tr_bal, y2_tr_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X2_tr, y2_tr)

model2 = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
model2.fit(X2_tr_bal, y2_tr_bal)

y_pred2 = model2.predict(X2_te)
y_prob2 = model2.predict_proba(X2_te)[:, 1]
print("\n  --- Stage 2: Voluntary vs Involuntary ---")
print(classification_report(y2_te, y_pred2, target_names=["Voluntary", "Involuntary"]))
print(f"  ROC-AUC : {roc_auc_score(y2_te, y_prob2):.4f}")
print(f"  PR-AUC  : {average_precision_score(y2_te, y_prob2):.4f}")
print(f"  Macro F1: {f1_score(y2_te, y_pred2, average='macro'):.4f}")

# =============================================================================
# 5. SHAP EXPLAINABILITY
# =============================================================================
print("\n[5/7] Computing SHAP values...")

explainer = shap.TreeExplainer(model1)
shap_te   = explainer.shap_values(X_te)
shap_all  = explainer.shap_values(X)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_te, X_te, feature_names=FEATURES, show=False)
plt.title("Feature Importance — SHAP Summary", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary.png")

SHAP_TEMPLATES = {
    "payment_fail_rate":     lambda v: f"{v:.0%} of payment attempts failed — card likely declined",
    "fail_count":            lambda v: f"{int(v)} failed payment attempts in first 14 days",
    "card_country_mismatch": lambda v: "Card country differs from billing address — bank blocking",
    "has_prepaid":           lambda v: "Prepaid card detected — high subscription decline risk",
    "has_debit":             lambda v: "Debit card — balance may be insufficient at renewal",
    "high_risk_country":     lambda v: "Billing country has elevated card decline rate",
    "bank_country_tier":     lambda v: "Bank is in a lower-tier payment region",
    "billing_country_tier":  lambda v: "Billing country is in a lower-tier payment region",
    "is_business":           lambda v: "Business card — corporate billing, high payment reliability",
    "quiz_high_cost":        lambda v: "User reported platform feels too expensive at signup",
    "quiz_hard_prompt":      lambda v: "User reported difficulty with prompting at signup",
    "quiz_inconsistent":     lambda v: "User reported inconsistent results at signup",
    "quiz_limited":          lambda v: "User reported feeling limited by generation quota",
    "nsfw_rate":             lambda v: f"{v:.0%} of generations blocked as NSFW — content frustration",
    "frustration_score":     lambda v: f"Accumulated frustration score {v:.2f} from failed/nsfw gens",
    "gen_trend":             lambda v: "Generation activity declining over observation period",
    "completion_rate":       lambda v: f"Only {v:.0%} of generations completed successfully",
    "avg_credit_cost":       lambda v: f"Average {v:.0f} credits per generation — high consumption",
    "total_purchases":       lambda v: "No purchases in first 14 days — low platform commitment",
    "sub_updates":           lambda v: "No subscription upgrades — low engagement with platform value",
    "credit_packs":          lambda v: f"Bought {int(v)} credit pack(s) — actively engaged with platform",
    "is_beginner":           lambda v: "Beginner user — higher onboarding friction risk",
    "days_to_first_gen":     lambda v: f"Took {int(v)} days after signup to make first generation",
    "avg_resolution":        lambda v: f"Uses low-resolution generations (avg rank {v:.1f}/3)",
    "plan_tier":             lambda v: f"On plan tier {int(v)}/4 — lower plans have higher churn rate",
    "is_invited":            lambda v: "Invited user — typically higher retention than personal signups",
    "unique_models":         lambda v: f"Used only {int(v)} generation model type(s) — narrow exploration",
}

def get_shap_reasons(idx):
    sv    = shap_all[idx]
    row   = X.iloc[idx]
    pairs = sorted(zip(sv, FEATURES), key=lambda x: x[0], reverse=True)
    reasons = []
    for shap_val, feat in pairs:
        if shap_val <= 0 or feat not in SHAP_TEMPLATES:
            continue
        reasons.append(SHAP_TEMPLATES[feat](row[feat]))
        if len(reasons) == 3:
            break
    return reasons

# =============================================================================
# 6. PREDICTIONS
# =============================================================================
print("\n[6/7] Generating predictions...")

df["churn_prob"] = model1.predict_proba(X)[:, 1]
df["churn_type"] = "not_churned"

at_risk = df["churn_prob"] > 0.5
invol_p = model2.predict_proba(df.loc[at_risk, FEATURES])[:, 1]
df.loc[at_risk, "churn_type"] = np.where(invol_p > 0.5, "invol_churn", "vol_churn")

print(f"  Prediction distribution:\n{df['churn_type'].value_counts().to_string()}")

# =============================================================================
# 7. DISCOUNT + VOL OFFER + SUBMISSION
# =============================================================================
print("\n[7/7] Building submission file...")

max_purch = max(df["total_purchases"].max(), 1)

ALL_GEN_TYPES = [
    "video_model_7", "video_model_10", "video_model_11",
    "video_model_12", "video_model_13",
]

ROLE_TO_RECOMMENDATION = {
    "filmmaker":       ("video_model_13", "Cinematic Visuals — highest quality video model"),
    "creator":         ("video_model_12", "Cinematic video for content creators"),
    "designer":        ("video_model_11", "Fast video generation for design mockups"),
    "just-for-fun":    ("video_model_7",  "Short fun clips — easiest to start with"),
    "brand-owner":     ("video_model_12", "Cinematic Visuals for brand content"),
    "marketer":        ("video_model_10", "Commercial & Ad Videos"),
    "founder":         ("video_model_11", "Fast product demo videos"),
    "educator":        ("video_model_11", "Talking avatars for educational content"),
    "prompt-engineer": ("video_model_13", "Highest quality for prompt experimentation"),
    "developer":       ("video_model_10", "Versatile model for API integration"),
}

FIRST_FEATURE_LABELS = {
    "video-creation": "Video Creation", "image-creation": "Image Creation",
    "edit-image": "Image Editing", "consistent-character": "Consistent Character",
    "viral-effects": "Viral Effects", "product-placement": "Product Placement",
    "draw-to-video": "Draw to Video", "talking-avatars": "Talking Avatars",
    "upscale": "Video Upscaling", "Upscale": "Video Upscaling",
    "Commercial & Ad Videos": "Commercial & Ad Videos",
    "Video Generations": "Video Generations",
    "Cinematic Visuals": "Cinematic Visuals",
    "Viral Social Media Content": "Viral Social Media Content",
    "Realistic AI Avatars": "Realistic AI Avatars",
    "Image Editing & Inpaint": "Image Editing & Inpainting",
    "Realistic Avatars & AI Twins": "Realistic Avatars & AI Twins",
    "Storyboarding": "Storyboarding",
    "Lipsync & Talking Avatars": "Lipsync & Talking Avatars",
}

def calc_discount(row):
    # Discount based on loyalty: plan tier + purchases + credit usage
    plan    = (min(row.get("plan_tier", 1), 4) - 1) / 3       # 0-1
    purch   = min(row.get("total_purchases",  0) / max_purch, 1.0)
    credits = min(row.get("total_credits",    0) / 50000, 1.0) # normalized
    loyalty = plan * 0.45 + purch * 0.35 + credits * 0.20
    return int(round(10 + loyalty * 30))   # 10% to 40%

def build_invol_action(row, discount):
    parts = ["Payment recovery: smart retry on day 3, 7, 14."]
    if row.get("card_country_mismatch", 0) == 1:
        parts.append("Prompt user to update card to match billing country.")
    if row.get("has_prepaid", 0) == 1:
        parts.append("Suggest switching from prepaid to credit or debit card.")
    if row.get("bank_country_tier", 2) < 1:
        parts.append("Offer alternative payment methods (PayPal, local payment rails).")
    parts.append(f"Offer {discount}% personal loyalty discount (one-time, never repeated).")
    return " ".join(parts)

def build_vol_action(uid, row):
    quiz       = quiz_raw.get(uid, {})
    role       = str(quiz.get("role", "")).strip().lower()
    first_feat = quiz.get("first_feature", "")
    experience = str(quiz.get("experience", "")).strip().lower()
    used       = used_types_map.get(uid, [])
    unused     = [t for t in ALL_GEN_TYPES if t not in used]

    actions = []

    if row.get("nsfw_rate", 0) > 0.3:
        actions.append(
            f"NSFW friction ({row['nsfw_rate']:.0%} of gens blocked): "
            "send content guidelines and showcase what is possible within platform rules."
        )

    if row.get("completion_rate", 1) < 0.5 and row.get("gen_count", 0) > 5:
        actions.append(
            f"Low completion rate ({row['completion_rate']:.0%}): "
            "send role-specific prompt templates and a quick-start tutorial."
        )

    if row.get("days_to_first_gen", 0) > 3:
        actions.append(
            f"Slow activation ({int(row['days_to_first_gen'])} days to first gen): "
            "send onboarding nudge with a guided first-project walkthrough."
        )

    if isinstance(first_feat, str) and first_feat and first_feat not in str(used):
        label = FIRST_FEATURE_LABELS.get(first_feat, first_feat)
        actions.append(
            f"Unmet expectation: signed up for '{label}' but never used it. "
            "Send a direct tutorial with free trial credits."
        )

    role_rec = ROLE_TO_RECOMMENDATION.get(role)
    if role_rec and unused:
        model_name, model_desc = role_rec
        target = model_name if model_name in unused else unused[0]
        actions.append(
            f"Unexplored feature for {role}: try {target} — {model_desc}."
        )
    elif unused:
        actions.append(
            f"Unexplored model: {unused[0]} has not been tried. "
            "Send examples relevant to user content goals."
        )

    return " | ".join(actions[:2]) if actions else (
        "Re-engagement: show platform highlights, offer annual plan or downgrade option."
    )

rows = []
for idx, row in df.iterrows():
    r      = row.to_dict()
    uid    = row["user_id"]
    ctype  = row["churn_type"]

    reasons  = get_shap_reasons(idx)
    discount = None
    action   = ""

    if ctype == "invol_churn":
        discount = calc_discount(r)
        action   = build_invol_action(r, discount)
    elif ctype == "vol_churn":
        action   = build_vol_action(uid, r)

    rows.append({
        "user_id":            uid,
        "churn_probability":  round(float(row["churn_prob"]), 4),
        "churn_type":         ctype,
        "reason_1":           reasons[0] if len(reasons) > 0 else "",
        "reason_2":           reasons[1] if len(reasons) > 1 else "",
        "reason_3":           reasons[2] if len(reasons) > 2 else "",
        "discount_pct":       discount,
        "recommended_action": action,
    })

submission = pd.DataFrame(rows)
submission.to_csv("predictions.csv", index=False)

print(f"\n  Saved: predictions.csv ({len(submission)} rows)")
print("\n  Sample invol:")
for _, r in submission[submission["churn_type"] == "invol_churn"].head(3).iterrows():
    print(f"    prob={r['churn_probability']}  disc={r['discount_pct']}%")
    print(f"    reason: {r['reason_1']}")
    print(f"    action: {r['recommended_action'][:90]}")
    print()
print("  Sample vol:")
for _, r in submission[submission["churn_type"] == "vol_churn"].head(3).iterrows():
    print(f"    prob={r['churn_probability']}")
    print(f"    reason: {r['reason_1']}")
    print(f"    action: {r['recommended_action'][:90]}")
    print()
print("Done.")