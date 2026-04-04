import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("The Retention Architect — Churn Prediction Pipeline v4")
print("=" * 60)

# ── 1. LOAD ────────────────────────────────────────────────────────────────
print("\n[1/6] Loading files...")
users   = pd.read_csv('train_users.csv',                           low_memory=False)
props   = pd.read_csv('train_users_properties.csv',                low_memory=False)
purch   = pd.read_csv('train_users_purchases.csv',                 low_memory=False)
txn     = pd.read_csv('train_users_transaction_attempts_v1.csv',   low_memory=False)
quizzes = pd.read_csv('train_users_quizzes.csv',                   low_memory=False)
gens    = pd.read_csv('test_users_generations.csv',                low_memory=False)

print(f"  users:        {users.shape}")
print(f"  properties:   {props.shape}")
print(f"  purchases:    {purch.shape}")
print(f"  transactions: {txn.shape}  (now has user_id!)")
print(f"  quizzes:      {quizzes.shape}")
print(f"  generations:  {gens.shape}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────
print("\n[2/6] Building features...")

# --- Purchase features ---
purch_f = purch.groupby('user_id').agg(
    total_purchases = ('transaction_id', 'count'),
    lifetime_spend  = ('purchase_amount_dollars', 'sum'),
    avg_purchase    = ('purchase_amount_dollars', 'mean'),
    sub_creates     = ('purchase_type', lambda x: (x == 'Subscription Create').sum()),
    sub_updates     = ('purchase_type', lambda x: (x == 'Subscription Update').sum()),
    credit_packs    = ('purchase_type', lambda x: (x == 'Credits package').sum()),
).reset_index()

# --- Transaction / payment features (now directly has user_id) ---
txn_f = txn.groupby('user_id').agg(
    txn_count       = ('transaction_id', 'count'),
    fail_count      = ('failure_code',   lambda x: x.notna().sum()),
    total_spend_txn = ('amount_in_usd',  'sum'),
    avg_spend_txn   = ('amount_in_usd',  'mean'),
).reset_index()
txn_f['payment_fail_rate'] = txn_f['fail_count'] / txn_f['txn_count'].clip(lower=1)

# Card country mismatch (card issued in different country than billing)
mm = txn.copy()
mm['mismatch'] = (mm['billing_address_country'] != mm['card_country'])
mm_f = mm.groupby('user_id')['mismatch'].max().reset_index()
mm_f.columns = ['user_id', 'card_country_mismatch']
mm_f['card_country_mismatch'] = mm_f['card_country_mismatch'].astype(int)
txn_f = txn_f.merge(mm_f, on='user_id', how='left')

# Prepaid / virtual card flag (higher decline risk)
card_flags = txn.groupby('user_id').agg(
    has_prepaid = ('is_prepaid', lambda x: x.astype(str).str.lower().eq('true').any()),
    has_virtual = ('is_virtual', lambda x: x.astype(str).str.lower().eq('true').any()),
).reset_index()
card_flags['has_prepaid'] = card_flags['has_prepaid'].astype(int)
card_flags['has_virtual'] = card_flags['has_virtual'].astype(int)
txn_f = txn_f.merge(card_flags, on='user_id', how='left')

# High-risk country flag (countries with above-average failure rates)
country_fail = txn.groupby('billing_address_country')['failure_code'].apply(
    lambda x: x.notna().mean()
).reset_index()
country_fail.columns = ['billing_address_country', 'country_fail_rate']
high_risk_countries = set(
    country_fail[country_fail['country_fail_rate'] > country_fail['country_fail_rate'].mean()]['billing_address_country']
)
# Per user: did any of their transactions come from a high-risk country?
txn['high_risk_country'] = txn['billing_address_country'].isin(high_risk_countries).astype(int)
hr_f = txn.groupby('user_id')['high_risk_country'].max().reset_index()
txn_f = txn_f.merge(hr_f, on='user_id', how='left')

# --- Quiz features ---
quiz_f = quizzes[['user_id']].copy()
quiz_f['quiz_high_cost'] = (quizzes['frustration'] == 'high-cost').astype(int)
quiz_f['is_invited']     = (quizzes['flow_type']   == 'invited').astype(int)
quiz_f['is_personal']    = (quizzes['flow_type']   == 'personal').astype(int)
quiz_f['is_beginner']    = (quizzes['experience']  == 'beginner').astype(int)
quiz_f['is_advanced']    = (quizzes['experience']  == 'advanced').astype(int)
team_map = {'1': 1, 'small': 3, 'growing': 10, 'midsize': 30, 'enterprise': 100}
quiz_f['team_size_num']  = quizzes['team_size'].map(team_map).fillna(1)

# --- Properties features ---
props_f = props[['user_id', 'subscription_start_date', 'subscription_plan', 'country_code']].copy()
props_f['subscription_start_date'] = pd.to_datetime(
    props_f['subscription_start_date'], errors='coerce', utc=True)
ref_date = props_f['subscription_start_date'].max()
props_f['sub_tenure_days'] = (
    ref_date - props_f['subscription_start_date']
).dt.days.fillna(0).clip(lower=0)
plan_map = {'Higgsfield Basic': 1, 'Higgsfield Pro': 2, 'Higgsfield Max': 3, 'Higgsfield Teams': 4}
props_f['plan_tier'] = props_f['subscription_plan'].map(plan_map).fillna(1)

# ── 3. MERGE INTO ONE TABLE ───────────────────────────────────────────────
print("\n[3/6] Merging into one feature table...")
df = users.copy()
for f in [
    purch_f,
    txn_f,
    quiz_f,
    props_f[['user_id', 'sub_tenure_days', 'plan_tier']],
]:
    df = df.merge(f, on='user_id', how='left')

df = df.fillna(0)
print(f"  Feature table: {df.shape}")
print(f"  Labels:\n{df['churn_status'].value_counts().to_string()}")

# ── 4. TRAIN ──────────────────────────────────────────────────────────────
print("\n[4/6] Training models...")

FEATURES = [
    # Purchase behaviour
    'total_purchases', 'lifetime_spend', 'avg_purchase',
    'sub_creates', 'sub_updates', 'credit_packs',
    # Payment health (now much richer with direct user_id)
    'payment_fail_rate', 'fail_count', 'txn_count',
    'total_spend_txn', 'avg_spend_txn',
    'card_country_mismatch', 'has_prepaid', 'has_virtual',
    'high_risk_country',
    # Onboarding intent
    'quiz_high_cost', 'is_invited', 'is_personal',
    'is_beginner', 'is_advanced', 'team_size_num',
    # Subscription
    'sub_tenure_days', 'plan_tier',
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Using {len(FEATURES)} features")

X = df[FEATURES]
df['churned']  = (df['churn_status'] != 'not_churned').astype(int)
df['is_invol'] = (df['churn_status'] == 'invol_churn').astype(int)
y1 = df['churned']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y1, test_size=0.2, random_state=42, stratify=y1
)

# Stage 1 — churn vs not churned
pos_w = max(1, (y_tr==0).sum() / max((y_tr==1).sum(), 1))
model1 = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_w, eval_metric='aucpr',
    random_state=42, verbosity=0,
)
model1.fit(X_tr, y_tr)
print("\n  --- Stage 1: Churn vs Not Churned ---")
print(classification_report(y_te, model1.predict(X_te), target_names=['Not Churned', 'Churned']))

# Stage 2 — voluntary vs involuntary
ch_mask = df['churned'] == 1
X2, y2  = df.loc[ch_mask, FEATURES], df.loc[ch_mask, 'is_invol']
model2  = None
if y2.nunique() > 1:
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )
    model2 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    model2.fit(X2_tr, y2_tr)
    print("\n  --- Stage 2: Voluntary vs Involuntary ---")
    print(classification_report(y2_te, model2.predict(X2_te), target_names=['Voluntary', 'Involuntary']))

# ── 5. PREDICT ────────────────────────────────────────────────────────────
print("\n[5/6] Generating predictions...")

df['churn_prob'] = model1.predict_proba(X)[:, 1]
df['churn_type'] = 'not_churned'

at_risk = df['churn_prob'] > 0.5
if model2 is not None and at_risk.sum() > 0:
    invol_p = model2.predict_proba(df.loc[at_risk, FEATURES])[:, 1]
    df.loc[at_risk, 'churn_type'] = np.where(invol_p > 0.5, 'invol_churn', 'vol_churn')
elif at_risk.sum() > 0:
    df.loc[at_risk, 'churn_type'] = np.where(
        df.loc[at_risk, 'payment_fail_rate'] > 0.4, 'invol_churn', 'vol_churn'
    )

print(f"  Results:\n{df['churn_type'].value_counts().to_string()}")

# ── 6. REASONS + DISCOUNT + SUBMISSION ───────────────────────────────────
print("\n[6/6] Building output file...")

max_purch = max(df['total_purchases'].max(), 1)

def get_reasons(r):
    out = []
    if r.get('payment_fail_rate', 0) > 0.3:
        out.append(f"{r['payment_fail_rate']:.0%} of payment attempts failed — card likely declined")
    if r.get('card_country_mismatch', 0) == 1:
        out.append("Card country mismatches billing address — bank likely blocking")
    if r.get('high_risk_country', 0) == 1:
        out.append("Billing country has elevated card decline rate")
    if r.get('has_prepaid', 0) == 1:
        out.append("Using a prepaid card — higher decline risk")
    if r.get('quiz_high_cost', 0) == 1:
        out.append("Reported platform feels expensive during onboarding")
    if r.get('total_purchases', 0) == 0:
        out.append("No purchases made — user never converted from trial")
    if r.get('sub_updates', 0) == 0 and r.get('total_purchases', 0) > 0:
        out.append("Never upgraded subscription — low engagement with platform value")
    if r.get('credit_packs', 0) > 2:
        out.append("Repeatedly buying credit packs — subscription may feel too expensive")
    return out[:3]

def calc_discount(r):
    tenure  = min(r.get('sub_tenure_days', 0) / 365, 1.0)
    p_score = min(r.get('total_purchases',  0) / max_purch, 1.0)
    loyalty = tenure * 0.6 + p_score * 0.4
    return int(round(10 + loyalty * 30))

rows = []
for _, row in df.iterrows():
    r      = row.to_dict()
    ctype  = row['churn_type']
    reasons = get_reasons(r)
    discount = None
    action   = ""

    if ctype == 'invol_churn':
        discount = calc_discount(r)
        action = f"Payment recovery: smart retry day 3/7/14. Offer {discount}% loyalty discount (one-time only)."
    elif ctype == 'vol_churn':
        action = "Re-engagement: highlight unused features, offer annual plan or plan downgrade."

    rows.append({
        'user_id':            row['user_id'],
        'churn_probability':  round(float(row['churn_prob']), 4),
        'churn_type':         ctype,
        'reason_1':           reasons[0] if len(reasons) > 0 else '',
        'reason_2':           reasons[1] if len(reasons) > 1 else '',
        'reason_3':           reasons[2] if len(reasons) > 2 else '',
        'discount_pct':       discount,
        'recommended_action': action,
    })

submission = pd.DataFrame(rows)
submission.to_csv('predictions.csv', index=False)

print(f"\n  Saved: predictions.csv ({len(submission)} rows)")
print("\nSample:")
print(submission[['user_id','churn_probability','churn_type','discount_pct']].head(12).to_string(index=False))
print("\nDone!")
