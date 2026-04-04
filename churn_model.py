import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""
The Retention Architect — v5
=============================
Full logic:
  - invol_churn: payment failures + high credit cost + high-cost frustration
  - vol_churn:   nsfw frustration + declining activity + low completion + unmet first_feature
  - not_churned: stable payments + normal activity

  - Discount: only for invol_churn, based on tenure + plan tier + gen count + lifetime spend
  - Vol offer: unused generation types + role-based recommendation

Files needed:
  train_users.csv
  train_users_properties.csv
  train_users_purchases.csv
  train_users_transaction_attempts_v1.csv
  train_users_quizzes.csv
  test_users_generations.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("The Retention Architect — v5")
print("=" * 60)

# ── 1. LOAD ────────────────────────────────────────────────────────────────
print("\n[1/6] Loading files...")
users   = pd.read_csv('train_users.csv',                          low_memory=False)
props   = pd.read_csv('train_users_properties.csv',               low_memory=False)
purch   = pd.read_csv('train_users_purchases.csv',                low_memory=False)
txn     = pd.read_csv('train_users_transaction_attempts_v1.csv',  low_memory=False)
quizzes = pd.read_csv('train_users_quizzes.csv',                  low_memory=False)
gens    = pd.read_csv('test_users_generations.csv',               low_memory=False)

print(f"  users: {users.shape} | props: {props.shape} | purch: {purch.shape}")
print(f"  txn: {txn.shape} | quizzes: {quizzes.shape} | gens: {gens.shape}")

# ── 2. GENERATION FEATURES ────────────────────────────────────────────────
print("\n[2/6] Building features...")

gens['created_at'] = pd.to_datetime(gens['created_at'], errors='coerce', utc=True)

gen_f = gens.groupby('user_id').agg(
    gen_count        = ('generation_id', 'count'),
    completed_count  = ('status', lambda x: (x == 'completed').sum()),
    failed_count     = ('status', lambda x: (x == 'failed').sum()),
    nsfw_count       = ('status', lambda x: (x == 'nsfw').sum()),
    avg_credit_cost  = ('credit_cost', 'mean'),
    total_credits    = ('credit_cost', 'sum'),
    unique_models    = ('generation_type', 'nunique'),
    first_gen_date   = ('created_at', 'min'),
    last_gen_date    = ('created_at', 'max'),
).reset_index()

gen_f['completion_rate'] = gen_f['completed_count'] / gen_f['gen_count'].clip(lower=1)
gen_f['nsfw_rate']       = gen_f['nsfw_count']      / gen_f['gen_count'].clip(lower=1)
gen_f['fail_rate_gen']   = gen_f['failed_count']    / gen_f['gen_count'].clip(lower=1)

# Activity trend: gens in second half vs first half of their history
def activity_trend(group):
    if len(group) < 4:
        return 1.0
    group = group.sort_values('created_at')
    mid = len(group) // 2
    first_half  = len(group.iloc[:mid])
    second_half = len(group.iloc[mid:])
    return second_half / max(first_half, 1)

trend = gens.groupby('user_id').apply(activity_trend).reset_index()
trend.columns = ['user_id', 'gen_trend']
gen_f = gen_f.merge(trend, on='user_id', how='left')

# Which generation types did this user use
used_types = gens.groupby('user_id')['generation_type'].apply(
    lambda x: list(x.dropna().unique())
).reset_index()
used_types.columns = ['user_id', 'used_gen_types']
gen_f = gen_f.merge(used_types, on='user_id', how='left')

# ── 3. PURCHASE FEATURES ─────────────────────────────────────────────────
purch_f = purch.groupby('user_id').agg(
    total_purchases  = ('transaction_id', 'count'),
    lifetime_spend   = ('purchase_amount_dollars', 'sum'),
    avg_purchase     = ('purchase_amount_dollars', 'mean'),
    sub_creates      = ('purchase_type', lambda x: (x == 'Subscription Create').sum()),
    sub_updates      = ('purchase_type', lambda x: (x == 'Subscription Update').sum()),
    credit_packs     = ('purchase_type', lambda x: (x == 'Credits package').sum()),
).reset_index()

# ── 4. TRANSACTION FEATURES ───────────────────────────────────────────────
txn_f = txn.groupby('user_id').agg(
    txn_count       = ('transaction_id', 'count'),
    fail_count      = ('failure_code',   lambda x: x.notna().sum()),
    total_spend_txn = ('amount_in_usd',  'sum'),
    avg_spend_txn   = ('amount_in_usd',  'mean'),
).reset_index()
txn_f['payment_fail_rate'] = txn_f['fail_count'] / txn_f['txn_count'].clip(lower=1)

mm = txn.copy()
mm['mismatch'] = (mm['billing_address_country'] != mm['card_country']).astype(int)
mm_f = mm.groupby('user_id')['mismatch'].max().reset_index()
mm_f.columns = ['user_id', 'card_country_mismatch']
txn_f = txn_f.merge(mm_f, on='user_id', how='left')

card_f = txn.copy()
card_f['is_prepaid_flag'] = card_f['is_prepaid'].astype(str).str.lower().eq('true').astype(int)
card_f['is_debit_flag']   = card_f['card_funding'].eq('debit').astype(int)
cf = card_f.groupby('user_id').agg(
    has_prepaid = ('is_prepaid_flag', 'max'),
    has_debit   = ('is_debit_flag',   'max'),
).reset_index()
txn_f = txn_f.merge(cf, on='user_id', how='left')

country_fail = txn.groupby('billing_address_country')['failure_code'].apply(
    lambda x: x.notna().mean()).reset_index()
country_fail.columns = ['billing_address_country', 'country_fail_rate']
high_risk = set(country_fail[
    country_fail['country_fail_rate'] > country_fail['country_fail_rate'].mean()
]['billing_address_country'])
txn['high_risk_country'] = txn['billing_address_country'].isin(high_risk).astype(int)
hr_f = txn.groupby('user_id')['high_risk_country'].max().reset_index()
txn_f = txn_f.merge(hr_f, on='user_id', how='left')

# ── 5. QUIZ FEATURES ──────────────────────────────────────────────────────
quiz_f = quizzes[['user_id', 'frustration', 'first_feature', 'role',
                   'flow_type', 'experience', 'usage_plan', 'team_size']].copy()
quiz_f['quiz_high_cost']       = quizzes['frustration'].isin(
    ['high-cost', 'High cost of top models']).astype(int)
quiz_f['quiz_hard_prompt']     = quizzes['frustration'].isin(
    ['hard-prompt', 'Hard to prompt', 'confusing', 'AI is confusing to me']).astype(int)
quiz_f['quiz_limited_gens']    = quizzes['frustration'].isin(
    ['limited', 'Limited generations']).astype(int)
quiz_f['quiz_inconsistent']    = quizzes['frustration'].isin(
    ['inconsistent', 'Inconsistent results']).astype(int)
quiz_f['quiz_quality_issue']   = quizzes['frustration'].isin(
    ['hard-prompt', 'Hard to prompt', 'confusing', 'AI is confusing to me',
     'inconsistent', 'Inconsistent results']).astype(int)
quiz_f['quiz_nsfw_frustration']= quizzes['frustration'].isin(
    ['nsfw', 'content-restrictions']).astype(int)
quiz_f['is_beginner']         = (quizzes['experience'] == 'beginner').astype(int)
quiz_f['is_advanced']         = (quizzes['experience'] == 'advanced').astype(int)
quiz_f['is_personal']         = (quizzes['flow_type']  == 'personal').astype(int)
quiz_f['is_invited']          = (quizzes['flow_type']  == 'invited').astype(int)
team_map = {'1': 1, 'small': 3, 'growing': 10, 'midsize': 30, 'enterprise': 100}
quiz_f['team_size_num']       = quizzes['team_size'].map(team_map).fillna(1)

# ── 6. PROPERTIES FEATURES ────────────────────────────────────────────────
props_f = props[['user_id', 'subscription_start_date', 'subscription_plan', 'country_code']].copy()
props_f['subscription_start_date'] = pd.to_datetime(
    props_f['subscription_start_date'], errors='coerce', utc=True)
ref_date = props_f['subscription_start_date'].max()
props_f['sub_tenure_days'] = (
    ref_date - props_f['subscription_start_date']
).dt.days.fillna(0).clip(lower=0)

# Plan tier: Basic=1, Pro=2, Creator=3, Ultimate=4
plan_map = {
    'Higgsfield Basic': 1, 'Higgsfield Pro': 2,
    'Higgsfield Creator': 3, 'Higgsfield Ultimate': 4,
    'Higgsfield Teams': 3,
}
props_f['plan_tier'] = props_f['subscription_plan'].map(plan_map).fillna(1)

# ── 7. MERGE ──────────────────────────────────────────────────────────────
print("\n[3/6] Merging feature table...")
df = users.copy()
for f in [
    purch_f,
    txn_f,
    quiz_f.drop(columns=['frustration','first_feature','role','flow_type',
                          'experience','usage_plan','team_size'], errors='ignore'),
    props_f[['user_id','sub_tenure_days','plan_tier']],
    gen_f.drop(columns=['first_gen_date','last_gen_date','used_gen_types'], errors='ignore'),
]:
    df = df.merge(f, on='user_id', how='left')

# Keep raw quiz columns separately for recommendation logic
quiz_raw = quizzes[['user_id','frustration','first_feature','role',
                     'flow_type','experience']].copy()

df = df.fillna(0)
print(f"  Feature table: {df.shape}")
print(f"  Labels:\n{df['churn_status'].value_counts().to_string()}")

# ── 8. TRAIN ──────────────────────────────────────────────────────────────
print("\n[4/6] Training models...")

FEATURES = [
    # Payment health (invol signals)
    'payment_fail_rate', 'fail_count', 'txn_count',
    'card_country_mismatch', 'has_prepaid', 'has_debit', 'high_risk_country',
    'total_spend_txn', 'avg_spend_txn',
    # Purchase behaviour
    'total_purchases', 'lifetime_spend', 'avg_purchase',
    'sub_creates', 'sub_updates', 'credit_packs',
    # Generation activity (vol signals)
    'gen_count', 'completion_rate', 'nsfw_rate', 'fail_rate_gen',
    'avg_credit_cost', 'total_credits', 'unique_models', 'gen_trend',
    # Quiz signals
    'quiz_high_cost', 'quiz_nsfw_frustration', 'quiz_quality_issue',
    'quiz_hard_prompt', 'quiz_limited_gens', 'quiz_inconsistent',
    'is_beginner', 'is_advanced', 'is_personal', 'is_invited', 'team_size_num',
    # Subscription
    'sub_tenure_days', 'plan_tier',
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Using {len(FEATURES)} features")

X  = df[FEATURES]
df['churned']  = (df['churn_status'] != 'not_churned').astype(int)
df['is_invol'] = (df['churn_status'] == 'invol_churn').astype(int)
y1 = df['churned']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y1, test_size=0.2, random_state=42, stratify=y1)

pos_w = max(1, (y_tr==0).sum() / max((y_tr==1).sum(), 1))
model1 = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_w, eval_metric='aucpr',
    random_state=42, verbosity=0,
)
model1.fit(X_tr, y_tr)
print("\n  --- Stage 1: Churn vs Not Churned ---")
print(classification_report(y_te, model1.predict(X_te),
      target_names=['Not Churned', 'Churned']))

# Stage 2: vol vs invol
ch_mask = df['churned'] == 1
X2, y2  = df.loc[ch_mask, FEATURES], df.loc[ch_mask, 'is_invol']
model2  = None
if y2.nunique() > 1:
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2)
    model2 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    model2.fit(X2_tr, y2_tr)
    print("\n  --- Stage 2: Voluntary vs Involuntary ---")
    print(classification_report(y2_te, model2.predict(X2_te),
          target_names=['Voluntary', 'Involuntary']))

# ── 9. PREDICT ────────────────────────────────────────────────────────────

# ── SHAP EXPLAINABILITY ───────────────────────────────────────────────────
print("\n[SHAP] Считаем объяснения модели...")

explainer   = shap.TreeExplainer(model1)
shap_values = explainer.shap_values(X_te)

# 1. Глобальный график — какие признаки важнее всего для модели
print("  Сохраняем shap_summary.png...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_te,
    feature_names=FEATURES,
    show=False,
    plot_size=(10, 7)
)
plt.title("SHAP — важность признаков (глобально)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Сохранено: shap_summary.png")

# 2. Считаем SHAP значения для ВСЕХ пользователей (для reasons в CSV)
print("  Считаем SHAP для всех пользователей...")
shap_all = explainer.shap_values(X)
print("  Готово.")

# 3. Функция: топ-3 признака по SHAP для одного пользователя
def get_shap_reasons(idx, ctype):
    sv    = shap_all[idx]                        # SHAP значения для этого юзера
    pairs = list(zip(sv, FEATURES))              # [(shap_val, feature_name), ...]

    if ctype == 'invol_churn':
        # Для invol берём признаки с наибольшим положительным SHAP
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    elif ctype == 'vol_churn':
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    else:
        return []

    # Переводим название признака в человеческий текст
    templates = {
        'payment_fail_rate':     lambda v, s: f"{v:.0%} платежей провалилось — карта отклонена",
        'card_country_mismatch': lambda v, s: "Страна карты не совпадает с биллингом — блокировка банка",
        'has_prepaid':           lambda v, s: "Предоплаченная карта — высокий риск отказа",
        'has_debit':             lambda v, s: "Дебетовая карта — средства могут закончиться",
        'high_risk_country':     lambda v, s: "Страна биллинга с высоким процентом отказов",
        'quiz_high_cost':        lambda v, s: "При регистрации указал что сервис слишком дорогой",
        'nsfw_rate':             lambda v, s: f"{v:.0%} генераций заблокировано как NSFW",
        'gen_trend':             lambda v, s: "Активность генераций резко падает со временем",
        'completion_rate':       lambda v, s: f"Только {v:.0%} генераций завершилось успешно",
        'avg_credit_cost':       lambda v, s: f"Средняя стоимость генерации: {v:.0f} кредитов",
        'total_purchases':       lambda v, s: "Ни одной покупки — низкая вовлечённость",
        'sub_updates':           lambda v, s: "Никогда не обновлял подписку",
        'credit_packs':          lambda v, s: "Постоянно покупает кредиты вместо подписки",
        'sub_tenure_days':       lambda v, s: f"Очень новый пользователь — {int(v)} дней",
        'plan_tier':             lambda v, s: "Базовый тариф — минимальная вовлечённость",
        'is_beginner':           lambda v, s: "Начинающий пользователь — сложный онбординг",
        'quiz_nsfw_frustration': lambda v, s: "Изначально недоволен ограничениями контента",
        'quiz_quality_issue':    lambda v, s: "Сообщил о проблемах с качеством при регистрации",
        'fail_count':            lambda v, s: f"{int(v)} неудачных попыток оплаты",
        'unique_models':         lambda v, s: f"Использовал только {int(v)} тип(а) моделей",
    }

    reasons = []
    raw_vals = X.iloc[idx]  # реальные значения признаков для этого юзера

    for shap_val, feat in pairs_sorted:
        if shap_val <= 0:
            continue
        if feat in templates:
            raw = raw_vals.get(feat, 0) if hasattr(raw_vals, 'get') else raw_vals[feat]
            reasons.append(templates[feat](raw, shap_val))
        if len(reasons) >= 3:
            break

    return reasons

print("  SHAP готов — причины будут браться из модели, не из правил.")

print("\n[5/6] Predicting...")

df['churn_prob'] = model1.predict_proba(X)[:, 1]
df['churn_type'] = 'not_churned'

at_risk = df['churn_prob'] > 0.5
if model2 is not None and at_risk.sum() > 0:
    invol_p = model2.predict_proba(df.loc[at_risk, FEATURES])[:, 1]
    df.loc[at_risk, 'churn_type'] = np.where(
        invol_p > 0.5, 'invol_churn', 'vol_churn')
elif at_risk.sum() > 0:
    df.loc[at_risk, 'churn_type'] = np.where(
        df.loc[at_risk, 'payment_fail_rate'] > 0.4, 'invol_churn', 'vol_churn')

print(f"  Results:\n{df['churn_type'].value_counts().to_string()}")

# ── 10. REASONS ───────────────────────────────────────────────────────────

def get_reasons(r, ctype):
    out = []
    if ctype == 'invol_churn':
        if r.get('payment_fail_rate', 0) > 0.3:
            out.append(f"{r['payment_fail_rate']:.0%} of payment attempts failed — card likely declined")
        if r.get('card_country_mismatch', 0) == 1:
            out.append("Card country differs from billing address — bank blocking")
        if r.get('has_prepaid', 0) == 1:
            out.append("Using a prepaid card — high decline risk for subscriptions")
        if r.get('high_risk_country', 0) == 1:
            out.append("Billing country has elevated card decline rate")
        if r.get('avg_credit_cost', 0) > 3000 and r.get('credit_packs', 0) == 0:
            out.append("Generates expensive content but hasn't bought credit packs")
        if r.get('quiz_high_cost', 0) == 1:
            out.append("Reported platform feels too expensive during onboarding")
        if r.get('sub_updates', 0) == 0 and r.get('total_purchases', 0) > 0:
            out.append("Never renewed or upgraded subscription")
    elif ctype == 'vol_churn':
        if r.get('nsfw_rate', 0) > 0.3:
            out.append(f"{r['nsfw_rate']:.0%} of generations blocked as NSFW — frustrated by content limits")
        if r.get('gen_trend', 1) < 0.5:
            out.append("Generation activity dropped significantly over time — disengaging")
        if r.get('completion_rate', 1) < 0.5 and r.get('gen_count', 0) > 5:
            out.append(f"Only {r['completion_rate']:.0%} of generations completed — too much friction")
        if r.get('quiz_quality_issue', 0) == 1:
            out.append("Reported quality or workflow issues during onboarding")
        if r.get('quiz_nsfw_frustration', 0) == 1:
            out.append("Explicitly frustrated by content restrictions at signup")
        if r.get('total_purchases', 0) == 0:
            out.append("Never made a purchase — low platform commitment")
        if r.get('is_beginner', 0) == 1:
            out.append("Beginner user — onboarding friction may have caused drop-off")
    return out[:3]

# ── 11. DISCOUNT (invol only) ─────────────────────────────────────────────

max_gen   = max(df['gen_count'].max(),   1)
max_spend = max(df['lifetime_spend'].max(), 1)

def calc_discount(r):
    tenure_score = min(r.get('sub_tenure_days', 0) / 365, 1.0)
    plan_score   = (min(r.get('plan_tier', 1), 4) - 1) / 3
    gen_score    = min(r.get('gen_count',   0) / 100, 1.0)
    spend_score  = min(r.get('lifetime_spend', 0) / 500, 1.0)
    loyalty = (tenure_score * 0.35 + plan_score * 0.30
             + gen_score    * 0.20 + spend_score * 0.15)
    return int(round(10 + loyalty * 30))

# ── 12. VOL CHURN OFFER ───────────────────────────────────────────────────

# Real generation_type values from test_users_generations.csv
ALL_GEN_TYPES = [
    'video_model_7', 'video_model_10', 'video_model_11',
    'video_model_12', 'video_model_13',
]

# Map first_feature values to human-readable names
FIRST_FEATURE_LABELS = {
    'Commercial & Ad Videos':       'Commercial & Ad Videos',
    'Video Generations':            'Video Generations',
    'video-creation':               'Video Creation',
    'Cinematic Visuals':            'Cinematic Visuals',
    'image-creation':               'Image Creation',
    'Viral Social Media Content':   'Viral Social Media Content',
    'Realistic AI Avatars':         'Realistic AI Avatars',
    'Image Editing & Inpaint':      'Image Editing & Inpainting',
    'consistent-character':         'Consistent Character',
    'Realistic Avatars & AI Twins': 'Realistic Avatars & AI Twins',
    'viral-effects':                'Viral Effects',
    'edit-image':                   'Image Editing',
    'product-placement':            'Product Placement',
    'draw-to-video':                'Draw to Video',
    'Storyboarding':                'Storyboarding',
    'talking-avatars':              'Talking Avatars',
    'Upscale':                      'Video Upscaling',
    'upscale':                      'Video Upscaling',
    'Lipsync & Talking Avatars':    'Lipsync & Talking Avatars',
}

ROLE_TO_FEATURE = {
    # Real role values from quiz data
    'filmmaker':        ('video_model_13', 'Cinematic Visuals — highest quality video model'),
    'creator':          ('video_model_12', 'Cinematic video for content creators'),
    'designer':         ('video_model_11', 'Fast video generation for design mockups'),
    'just-for-fun':     ('video_model_7',  'Short fun clips — easiest to start with'),
    'brand-owner':      ('video_model_12', 'Cinematic Visuals for brand content'),
    'marketer':         ('video_model_10', 'Commercial & Ad Videos'),
    'founder':          ('video_model_11', 'Fast product demo videos'),
    'educator':         ('video_model_11', 'Talking avatars for educational content'),
    'prompt-engineer':  ('video_model_13', 'Highest quality for prompt experimentation'),
    'developer':        ('video_model_10', 'Versatile model for API integration'),
    'product-lead':     ('video_model_11', 'Fast product demo videos'),
    'editor':           ('video_model_12', 'Cinematic video editing'),
}

def get_vol_offer(user_id, r, used_types_map, quiz_map):
    quiz         = quiz_map.get(user_id, {})
    role         = str(quiz.get('role', '')).strip().lower()
    first_feat   = quiz.get('first_feature', '')
    experience   = str(quiz.get('experience', '')).strip().lower()
    frustration  = str(quiz.get('frustration', '')).strip()
    nsfw_rate    = r.get('nsfw_rate', 0)
    completion   = r.get('completion_rate', 1)
    gen_trend    = r.get('gen_trend', 1)
    used         = used_types_map.get(user_id, [])
    unused       = [t for t in ALL_GEN_TYPES if t not in used]

    offers = []

    # 1. Много NSFW блокировок — раздражение от ограничений контента
    if nsfw_rate > 0.3:
        offers.append(
            f"Блокировки контента: {nsfw_rate:.0%} генераций заблокировано как NSFW. "
            f"Отправить гайд по правилам + показать что можно создавать в рамках платформы."
        )

    # 2. Низкий completion rate — слишком много правок, не получается
    if completion < 0.5 and r.get('gen_count', 0) > 5:
        frustration_hint = ""
        if frustration in ['hard-prompt', 'Hard to prompt']:
            frustration_hint = " Пользователь сам указал что промпты сложные."
        elif frustration in ['inconsistent', 'Inconsistent results']:
            frustration_hint = " Пользователь недоволен нестабильностью результатов."
        elif frustration in ['confusing', 'AI is confusing to me']:
            frustration_hint = " Пользователь указал что сервис непонятен."
        offers.append(
            f"Сложности с генерацией: только {completion:.0%} успешных результатов.{frustration_hint} "
            f"Предложить шаблоны промптов и туториал для роли '{role}'."
        )

    # 3. Пришёл за конкретной фичей но так и не попробовал её
    if first_feat and isinstance(first_feat, str) and first_feat not in str(used):
        feat_label = FIRST_FEATURE_LABELS.get(first_feat, first_feat)
        offers.append(
            f"Нереализованное ожидание: зарегистрировался ради '{feat_label}' "
            f"но так и не попробовал. Отправить туториал + бесплатные кредиты на первую попытку."
        )

    # 4. Предложить неиспользованные модели исходя из роли
    if unused and not offers:  # только если ещё нет офферов
        role_match = ROLE_TO_FEATURE.get(role)
        if role_match:
            model_name, model_desc = role_match
            if model_name in unused:
                offers.append(
                    f"Неиспользованная функция для '{role}': попробуй {model_desc} — "
                    f"ты ещё не использовал эту модель."
                )
            else:
                # Role's recommended model already used — suggest next unused
                offers.append(
                    f"Исследуй новые возможности: у тебя ещё не опробована модель {unused[0]}. "
                    f"Для {role} это может открыть новые форматы контента."
                )
        elif unused:
            offers.append(
                f"Новая возможность: модель {unused[0]} ещё не использовалась. "
                f"Отправить примеры и шаблон для первой попытки."
            )

    # 5. Активность падает + новичок
    if gen_trend < 0.5 and experience in ['beginner', 'intermediate']:
        offers.append(
            f"Снижение активности: пользователь ({experience}) генерирует всё меньше. "
            f"Предложить онбординг-сессию или пошаговый гайд по созданию первого проекта."
        )

    # Fallback если ничего не сработало
    if not offers:
        if unused:
            offers.append(
                f"Показать неиспользованные функции: {', '.join(unused[:2])}. "
                f"Отправить вдохновляющие примеры работ."
            )
        else:
            offers.append(
                "Реактивация: пользователь использовал все основные функции. "
                "Предложить скидку на годовую подписку или ранний доступ к новым моделям."
            )

    return offers[:2]

# ── 13. BUILD SUBMISSION ──────────────────────────────────────────────────
print("\n[6/6] Building submission file...")

# Build lookup maps
used_types_map = dict(zip(
    gen_f['user_id'],
    gen_f['used_gen_types'].apply(lambda x: x if isinstance(x, list) else [])
))
quiz_raw_dedup = quiz_raw.drop_duplicates(subset='user_id', keep='first')
quiz_map = quiz_raw_dedup.set_index('user_id').to_dict('index')

# Merge gen features back for discount calc
df = df.merge(
    gen_f[['user_id','gen_count']].rename(columns={'gen_count':'gen_count_for_disc'}),
    on='user_id', how='left'
)

rows = []
df = df.reset_index(drop=True)  # ensure clean integer index for SHAP
for idx, row in df.iterrows():
    r      = row.to_dict()
    uid    = row['user_id']
    ctype  = row['churn_type']

    reasons = get_shap_reasons(idx, ctype)
    discount = None
    action   = ""

    if ctype == 'invol_churn':
        discount = calc_discount(r)
        action = (
            f"Payment recovery: smart retry day 3/7/14. "
            f"Offer {discount}% personal loyalty discount (one-time only)."
        )
    elif ctype == 'vol_churn':
        offers = get_vol_offer(uid, r, used_types_map, quiz_map)
        action = " | ".join(offers) if offers else "Re-engagement: show unused features, offer plan downgrade."

    rows.append({
        'user_id':            uid,
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
print("\nInvol sample actions:")
invol_sample = submission[submission['churn_type']=='invol_churn'].head(3)
for _, r in invol_sample.iterrows():
    print(f"  {r['user_id'][:20]}... | discount: {r['discount_pct']}% | {r['reason_1']}")
print("\nVol sample actions:")
vol_sample = submission[submission['churn_type']=='vol_churn'].head(3)
for _, r in vol_sample.iterrows():
    print(f"  {r['user_id'][:20]}... | {r['recommended_action'][:80]}")
print("\nDone!")