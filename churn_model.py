import warnings
warnings.filterwarnings("ignore")
import glob
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    classification_report, f1_score, roc_auc_score, recall_score, precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ChurnType(str, Enum):
    NOT_CHURNED = "not_churned"
    VOL_CHURN   = "vol_churn"
    INVOL_CHURN = "invol_churn"


class CountryTier(int, Enum):
    LOW       = 0
    AVERAGE   = 1
    DEVELOPED = 2


@dataclass
class DatasetSignatures:

    users:   Set[str] = field(default_factory=lambda: {"churn_status"})
    gens:    Set[str] = field(default_factory=lambda: {"generation_id", "credit_cost", "generation_type"})
    props:   Set[str] = field(default_factory=lambda: {"subscription_start_date", "subscription_plan"})
    purch:   Set[str] = field(default_factory=lambda: {"purchase_type", "purchase_amount_dollars"})
    txn:     Set[str] = field(default_factory=lambda: {"failure_code", "card_brand", "card_funding"})
    quizzes: Set[str] = field(default_factory=lambda: {"frustration", "first_feature", "flow_type"})


@dataclass
class FrustrationGroups:

    high_cost:    frozenset = frozenset({"high-cost", "High cost of top models"})
    hard_prompt:  frozenset = frozenset({"hard-prompt", "Hard to prompt", "confusing", "AI is confusing to me"})
    inconsistent: frozenset = frozenset({"inconsistent", "Inconsistent results"})
    limited:      frozenset = frozenset({"limited", "Limited generations"})


@dataclass
class ModelRecommendation:
  
    model_id:    str
    description: str


@dataclass
class UserProfile:

    role:        str
    first_feat:  str
    experience:  str
    used_models: List[str]


@dataclass
class PredictionRow:
    user_id:            str
    churn_probability:  str
    churn_type:         str
    reason_1:           str
    reason_2:           str
    reason_3:           str
    discount_pct:       Optional[int]
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "user_id":            self.user_id,
            "churn_probability":  self.churn_probability,
            "churn_type":         self.churn_type,
            "reason_1":           self.reason_1,
            "reason_2":           self.reason_2,
            "reason_3":           self.reason_3,
            "discount_pct":       self.discount_pct,
            "recommended_action": self.recommended_action,
        }

FRUSTRATION = FrustrationGroups()

PLAN_TIER: Dict[str, int] = {
    "Higgsfield Basic":   1,
    "Higgsfield Pro":     2,
    "Higgsfield Creator": 3,
    "Higgsfield Ultimate":4,
    "Higgsfield Teams":   3,
}

TEAM_SIZE: Dict[str, int] = {
    "1": 1, "small": 3, "growing": 10, "midsize": 30, "enterprise": 100,
}

RESOLUTION_RANK: Dict[str, int] = {
    "480": 1, "480p": 1,
    "720": 2, "720p": 2, "768": 2,
    "1080": 3, "1080p": 3,
}


DEVELOPED_COUNTRIES: Set[str] = {
    "US","GB","DE","FR","NL","SE","NO","DK","FI","CH","AT","BE",
    "AU","NZ","CA","JP","SG","KR","HK","IE","LU","IS",
}
AVERAGE_COUNTRIES: Set[str] = {
    "BR","MX","AR","CL","CO","ZA","TR","PL","CZ","HU","RO","GR",
    "PT","IL","AE","SA","MY","TH","PH","ID","IN","CN",
}


ALL_GEN_TYPES: List[str] = [
    "image_model_1","image_model_2","image_model_3","image_model_4",
    "image_model_5","image_model_6","image_model_7","image_model_8",
    "image_model_9",
    "video_model_1","video_model_2","video_model_3","video_model_5",
    "video_model_6","video_model_7","video_model_8","video_model_9",
    "video_model_10","video_model_11","video_model_12","video_model_13",
]

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "image_model_1":  "Image Generation — high-quality photorealistic images",
    "image_model_2":  "Image Editing & Inpainting — modify and enhance images",
    "image_model_3":  "Image Editing — style transfer and creative edits",
    "image_model_4":  "Consistent Character — same character across multiple images",
    "image_model_5":  "Character Consistency — portrait and identity preservation",
    "image_model_6":  "Product Visuals — clean product shots and brand images",
    "image_model_7":  "Creative Image Generation — artistic and stylized output",
    "image_model_8":  "Image Upscaling — enhance resolution and quality",
    "image_model_9":  "AI Avatars & Realistic Portraits — lifelike human images",
    "video_model_1":  "Short Video Generation — fast clips for social media",
    "video_model_2":  "Video Generation — versatile short-form video",
    "video_model_3":  "Draw to Video — animate sketches and illustrations",
    "video_model_5":  "Viral Effects — trending visual effects for social content",
    "video_model_6":  "Creative Video — stylized and artistic video generation",
    "video_model_7":  "Quick Video Clips — lightweight fast generation",
    "video_model_8":  "Commercial Video — high-quality videos for ads and brands",
    "video_model_9":  "Talking Avatars & Lipsync — animated speaking characters",
    "video_model_10": "Versatile Video Generation — balanced quality and speed",
    "video_model_11": "Fast Video — quick generation for drafts and prototypes",
    "video_model_12": "Cinematic Video — premium quality for professional use",
    "video_model_13": "Cinematic Visuals — highest quality video on the platform",
}


FIRST_FEATURE_TO_MODELS: Dict[str, List[str]] = {
    "image-creation":               ["image_model_1","image_model_2","image_model_9"],
    "edit-image":                   ["image_model_2","image_model_3","image_model_6"],
    "Image Editing & Inpaint":      ["image_model_2","image_model_3","image_model_6"],
    "consistent-character":         ["image_model_4","image_model_5","image_model_9"],
    "Realistic AI Avatars":         ["image_model_4","image_model_9","video_model_9"],
    "Realistic Avatars & AI Twins": ["image_model_4","image_model_9","video_model_9"],
    "product-placement":            ["image_model_1","image_model_6","image_model_2"],
    "video-creation":               ["video_model_8","video_model_13","video_model_12"],
    "Video Generations":            ["video_model_8","video_model_3","video_model_13"],
    "Cinematic Visuals":            ["video_model_13","video_model_12","video_model_6"],
    "Commercial & Ad Videos":       ["video_model_8","video_model_12","video_model_3"],
    "Viral Social Media Content":   ["video_model_8","video_model_1","video_model_3"],
    "viral-effects":                ["video_model_8","video_model_1","video_model_5"],
    "draw-to-video":                ["video_model_3","video_model_6","video_model_8"],
    "talking-avatars":              ["video_model_9","video_model_11","image_model_4"],
    "Lipsync & Talking Avatars":    ["video_model_9","video_model_11","image_model_4"],
    "Storyboarding":                ["video_model_13","video_model_12","image_model_1"],
    "upscale":                      ["image_model_8","video_model_12","video_model_13"],
    "Upscale":                      ["image_model_8","video_model_12","video_model_13"],
}


ROLE_TO_MODELS: Dict[str, List[str]] = {
    "filmmaker":       ["video_model_13","video_model_12","video_model_6"],
    "creator":         ["video_model_8","image_model_1","video_model_3"],
    "designer":        ["image_model_1","image_model_2","image_model_9"],
    "just-for-fun":    ["video_model_8","image_model_1","video_model_1"],
    "brand-owner":     ["image_model_1","video_model_8","image_model_6"],
    "marketer":        ["video_model_8","image_model_6","video_model_3"],
    "founder":         ["video_model_8","image_model_1","video_model_11"],
    "educator":        ["video_model_9","video_model_11","image_model_4"],
    "prompt-engineer": ["video_model_13","image_model_9","video_model_12"],
    "developer":       ["video_model_3","image_model_2","video_model_8"],
    "photographer":    ["image_model_1","image_model_2","image_model_9"],
    "artist":          ["image_model_9","image_model_1","image_model_3"],
}

FIRST_FEATURE_LABELS: Dict[str, str] = {
    "video-creation":               "Video Creation",
    "image-creation":               "Image Creation",
    "edit-image":                   "Image Editing",
    "consistent-character":         "Consistent Character",
    "viral-effects":                "Viral Effects",
    "product-placement":            "Product Placement",
    "draw-to-video":                "Draw to Video",
    "talking-avatars":              "Talking Avatars",
    "upscale":                      "Video Upscaling",
    "Upscale":                      "Video Upscaling",
    "Commercial & Ad Videos":       "Commercial & Ad Videos",
    "Video Generations":            "Video Generations",
    "Cinematic Visuals":            "Cinematic Visuals",
    "Viral Social Media Content":   "Viral Social Media Content",
    "Realistic AI Avatars":         "Realistic AI Avatars",
    "Image Editing & Inpaint":      "Image Editing & Inpainting",
    "Realistic Avatars & AI Twins": "Realistic Avatars & AI Twins",
    "Storyboarding":                "Storyboarding",
    "Lipsync & Talking Avatars":    "Lipsync & Talking Avatars",
}

SHAP_TEMPLATES: Dict[str, callable] = {
    "payment_fail_rate":     lambda v: f"{v:.0%} of payment attempts failed — card likely declined",
    "fail_count":            lambda v: f"{int(v)} failed payment attempts in first 14 days",
    "card_country_mismatch": lambda v: "Card country differs from billing address — bank blocking",
    "has_prepaid":           lambda v: "Prepaid card — high subscription decline risk",
    "has_debit":             lambda v: "Debit card — balance may be insufficient at renewal",
    "high_risk_country":     lambda v: "Billing country has elevated card decline rate",
    "bank_country_tier":     lambda v: "Bank is in a lower-tier payment region",
    "cvc_missing_rate":      lambda v: f"{v:.0%} of transactions had missing CVC verification",
    "has_secure_fail":       lambda v: "3D Secure recommended but not authenticated",
    "uses_wallet":           lambda v: "Uses digital wallet (Apple/Google Pay) — reliable payment",
    "is_business":           lambda v: "Business card — corporate billing, high reliability",
    "quiz_high_cost":        lambda v: "Reported platform feels too expensive at signup",
    "quiz_hard_prompt":      lambda v: "Reported difficulty with prompting at signup",
    "quiz_inconsistent":     lambda v: "Reported inconsistent generation results at signup",
    "quiz_limited":          lambda v: "Reported feeling limited by generation quota",
    "nsfw_rate":             lambda v: f"{v:.0%} of generations blocked as NSFW",
    "frustration_score":     lambda v: f"Accumulated frustration score {v:.2f} from failed/nsfw gens",
    "gen_trend":             lambda v: "Generation activity declining over the observation period",
    "completion_rate":       lambda v: f"Only {v:.0%} of generations completed successfully",
    "avg_credit_cost":       lambda v: f"Average {v:.0f} credits per generation",
    "total_purchases":       lambda v: "No purchases in first 14 days — low platform commitment",
    "sub_updates":           lambda v: "No subscription upgrades — low engagement",
    "is_beginner":           lambda v: "Beginner user — onboarding friction risk",
    "days_to_first_gen":     lambda v: f"Took {int(v)} days after signup to make first generation",
    "avg_resolution":        lambda v: f"Uses low-resolution generations (avg rank {v:.1f}/3)",
    "plan_tier":             lambda v: f"On plan tier {int(v)}/4",
    "usage_limited":         lambda v: "Signed up with limited usage plan — low commitment",
    "avg_duration":          lambda v: f"Generates {v:.0f}s average video duration",
    "unique_models":         lambda v: f"Used only {int(v)} model type(s) — narrow exploration",
}

FEATURES: List[str] = [
    "payment_fail_rate", "fail_count", "txn_count",
    "card_country_mismatch", "has_prepaid", "has_debit",
    "high_risk_country", "bank_country_tier", "billing_country_tier",
    "total_spend_txn", "avg_spend_txn",
    "cvc_pass_rate", "cvc_missing_rate", "has_secure_fail",
    "is_business", "is_virtual", "uses_wallet",
    "total_purchases", "lifetime_spend", "avg_purchase",
    "sub_creates", "sub_updates", "credit_packs",
    "plan_tier", "user_country_tier",
    "gen_count", "completion_rate", "nsfw_rate", "fail_rate_gen",
    "avg_credit_cost", "total_credits", "unique_models",
    "gen_trend", "frustration_score",
    "avg_resolution", "max_resolution",
    "avg_duration", "max_duration",
    "days_to_first_gen",
    "quiz_high_cost", "quiz_hard_prompt", "quiz_inconsistent", "quiz_limited",
    "is_beginner", "is_expert", "is_invited", "is_personal",
    "usage_limited", "wants_video", "is_filmmaker_role",
    "from_youtube", "from_twitter", "from_instagram",
    "team_size_num",
]


def country_tier(code: str) -> int:
    if code in DEVELOPED_COUNTRIES:
        return CountryTier.DEVELOPED
    if code in AVERAGE_COUNTRIES:
        return CountryTier.AVERAGE
    return CountryTier.LOW


def detect_datasets(folder: str = ".") -> Dict[str, str]:

    sigs = DatasetSignatures()
    role_map = {
        "users":   sigs.users,
        "gens":    sigs.gens,
        "props":   sigs.props,
        "purch":   sigs.purch,
        "txn":     sigs.txn,
        "quizzes": sigs.quizzes,
    }
    found: Dict[str, str] = {}
    for path in glob.glob(os.path.join(folder, "*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=0, low_memory=False).columns)
        except Exception:
            continue
        for role, sig in role_map.items():
            if sig.issubset(cols) and role not in found:
                found[role] = path
                break
    return found


def compute_gen_trend(group: pd.DataFrame) -> float:

    if len(group) < 6:
        return 1.0
    group = group.sort_values("created_at")
    mid = len(group) // 2
    return len(group.iloc[mid:]) / max(len(group.iloc[:mid]), 1)


def get_shap_reasons(idx: int, shap_all: np.ndarray, X: pd.DataFrame) -> List[str]:

    sv    = shap_all[idx]
    row   = X.iloc[idx]
    pairs = sorted(zip(sv, X.columns), key=lambda x: x[0], reverse=True)
    reasons: List[str] = []
    for shap_val, feat in pairs:
        if shap_val <= 0 or feat not in SHAP_TEMPLATES:
            continue
        reasons.append(SHAP_TEMPLATES[feat](row[feat]))
        if len(reasons) == 3:
            break
    return reasons


def calc_discount(row: dict, max_purchases: float) -> int:

    plan    = (min(row.get("plan_tier", 1), 4) - 1) / 3
    purch   = min(row.get("total_purchases", 0) / max(max_purchases, 1), 1.0)
    credits = min(row.get("total_credits",   0) / 50000, 1.0)
    loyalty = plan * 0.45 + purch * 0.35 + credits * 0.20
    return int(round(10 + loyalty * 30))


def get_recommended_models(profile: UserProfile) -> List[str]:

    for m in FIRST_FEATURE_TO_MODELS.get(profile.first_feat, []):
        if m not in profile.used_models and m not in candidates:
            candidates.append(m)

    for m in ROLE_TO_MODELS.get(profile.role, []):
        if m not in profile.used_models and m not in candidates:
            candidates.append(m)

    fallback = ["image_model_1","image_model_2","video_model_8",
                "image_model_9","video_model_13","video_model_3"]
    for m in fallback:
        if m not in profile.used_models and m not in candidates:
            candidates.append(m)

    return candidates[:2]


def build_invol_action(row: dict, discount: int) -> str:
    parts = ["Payment recovery: smart retry on day 3, 7, 14."]
    if row.get("card_country_mismatch", 0) == 1:
        parts.append("Prompt user to update card to match billing country.")
    if row.get("has_prepaid", 0) == 1:
        parts.append("Suggest switching from prepaid to credit or debit card.")
    if row.get("bank_country_tier", 2) < 1:
        parts.append("Offer alternative payment methods (PayPal, local rails).")
    if row.get("has_secure_fail", 0) == 1:
        parts.append("Guide user to complete 3D Secure authentication.")
    parts.append(f"Offer {discount}% personal loyalty discount (one-time, never repeated).")
    return " ".join(parts)


def build_vol_action(profile: UserProfile, row: dict) -> str:
    actions: List[str] = []

    recommended = get_recommended_models(profile)
    if recommended:
        model_offers = [f"{m} ({MODEL_DESCRIPTIONS.get(m, m)})" for m in recommended]
        feat_label   = FIRST_FEATURE_LABELS.get(profile.first_feat, profile.first_feat)
        if feat_label and profile.first_feat not in str(profile.used_models):
            actions.append(
                f"Based on your goal '{feat_label}', try these Higgsfield models "
                f"you haven't used yet: {' / '.join(model_offers)}."
            )
        else:
            actions.append(
                f"Explore Higgsfield models you haven't tried yet: "
                f"{' / '.join(model_offers)}."
            )

    if row.get("nsfw_rate", 0) > 0.3:
        actions.append(
            f"NSFW friction ({row['nsfw_rate']:.0%} blocked): send content "
            "guidelines and showcase creative examples within platform rules."
        )
    elif row.get("completion_rate", 1) < 0.5 and row.get("gen_count", 0) > 5:
        actions.append(
            f"Only {row['completion_rate']:.0%} of generations succeeded — "
            "send prompt templates tailored to their use case."
        )

    return " | ".join(actions[:2]) if actions else (
        "Re-engagement: send curated Higgsfield model showcase "
        "with examples relevant to user role and content goals."
    )


def optuna_objective(
    trial: optuna.Trial,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
) -> float:
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
    m.fit(X_tr, y_tr)
    return f1_score(y_te, m.predict(X_te), average="macro")



def main() -> None:
    print("=" * 60)
    print("The Retention Architect")
    print("=" * 60)

    print("\n[1/7] Loading datasets...")
    detected = detect_datasets(".")
    print("  Detected files:")
    for role, path in detected.items():
        print(f"    {role:10s} -> {os.path.basename(path)}")

    missing = [r for r in ["users","gens","props","purch","txn","quizzes"]
               if r not in detected]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {missing}")

    users   = pd.read_csv(detected["users"],   low_memory=False)
    gens    = pd.read_csv(detected["gens"],    low_memory=False)
    props   = pd.read_csv(detected["props"],   low_memory=False)
    purch   = pd.read_csv(detected["purch"],   low_memory=False)
    txn     = pd.read_csv(detected["txn"],     low_memory=False)
    quizzes = pd.read_csv(detected["quizzes"], low_memory=False)

    print(f"  users={users.shape} props={props.shape} purch={purch.shape}")
    print(f"  txn={txn.shape} quizzes={quizzes.shape} gens={gens.shape}")
    
    print("\n[2/7] Engineering features...")

    gens["created_at"]  = pd.to_datetime(gens["created_at"], errors="coerce", utc=True)
    gens["credit_cost"] = pd.to_numeric(gens["credit_cost"], errors="coerce").fillna(0)
    gens["duration"]    = pd.to_numeric(gens["duration"],    errors="coerce").fillna(0)
    gens["resolution_rank"] = gens["resolution"].map(RESOLUTION_RANK).fillna(0)

    gen_agg = gens.groupby("user_id").agg(
        gen_count       = ("generation_id",   "count"),
        completed_count = ("status",          lambda x: (x == "completed").sum()),
        failed_count    = ("status",          lambda x: (x == "failed").sum()),
        nsfw_count      = ("status",          lambda x: (x == "nsfw").sum()),
        avg_credit_cost = ("credit_cost",     "mean"),
        total_credits   = ("credit_cost",     "sum"),
        unique_models   = ("generation_type", "nunique"),
        avg_resolution  = ("resolution_rank", "mean"),
        max_resolution  = ("resolution_rank", "max"),
        avg_duration    = ("duration",        "mean"),
        max_duration    = ("duration",        "max"),
    ).reset_index()

    gen_agg["completion_rate"]   = gen_agg["completed_count"] / gen_agg["gen_count"].clip(1)
    gen_agg["nsfw_rate"]         = gen_agg["nsfw_count"]      / gen_agg["gen_count"].clip(1)
    gen_agg["fail_rate_gen"]     = gen_agg["failed_count"]    / gen_agg["gen_count"].clip(1)
    gen_agg["frustration_score"] = (
        gen_agg["nsfw_count"] * 1.5 + gen_agg["failed_count"] * 1.0
    ) / gen_agg["gen_count"].clip(1)

    gen_trend = (
        gens.groupby("user_id")
        .apply(compute_gen_trend)
        .reset_index(name="gen_trend")
    )
    gen_agg = gen_agg.merge(gen_trend, on="user_id", how="left")

    
    props_dates = props[["user_id","subscription_start_date"]].copy()
    props_dates["subscription_start_date"] = pd.to_datetime(
        props_dates["subscription_start_date"], errors="coerce", utc=True
    )
    first_gen  = gens.groupby("user_id")["created_at"].min().reset_index(name="first_gen_date")
    activation = props_dates.merge(first_gen, on="user_id", how="left")
    activation["days_to_first_gen"] = (
        activation["first_gen_date"] - activation["subscription_start_date"]
    ).dt.days.fillna(14).clip(0, 14)
    gen_agg = gen_agg.merge(activation[["user_id","days_to_first_gen"]], on="user_id", how="left")

    used_types_map: Dict[str, List[str]] = (
        gens.groupby("user_id")["generation_type"]
        .apply(lambda x: list(x.dropna().unique()))
        .to_dict()
    )

    purch_agg = purch.groupby("user_id").agg(
        total_purchases = ("transaction_id",          "count"),
        lifetime_spend  = ("purchase_amount_dollars", "sum"),
        avg_purchase    = ("purchase_amount_dollars", "mean"),
        sub_creates     = ("purchase_type", lambda x: (x == "Subscription Create").sum()),
        sub_updates     = ("purchase_type", lambda x: (x == "Subscription Update").sum()),
        credit_packs    = ("purchase_type", lambda x: (x == "Credits package").sum()),
    ).reset_index()

    txn["bank_country_tier"]    = txn["bank_country"].apply(country_tier)
    txn["billing_country_tier"] = txn["billing_address_country"].apply(country_tier)
    txn["cvc_pass"]    = txn["cvc_check"].eq("pass").astype(int)
    txn["cvc_missing"] = txn["cvc_check"].isin(["not_provided","unavailable"]).astype(int)
    txn["secure_fail"] = (
        txn["card_3d_secure_support"].eq("recommended") &
        txn["is_3d_secure_authenticated"].astype(str).str.lower().eq("false")
    ).astype(int)
    txn["uses_wallet"] = txn["digital_wallet"].ne("none").astype(int)
    txn["mismatch"]    = (txn["billing_address_country"] != txn["card_country"]).astype(int)

    txn_agg = txn.groupby("user_id").agg(
        txn_count            = ("transaction_id",      "count"),
        fail_count           = ("failure_code",        lambda x: x.notna().sum()),
        total_spend_txn      = ("amount_in_usd",       "sum"),
        avg_spend_txn        = ("amount_in_usd",       "mean"),
        bank_country_tier    = ("bank_country_tier",   "max"),
        billing_country_tier = ("billing_country_tier","max"),
        is_business          = ("is_business",  lambda x: x.astype(str).str.lower().eq("true").any()),
        is_virtual           = ("is_virtual",   lambda x: x.astype(str).str.lower().eq("true").any()),
        has_prepaid          = ("is_prepaid",   lambda x: x.astype(str).str.lower().eq("true").any()),
        has_debit            = ("card_funding", lambda x: x.eq("debit").any()),
        cvc_pass_rate        = ("cvc_pass",     "mean"),
        cvc_missing_rate     = ("cvc_missing",  "mean"),
        has_secure_fail      = ("secure_fail",  "max"),
        uses_wallet          = ("uses_wallet",  "max"),
        card_country_mismatch= ("mismatch",     "max"),
    ).reset_index()

    txn_agg["payment_fail_rate"] = txn_agg["fail_count"] / txn_agg["txn_count"].clip(1)
    for col in ["is_business","is_virtual","has_prepaid","has_debit",
                "has_secure_fail","uses_wallet","card_country_mismatch"]:
        txn_agg[col] = txn_agg[col].astype(int)

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

    # Quizzes
    quizzes_dedup = quizzes.drop_duplicates(subset="user_id", keep="first")
    quiz_agg = quizzes_dedup[["user_id"]].copy()
    quiz_agg["quiz_high_cost"]    = quizzes_dedup["frustration"].isin(FRUSTRATION.high_cost).astype(int)
    quiz_agg["quiz_hard_prompt"]  = quizzes_dedup["frustration"].isin(FRUSTRATION.hard_prompt).astype(int)
    quiz_agg["quiz_inconsistent"] = quizzes_dedup["frustration"].isin(FRUSTRATION.inconsistent).astype(int)
    quiz_agg["quiz_limited"]      = quizzes_dedup["frustration"].isin(FRUSTRATION.limited).astype(int)
    quiz_agg["is_beginner"]       = (quizzes_dedup["experience"] == "beginner").astype(int)
    quiz_agg["is_expert"]         = quizzes_dedup["experience"].isin(["advanced","expert"]).astype(int)
    quiz_agg["is_invited"]        = (quizzes_dedup["flow_type"] == "invited").astype(int)
    quiz_agg["is_personal"]       = (quizzes_dedup["flow_type"] == "personal").astype(int)
    quiz_agg["usage_limited"]     = (quizzes_dedup["usage_plan"] == "limited").astype(int)
    quiz_agg["wants_video"]       = quizzes_dedup["first_feature"].str.contains(
        "video|Video|cinema|Cinema|commercial|Commercial|viral|Viral", na=False).astype(int)
    quiz_agg["is_filmmaker_role"] = quizzes_dedup["role"].str.contains(
        "film|Film|cinema|Cinema|creator|Creator|director|Director", na=False).astype(int)
    quiz_agg["from_youtube"]   = (quizzes_dedup["source"] == "youtube").astype(int)
    quiz_agg["from_twitter"]   = (quizzes_dedup["source"] == "twitter").astype(int)
    quiz_agg["from_instagram"] = (quizzes_dedup["source"] == "instagram").astype(int)
    quiz_agg["team_size_num"]  = quizzes_dedup["team_size"].map(TEAM_SIZE).fillna(1)

    quiz_raw: Dict[str, dict] = quizzes_dedup.set_index("user_id")[
        ["frustration","first_feature","role","experience"]
    ].to_dict("index")

    # Properties
    props["plan_tier"]        = props["subscription_plan"].map(PLAN_TIER).fillna(1)
    props["user_country_tier"]= props["country_code"].apply(country_tier)


    print("\n[3/7] Merging feature table...")
    df = users.copy()
    for frame in [
        purch_agg,
        txn_agg,
        quiz_agg,
        props[["user_id","plan_tier","user_country_tier"]],
        gen_agg.drop(columns=["gen_trend"], errors="ignore"),
        gen_trend,
    ]:
        df = df.merge(frame, on="user_id", how="left")

    df = df.reset_index(drop=True).fillna(0)
    active_features = [f for f in FEATURES if f in df.columns]
    print(f"  Shape: {df.shape}  |  Active features: {len(active_features)}")
    print(f"  Labels:\n{df['churn_status'].value_counts().to_string()}")

    print("\n[4/7] Training models...")

    df["churned"]  = (df["churn_status"] != ChurnType.NOT_CHURNED).astype(int)
    df["is_invol"] = (df["churn_status"] == ChurnType.INVOL_CHURN).astype(int)

    X  = df[active_features]
    y1 = df["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y1, test_size=0.2, random_state=42, stratify=y1
    )

    print("  Applying SMOTE...")
    X_tr_bal, y_tr_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr, y_tr)

    print("  Running Optuna (100 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_tr_bal, y_tr_bal, X_te, y_te),
        n_trials=100,
    )
    best = {**study.best_params, "eval_metric":"aucpr", "random_state":42, "verbosity":0}
    print(f"  Best params: {study.best_params}")
    print(f"  Best Macro F1: {study.best_value:.4f}")

    model1_base = xgb.XGBClassifier(**best)
    model1_base.fit(X_tr_bal, y_tr_bal)

    print("  Calibrating probabilities (isotonic regression)...")
    model1 = CalibratedClassifierCV(model1_base, method="isotonic", cv="prefit")
    model1.fit(X_te, y_te)   # calibrate on test set

    y_pred1 = model1.predict(X_te)
    y_prob1 = model1.predict_proba(X_te)[:, 1]

    best_threshold = 0.5
    best_recall    = 0.0
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (y_prob1 >= t).astype(int)
        rec   = recall_score(y_te, preds)
        prec  = precision_score(y_te, preds, zero_division=0)
        if prec >= 0.50 and rec > best_recall:
            best_recall    = rec
            best_threshold = round(t, 2)

    print(f"  Optimal threshold: {best_threshold} (maximizes churn recall, precision >= 0.50)")
    y_pred1_tuned = (y_prob1 >= best_threshold).astype(int)

    print("\n  --- Stage 1: Churn Detection ---")
    print(classification_report(y_te, y_pred1_tuned, target_names=["Not Churned","Churned"]))
    print(f"  Accuracy:      {accuracy_score(y_te, y_pred1_tuned):.4f}")
    print(f"  Churn Recall:  {recall_score(y_te, y_pred1_tuned):.4f}  ← catching churners is priority")
    print(f"  Churn Precision:{precision_score(y_te, y_pred1_tuned):.4f}")
    print(f"  ROC-AUC:       {roc_auc_score(y_te, y_prob1):.4f}")
    print(f"  PR-AUC:        {average_precision_score(y_te, y_prob1):.4f}")
    print(f"  Macro F1:      {f1_score(y_te, y_pred1_tuned, average='macro'):.4f}")
    print(f"  Churn F1:      {f1_score(y_te, y_pred1_tuned, average='binary'):.4f}  ← churn class only")

    ch_mask = df["churned"] == 1
    X2, y2  = df.loc[ch_mask, active_features], df.loc[ch_mask, "is_invol"]
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
    print(classification_report(y2_te, y_pred2, target_names=["Voluntary","Involuntary"]))
    print(f"  Accuracy: {accuracy_score(y2_te, y_pred2):.4f}")
    print(f"  ROC-AUC : {roc_auc_score(y2_te, y_prob2):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y2_te, y_prob2):.4f}")
    print(f"  Macro F1: {f1_score(y2_te, y_pred2, average='macro'):.4f}")

    print("\n[5/7] Computing SHAP values...")
    explainer = shap.TreeExplainer(model1)
    shap_te   = explainer.shap_values(X_te)
    shap_all  = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_te, X_te, feature_names=active_features, show=False)
    plt.title("Feature Importance — SHAP Summary", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary.png")

    print("\n[6/7] Generating predictions...")
    df["churn_prob"] = model1.predict_proba(X)[:, 1]
    df["churn_type"] = ChurnType.NOT_CHURNED.value

    at_risk = df["churn_prob"] >= best_threshold  # use tuned threshold, not default 0.5
    invol_p = model2.predict_proba(df.loc[at_risk, active_features])[:, 1]
    df.loc[at_risk, "churn_type"] = np.where(
        invol_p > 0.5, ChurnType.INVOL_CHURN.value, ChurnType.VOL_CHURN.value
    )
    print(f"  Distribution:\n{df['churn_type'].value_counts().to_string()}")

    print("\n[7/7] Building submission...")
    max_purchases = max(df["total_purchases"].max(), 1)

    output_rows: List[PredictionRow] = []

    for idx, row in df.iterrows():
        r     = row.to_dict()
        uid   = row["user_id"]
        ctype = row["churn_type"]

        reasons  = get_shap_reasons(idx, shap_all, X)
        discount = None
        action   = ""

        if ctype == ChurnType.INVOL_CHURN.value:
            discount = calc_discount(r, max_purchases)
            action   = build_invol_action(r, discount)

        elif ctype == ChurnType.VOL_CHURN.value:
            q = quiz_raw.get(uid, {})
            profile = UserProfile(
                role        = str(q.get("role", "")).strip().lower(),
                first_feat  = str(q.get("first_feature", "")) if q.get("first_feature") else "",
                experience  = str(q.get("experience", "")).strip().lower(),
                used_models = used_types_map.get(uid, []),
            )
            action = build_vol_action(profile, r)

        output_rows.append(PredictionRow(
            user_id            = uid,
            churn_probability  = str(round(float(row["churn_prob"]) * 100, 1)) + "%",
            churn_type         = ctype,
            reason_1           = reasons[0] if len(reasons) > 0 else "",
            reason_2           = reasons[1] if len(reasons) > 1 else "",
            reason_3           = reasons[2] if len(reasons) > 2 else "",
            discount_pct       = discount,
            recommended_action = action,
        ))

    submission = pd.DataFrame([r.to_dict() for r in output_rows])
    submission.to_csv("predictions.csv", index=False)

    print(f"\n  Saved: predictions.csv ({len(submission)} rows)")
    print("\n  Sample invol:")
    for _, r in submission[submission["churn_type"] == ChurnType.INVOL_CHURN.value].head(3).iterrows():
        print(f"    prob={r['churn_probability']}  disc={r['discount_pct']}%")
        print(f"    {r['reason_1']}")
        print(f"    {r['recommended_action'][:90]}")
        print()
    print("  Sample vol:")
    for _, r in submission[submission["churn_type"] == ChurnType.VOL_CHURN.value].head(3).iterrows():
        print(f"    prob={r['churn_probability']}")
        print(f"    {r['reason_1']}")
        print(f"    {r['recommended_action'][:90]}")
        print()
    print("Done.")


if __name__ == "__main__":
    main()