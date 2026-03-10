#!/usr/bin/env python3
"""
MALLORN: FOCUSED VARIANTS - Fine-Grained Training

After diagnostic identifies:
1. Best feature set (core vs all)
2. Best CV strategy (GroupKFold)
3. Best thresholding approach

Generate only the most promising variants:
- Top 3 pseudo-labeling thresholds (based on diagnostic)
- Top 3 weight combinations (CatBoost-only, best ensemble, conservative)
- Fine-grained optimization:
  * 150 Optuna trials (was 30) for better hyperparameter search
  * Additional hyperparameters (min_child_weight, gamma, max_delta_step for XGBoost)
  * Finer threshold optimization (0.005 steps, then 0.001 around optimum)
  * PR curve + fine-grained search for ensemble threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import multiprocessing
import time
import shutil
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("ERROR: CatBoost not available!")
    exit(1)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("WARNING: LightGBM not available! Install with: pip install lightgbm")

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

DATA_DIR = Path("mallorn-astronomical-classification-challenge")
CACHE_DIR = DATA_DIR / "template_cache"
N_SPLITS = 5  # Full CV for final training
N_SPLITS_TUNING = 3  # Faster CV for hyperparameter tuning
SEED = 42
# Phase A: Efficient tuning with pruners (same compute, better output)
# Allow override via environment variable for testing different trial counts
N_TRIALS = int(os.environ.get('N_TRIALS', '150'))  # Keep at 150 with pruners (Phase A recommendation)
USE_PRUNER = True  # Use HyperbandPruner for efficient early stopping
PRUNER_TYPE = 'hyperband'  # 'hyperband' or 'median'

# Checkpoint directory on external disk
TRANSCEND_DIR = Path("/Volumes/TRANSCEND")
if TRANSCEND_DIR.exists():
    CHECKPOINT_DIR = TRANSCEND_DIR / "kaggle_astro_checkpoints"
    try:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        # Test write access
        test_file = CHECKPOINT_DIR / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✅ Using TRANSCEND disk for checkpoints: {CHECKPOINT_DIR}")
    except Exception as e:
        CHECKPOINT_DIR = Path("checkpoints")
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        print(f"⚠️  TRANSCEND write failed ({e}), using local: {CHECKPOINT_DIR}")
else:
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"⚠️  TRANSCEND not found, using local checkpoints: {CHECKPOINT_DIR}")

# Disk space check (minimum 10GB free)
def check_disk_space(path=".", min_gb=10):
    """Check if enough disk space is available"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        print(f"⚠️  WARNING: Only {free_gb:.1f} GB free (need {min_gb} GB)")
        print("   Training may fail with 'No space left on device' error")
        print("   Consider: cleaning catboost_info/, moving to external disk, or freeing space")
        return False
    return True

# Cleanup function
def cleanup_temp_files():
    """Remove temporary files to free space"""
    catboost_dir = Path("catboost_info")
    if catboost_dir.exists():
        try:
            shutil.rmtree(catboost_dir)
            print("  ✅ Cleaned catboost_info/")
        except:
            pass

# FOCUSED: Only most promising combinations
# Will be updated after diagnostic runs
# Allow override via environment
PSEUDO_THRESHOLDS_STR = os.environ.get('PSEUDO_THRESHOLDS', '')
if PSEUDO_THRESHOLDS_STR:
    # Parse from env: "0.85,0.15;0.80,0.20"
    PSEUDO_THRESHOLDS = [tuple(map(float, p.split(','))) for p in PSEUDO_THRESHOLDS_STR.split(';')]
else:
    PSEUDO_THRESHOLDS = [
        (0.85, 0.15),  # Stricter (achieved 0.6519) - focus on this
        (0.80, 0.20),  # Golden (0.6298)
    ]

# Use STACKING (meta-learner) as primary approach
# Stacking learns optimal combination automatically from OOF predictions
# Much better than brute-forcing 17 weight combinations

# Manual weight combinations (fallback/comparison only - focused on known good)
# w28 (0.2, 0.8, 0.0) scored 0.6519 on LB - keep testing this family
# Allow override via environment
WEIGHTS_STR = os.environ.get('WEIGHT_COMBINATIONS', '')
if WEIGHTS_STR:
    # Parse from env: "0.2,0.8,0.0;0.0,1.0,0.0"
    WEIGHT_COMBINATIONS_MANUAL = [tuple(map(float, w.split(','))) for w in WEIGHTS_STR.split(';')]
else:
    WEIGHT_COMBINATIONS_MANUAL = [
        (0.2, 0.8, 0.0),  # w28 - proven winner (0.6519) - KEEP THIS
    ]

# For compatibility with existing code
WEIGHT_COMBINATIONS_3MODEL = WEIGHT_COMBINATIONS_MANUAL

print("="*70)
print("MALLORN: FOCUSED VARIANTS - Fine-Grained Training")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CPU cores: {multiprocessing.cpu_count()} | Parallel Optuna: {min(8, multiprocessing.cpu_count())} workers")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# Check disk space
if not check_disk_space():
    print("\n⚠️  Continuing anyway, but monitor disk space...")
    cleanup_temp_files()
print()

# Progress tracking
class TrainingProgress:
    def __init__(self):
        self.start_time = time.time()
        self.stage = ""
        self.trial_count = 0
        
    def update_stage(self, stage_name):
        self.stage = stage_name
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"[{elapsed/60:.1f}m] {stage_name}")
        print(f"{'='*70}")
        
    def log_trial(self, model_name, trial_num, total_trials, best_f1):
        self.trial_count += 1
        elapsed = time.time() - self.start_time
        pct = (trial_num / total_trials) * 100
        print(f"  {model_name}: Trial {trial_num}/{total_trials} ({pct:.1f}%) | Best F1: {best_f1:.4f} | Elapsed: {elapsed/60:.1f}m", end='\r')
        
    def save_checkpoint(self, data, filename):
        """Save checkpoint to external disk"""
        try:
            checkpoint_path = CHECKPOINT_DIR / filename
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"  ⚠️  Failed to save checkpoint: {e}")
            
    def load_checkpoint(self, filename):
        """Load checkpoint from external disk"""
        try:
            checkpoint_path = CHECKPOINT_DIR / filename
            if checkpoint_path.exists():
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
        return None

progress = TrainingProgress()
print("Strategy:")
print("  1. Use GroupKFold (fix leakage)")
print("  2. Remove FFT features (diagnostic: improves CV F1 by +0.01)")
print("  3. NO calibration (diagnostic: hurts performance)")
print("  4. PR curve thresholding (diagnostic: best method)")
print("  5. Fine-grained optimization:")
print("     - 150 Optuna trials per model (was 30)")
print("     - Additional hyperparameters (min_child_weight, gamma, etc.)")
print("     - Finer threshold search (0.005 steps, then 0.001 around optimum)")
print("  6. Only 3 pseudo-thresholds × 3 weights = 9 variants")
print()

# Load features (FULL SET)
print("[1] Loading features...")
train_log = pd.read_csv(DATA_DIR / "train_log.csv")
test_log = pd.read_csv(DATA_DIR / "test_log.csv")

train_feats = pd.read_parquet(CACHE_DIR / "train_comprehensive_features.parquet")
test_feats = pd.read_parquet(CACHE_DIR / "test_comprehensive_features.parquet")
train_template = pd.read_parquet(CACHE_DIR / "train_template_features_fixed.parquet")
test_template = pd.read_parquet(CACHE_DIR / "test_template_features_fixed.parquet")
train_upgraded = pd.read_parquet(CACHE_DIR / "train_upgraded_features.parquet")
test_upgraded = pd.read_parquet(CACHE_DIR / "test_upgraded_features.parquet")

# Load paper features (if available)
train_paper_path = CACHE_DIR / "train_paper_features.parquet"
test_paper_path = CACHE_DIR / "test_paper_features.parquet"
if train_paper_path.exists() and test_paper_path.exists():
    print("  Loading paper features...")
    train_paper = pd.read_parquet(train_paper_path)
    test_paper = pd.read_parquet(test_paper_path)
    print(f"    Paper features: {len(train_paper.columns) - 1}")
else:
    print("  ⚠️  Paper features not found. Run mallorn_paper_features.py first.")
    train_paper = pd.DataFrame({'object_id': train_log['object_id']})
    test_paper = pd.DataFrame({'object_id': test_log['object_id']})

# Load SNcosmo + Blackbody features (if available)
train_sncosmo_bb_path = CACHE_DIR / "train_sncosmo_blackbody_features.parquet"
test_sncosmo_bb_path = CACHE_DIR / "test_sncosmo_blackbody_features.parquet"
if train_sncosmo_bb_path.exists() and test_sncosmo_bb_path.exists():
    print("  Loading SNcosmo + Blackbody features...")
    train_sncosmo_bb = pd.read_parquet(train_sncosmo_bb_path)
    test_sncosmo_bb = pd.read_parquet(test_sncosmo_bb_path)
    print(f"    SNcosmo + Blackbody features: {len(train_sncosmo_bb.columns) - 1}")
else:
    print("  ⚠️  SNcosmo + Blackbody features not found. Run sncosmo_blackbody_features.py first.")
    train_sncosmo_bb = pd.DataFrame({'object_id': train_log['object_id']})
    test_sncosmo_bb = pd.DataFrame({'object_id': test_log['object_id']})

# Load TDE discriminative features (if available)
train_tde_disc_path = CACHE_DIR / "train_tde_discriminative_features.parquet"
test_tde_disc_path = CACHE_DIR / "test_tde_discriminative_features.parquet"
if train_tde_disc_path.exists() and test_tde_disc_path.exists():
    print("  Loading TDE discriminative features...")
    train_tde_disc = pd.read_parquet(train_tde_disc_path)
    test_tde_disc = pd.read_parquet(test_tde_disc_path)
    print(f"    TDE discriminative features: {len(train_tde_disc.columns) - 1}")
else:
    print("  ⚠️  TDE discriminative features not found. Run tde_discriminative_features.py first.")
    train_tde_disc = pd.DataFrame({'object_id': train_log['object_id']})
    test_tde_disc = pd.DataFrame({'object_id': test_log['object_id']})

# Feature selection flags (set via environment or command line) - MUST BE BEFORE USE
USE_SNCOSMO_BB = os.environ.get('USE_SNCOSMO_BB', 'true').lower() == 'true'
USE_TDE_DISC = os.environ.get('USE_TDE_DISC', 'true').lower() == 'true'
USE_POWERLAW = os.environ.get('USE_POWERLAW', 'false').lower() == 'true'  # A2: Power-law features
USE_BAZIN = os.environ.get('USE_BAZIN', 'false').lower() == 'true'  # Bazin parametric fit features

# Load power-law features (A2) if requested
train_powerlaw_path = CACHE_DIR / "train_powerlaw_features.parquet"
test_powerlaw_path = CACHE_DIR / "test_powerlaw_features.parquet"
if USE_POWERLAW:
    if train_powerlaw_path.exists() and test_powerlaw_path.exists():
        print("  Loading power-law decay features...")
        train_powerlaw = pd.read_parquet(train_powerlaw_path)
        test_powerlaw = pd.read_parquet(test_powerlaw_path)
        print(f"    Power-law features: {len(train_powerlaw.columns) - 1}")
    else:
        print("  ⚠️  Power-law features not found. Run powerlaw_decay_features.py first.")
        train_powerlaw = pd.DataFrame({'object_id': train_log['object_id']})
        test_powerlaw = pd.DataFrame({'object_id': test_log['object_id']})
else:
    train_powerlaw = pd.DataFrame({'object_id': train_log['object_id']})
    test_powerlaw = pd.DataFrame({'object_id': test_log['object_id']})

# Load Bazin features if requested
train_bazin_path = CACHE_DIR / "train_bazin_features.parquet"
test_bazin_path = CACHE_DIR / "test_bazin_features.parquet"
if USE_BAZIN:
    if train_bazin_path.exists() and test_bazin_path.exists():
        print("  Loading Bazin parametric fit features...")
        train_bazin = pd.read_parquet(train_bazin_path)
        test_bazin = pd.read_parquet(test_bazin_path)
        print(f"    Bazin features: {len(train_bazin.columns) - 1}")
    else:
        print("  ⚠️  Bazin features not found. Run bazin_features.py first.")
        train_bazin = pd.DataFrame({'object_id': train_log['object_id']})
        test_bazin = pd.DataFrame({'object_id': test_log['object_id']})
else:
    train_bazin = pd.DataFrame({'object_id': train_log['object_id']})
    test_bazin = pd.DataFrame({'object_id': test_log['object_id']})

# Additional environment variable controls for systematic testing
USE_FEATURE_SELECTION = os.environ.get('USE_FEATURE_SELECTION', 'true').lower() == 'true'
FEATURE_SELECTION_PERCENTILE = int(os.environ.get('FEATURE_SELECTION_PERCENTILE', '10'))
USE_STACKING = os.environ.get('USE_STACKING', 'true').lower() == 'true'
USE_ENHANCED_STACKING = os.environ.get('USE_ENHANCED_STACKING', 'true').lower() == 'true'
PSEUDO_WEIGHT_METHOD = os.environ.get('PSEUDO_WEIGHT_METHOD', 'confidence').lower()  # 'fixed' or 'confidence'
TEST_ID = os.environ.get('TEST_ID', '')  # Optional test identifier for submission filenames

# B1-B3: Tree count and regularization controls
XGB_TREES = int(os.environ.get('XGB_TREES', '0'))  # 0 = use default (2000), else override
CAT_TREES = int(os.environ.get('CAT_TREES', '0'))  # 0 = use default (4000), else override
REG_MULTIPLIER = float(os.environ.get('REG_MULTIPLIER', '1.0'))  # B2: Multiply regularization by this
XGB_SIMPLER = os.environ.get('XGB_SIMPLER', 'false').lower() == 'true'  # B3: Make XGB simpler

# C1-C3: Pseudo-labeling controls
PSEUDO_ITERATIONS = int(os.environ.get('PSEUDO_ITERATIONS', '1'))  # C1: Number of pseudo-label iterations
PSEUDO_SOURCE = os.environ.get('PSEUDO_SOURCE', 'ensemble').lower()  # C3: 'ensemble' or 'catboost'

# A3-A4: Feature exclusion
EXCLUDE_FEATURES = os.environ.get('EXCLUDE_FEATURES', '')  # Comma-separated list of features to exclude

train = train_log[['object_id', 'target']].merge(train_feats, on='object_id', how='left')
train = train.merge(train_template, on='object_id', how='left')
train = train.merge(train_upgraded, on='object_id', how='left', suffixes=('', '_upg'))
train = train.merge(train_paper, on='object_id', how='left', suffixes=('', '_paper'))
if USE_SNCOSMO_BB:
    train = train.merge(train_sncosmo_bb, on='object_id', how='left', suffixes=('', '_sncosmo_bb'))
if USE_TDE_DISC:
    train = train.merge(train_tde_disc, on='object_id', how='left', suffixes=('', '_tde_disc'))
if USE_POWERLAW:
    train = train.merge(train_powerlaw, on='object_id', how='left', suffixes=('', '_powerlaw'))
if USE_BAZIN:
    train = train.merge(train_bazin, on='object_id', how='left', suffixes=('', '_bazin'))

test = test_log[['object_id']].merge(test_feats, on='object_id', how='left')
test = test.merge(test_template, on='object_id', how='left')
test = test.merge(test_upgraded, on='object_id', how='left', suffixes=('', '_upg'))
test = test.merge(test_paper, on='object_id', how='left', suffixes=('', '_paper'))
if USE_SNCOSMO_BB:
    test = test.merge(test_sncosmo_bb, on='object_id', how='left', suffixes=('', '_sncosmo_bb'))
if USE_TDE_DISC:
    test = test.merge(test_tde_disc, on='object_id', how='left', suffixes=('', '_tde_disc'))
if USE_POWERLAW:
    test = test.merge(test_powerlaw, on='object_id', how='left', suffixes=('', '_powerlaw'))
if USE_BAZIN:
    test = test.merge(test_bazin, on='object_id', how='left', suffixes=('', '_bazin'))

# ENHANCED: Better data cleaning (use median/mean instead of 0 for missing values)
for df in [train, test]:
    cols_to_drop = []
    for col in df.columns:
        if col not in ['object_id', 'target']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Remove features with >90% missing (negligible data)
            if df[col].isna().sum() / len(df) > 0.9:
                cols_to_drop.append(col)
                continue
            
            # Use median for missing values (better than 0, preserves distribution)
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                if pd.isna(median_val) or np.isinf(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
    
    # Drop columns with too many missing values
    if cols_to_drop:
        print(f"    Removing {len(cols_to_drop)} features with >90% missing: {cols_to_drop[:5]}...")
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

y = train['target'].astype(int)
feature_cols = [c for c in train.columns if c not in ['object_id', 'target']]

# Calculate scale_pos_weight (needed for feature selection and training)
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# REMOVE FFT FEATURES (diagnostic showed they overfit)
fft_features = [c for c in feature_cols if 'fft' in c.lower() or 'freq' in c.lower()]
print(f"  Removing {len(fft_features)} FFT features (diagnostic: improves CV F1)")
feature_cols = [c for c in feature_cols if c not in fft_features]

# A3: Exclude specified features (for harmful feature ablation)
if EXCLUDE_FEATURES:
    exclude_list = [f.strip() for f in EXCLUDE_FEATURES.split(',')]
    excluded = [c for c in feature_cols if c in exclude_list]
    feature_cols = [c for c in feature_cols if c not in exclude_list]
    if excluded:
        print(f"  Excluding {len(excluded)} features (A3 ablation): {excluded[:5]}...")

# ENHANCED: Feature selection - remove low-importance noisy features
if USE_FEATURE_SELECTION:
    print(f"  Performing feature selection (removing bottom {FEATURE_SELECTION_PERCENTILE}% by importance)...")
    X_temp = train[feature_cols]
    selector_model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=200,
        scale_pos_weight=scale_pos_weight, random_state=SEED,
        tree_method='hist', verbosity=0
    )
    selector_model.fit(X_temp, y)
    importances = selector_model.feature_importances_
    threshold = np.percentile(importances, FEATURE_SELECTION_PERCENTILE)  # Remove bottom X%
    selected_mask = importances >= threshold
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    print(f"  Selected {len(selected_features)} / {len(feature_cols)} features (removed {len(feature_cols) - len(selected_features)} low-importance)")
    feature_cols = selected_features
else:
    print(f"  Using all {len(feature_cols)} features (feature selection disabled)")

# VERIFY PHYSICS FEATURES ARE INCLUDED
physics_features = [c for c in feature_cols if any(x in c.lower() for x in ['rest', 'z_phot', 'some_color', '_sn', 'bb_', 'sn_'])]
print(f"  ✅ Paper/physics features: {len(physics_features)} (includes rest-frame, photo-z, some_color, SNcosmo, blackbody)")
if len(physics_features) > 0:
    print(f"     Examples: {physics_features[:5]}")

X = train[feature_cols]
X_test = test[feature_cols]

# Memory optimization: convert to float32 (saves ~50% memory)
print("  Converting to float32 for memory efficiency...")
for col in X.columns:
    if X[col].dtype == 'float64':
        X[col] = X[col].astype('float32')
    if X_test[col].dtype == 'float64':
        X_test[col] = X_test[col].astype('float32')

print(f"  Train: {len(X)} objects ({y.sum()} TDEs, {100*y.sum()/len(y):.1f}%)")
print(f"  Test: {len(X_test)} objects")
print(f"  Features: {len(feature_cols)}")
print()

# Use GroupKFold (fix leakage)
gkf = GroupKFold(n_splits=N_SPLITS)
groups = train['object_id'].values
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Train initial models (quick tuning)
progress.update_stage("[2] Training initial models with GroupKFold")
cleanup_temp_files()  # Clean before training

if OPTUNA_AVAILABLE:
    print(f"  Tuning XGBoost ({N_TRIALS} trials - optimized for speed)...")
    print(f"    ⚡ Parallel: {min(8, multiprocessing.cpu_count())} workers | CV folds: {N_SPLITS_TUNING} (tuning) / {N_SPLITS} (final) | Estimators: 1000 (tuning) / 2000 (final)")
    start_time = time.time()
    elapsed = 0  # Initialize elapsed
    def objective_xgb(trial):
        # Phase A: Tightened search space based on current best params
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 8),  # Tightened (best ~6-7)
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),  # Tightened (best ~0.03-0.1)
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 15, log=True),  # Tightened (best ~2-8)
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-6, 2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5, log=True),  # Tightened (best ~0.01-1)
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5, log=True),  # Tightened (best ~0.01-2)
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                   scale_pos_weight * 0.7,
                                                   scale_pos_weight * 1.5,
                                                   log=True),
            'random_state': SEED,
            'tree_method': 'hist',
            'verbosity': 0,
            'n_estimators': 1000  # Reduced for faster tuning
        }
        # Phase A: Report intermediate values for pruning (early stopping per fold)
        gkf_tuning = GroupKFold(n_splits=N_SPLITS_TUNING)
        oof = np.zeros(len(X))
        fold_f1s = []
        
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf_tuning.split(X, y, groups)):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                verbose=False
            )
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            
            # Calculate F1 for this fold (for early stopping/pruning)
            fold_precision, fold_recall, fold_thresholds = precision_recall_curve(y.iloc[va_idx], oof[va_idx])
            fold_f1_scores = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall + 1e-10)
            fold_f1 = np.max(fold_f1_scores)
            fold_f1s.append(fold_f1)
            
            # Report intermediate value for pruning (after first fold)
            if fold_idx == 0 and USE_PRUNER:
                trial.report(fold_f1, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Final: OOF F1 after threshold selection on all folds
        precision, recall, thresholds = precision_recall_curve(y, oof)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores)
        
        # Report final value for pruning
        if USE_PRUNER:
            trial.report(best_f1, step=N_SPLITS_TUNING)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_f1
    
    # Use Optuna SQLite storage for automatic persistence
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    storage_url = f"sqlite:///{CHECKPOINT_DIR}/xgb_study.db"
    try:
        study_xgb = optuna.load_study(study_name='xgb_f1_fine', storage=storage_url)
        start_trial = len(study_xgb.trials)
        if start_trial > 0:
            try:
                best_val = study_xgb.best_value
                print(f"    📂 Resuming XGBoost tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming XGBoost tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new XGBoost tuning ({N_TRIALS} trials)")
    except:
        # Phase A: Add pruner for efficient early stopping
        pruner = None
        if USE_PRUNER and OPTUNA_AVAILABLE:
            if PRUNER_TYPE == 'hyperband':
                pruner = HyperbandPruner(
                    min_resource=1,  # Start pruning after 1 fold
                    max_resource=N_SPLITS_TUNING,  # Max resource = number of folds
                    reduction_factor=3  # Aggressiveness (higher = more aggressive)
                )
            else:  # median
                pruner = MedianPruner(
                    n_startup_trials=5,  # Don't prune first 5 trials
                    n_warmup_steps=1,  # Wait 1 step before pruning
                    interval_steps=1  # Check every step
                )
        
        study_xgb = optuna.create_study(
            direction='maximize', 
            study_name='xgb_f1_fine',
            storage=storage_url,
            load_if_exists=True,
            pruner=pruner
        )
        start_trial = len(study_xgb.trials)
        if start_trial > 0:
            try:
                best_val = study_xgb.best_value
                print(f"    📂 Resuming XGBoost tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming XGBoost tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new XGBoost tuning ({N_TRIALS} trials)")
    
    if start_trial >= N_TRIALS:
        print(f"    ✅ XGBoost tuning already complete ({start_trial}/{N_TRIALS} trials)")
        try:
            best_val = study_xgb.best_value
            print(f"    Best CV F1: {best_val:.4f}")
        except:
            pass
    else:
        progress.update_stage(f"Tuning XGBoost ({N_TRIALS} trials)")
        study_xgb.optimize(
            objective_xgb, 
            n_trials=N_TRIALS - start_trial,
            n_jobs=min(8, multiprocessing.cpu_count()), 
            show_progress_bar=True
        )
    if start_trial < N_TRIALS:
        elapsed = time.time() - start_time
        print(f"\n    ✅ XGBoost tuning complete: {elapsed/60:.1f} min | Best CV F1: {study_xgb.best_value:.4f}")
    else:
        print(f"\n    ✅ XGBoost tuning already complete | Best CV F1: {study_xgb.best_value:.4f}")
    best_params_xgb = study_xgb.best_params
    # B1: Override tree count if specified
    n_est_xgb = XGB_TREES if XGB_TREES > 0 else 2000
    # B2: Apply regularization multiplier
    if REG_MULTIPLIER != 1.0:
        if 'reg_lambda' in best_params_xgb:
            best_params_xgb['reg_lambda'] *= REG_MULTIPLIER
        if 'reg_alpha' in best_params_xgb:
            best_params_xgb['reg_alpha'] *= REG_MULTIPLIER
        if 'gamma' in best_params_xgb:
            best_params_xgb['gamma'] *= REG_MULTIPLIER
    # B3: Make XGB simpler if requested
    if XGB_SIMPLER:
        best_params_xgb['max_depth'] = min(5, best_params_xgb.get('max_depth', 6))
        n_est_xgb = min(1200, n_est_xgb)
    best_params_xgb.update({'n_estimators': n_est_xgb, 'random_state': SEED, 'tree_method': 'hist', 'verbosity': 0})
else:
    best_params_xgb = {
        'max_depth': 6, 'learning_rate': 0.0365, 'n_estimators': 1000,
        'subsample': 0.60, 'colsample_bytree': 0.99,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 2.24, 'reg_lambda': 0.011,
        'random_state': SEED, 'tree_method': 'hist', 'verbosity': 0
    }

print(f"  Tuning CatBoost ({N_TRIALS} trials - optimized for speed)...")
if OPTUNA_AVAILABLE:
    print(f"    ⚡ Parallel: {min(8, multiprocessing.cpu_count())} workers | CV folds: {N_SPLITS_TUNING} (tuning) / {N_SPLITS} (final) | Estimators: 2000 (tuning) / 4000 (final)")
    start_time = time.time()
    def objective_cat(trial):
        # Phase A: Tightened search space (CatBoost is 0.8 weight in w28 - most important)
        params = {
            'depth': trial.suggest_int('depth', 6, 9),  # Tightened (best ~7-8)
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),  # Tightened (best ~0.03-0.1)
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15, log=True),  # Tightened (best ~2-10)
            'random_strength': trial.suggest_float('random_strength', 0.5, 1.5),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.5),
            'rsm': trial.suggest_float('rsm', 0.7, 1.0),
            'border_count': trial.suggest_int('border_count', 128, 255),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                   scale_pos_weight * 0.7,  # Tightened (best ~16)
                                                   scale_pos_weight * 1.3,
                                                   log=True),
            'random_seed': SEED,
            'verbose': False,
            'iterations': 2000  # Reduced for faster tuning
        }
        # Phase A: Report intermediate values for pruning (early stopping per fold)
        gkf_tuning = GroupKFold(n_splits=N_SPLITS_TUNING)
        oof = np.zeros(len(X))
        fold_f1s = []
        
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf_tuning.split(X, y, groups)):
            params_safe = params.copy()
            params_safe['allow_writing_files'] = False
            model = cb.CatBoostClassifier(**params_safe)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
                verbose=False,
                early_stopping_rounds=50  # Early stopping for speed
            )
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            
            # Calculate F1 for this fold (for early stopping/pruning)
            fold_precision, fold_recall, fold_thresholds = precision_recall_curve(y.iloc[va_idx], oof[va_idx])
            fold_f1_scores = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall + 1e-10)
            fold_f1 = np.max(fold_f1_scores)
            fold_f1s.append(fold_f1)
            
            # Report intermediate value for pruning (after first fold)
            if fold_idx == 0 and USE_PRUNER:
                trial.report(fold_f1, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Final: OOF F1 after threshold selection on all folds
        precision, recall, thresholds = precision_recall_curve(y, oof)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores)
        
        # Report final value for pruning
        if USE_PRUNER:
            trial.report(best_f1, step=N_SPLITS_TUNING)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_f1
    
    # Use Optuna SQLite storage for automatic persistence
    storage_url = f"sqlite:///{CHECKPOINT_DIR}/cat_study.db"
    try:
        study_cat = optuna.load_study(study_name='cat_f1_fine', storage=storage_url)
        start_trial = len(study_cat.trials)
        if start_trial > 0:
            try:
                best_val = study_cat.best_value
                print(f"    📂 Resuming CatBoost tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming CatBoost tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new CatBoost tuning ({N_TRIALS} trials)")
    except:
        # Phase A: Add pruner for efficient early stopping (CatBoost is most important - 0.8 weight)
        pruner = None
        if USE_PRUNER and OPTUNA_AVAILABLE:
            if PRUNER_TYPE == 'hyperband':
                pruner = HyperbandPruner(
                    min_resource=1,  # Start pruning after 1 fold
                    max_resource=N_SPLITS_TUNING,  # Max resource = number of folds
                    reduction_factor=3  # Aggressiveness
                )
            else:  # median
                pruner = MedianPruner(
                    n_startup_trials=5,  # Don't prune first 5 trials
                    n_warmup_steps=1,  # Wait 1 step before pruning
                    interval_steps=1
                )
        
        study_cat = optuna.create_study(
            direction='maximize', 
            study_name='cat_f1_fine',
            storage=storage_url,
            load_if_exists=True,
            pruner=pruner
        )
        start_trial = len(study_cat.trials)
        if start_trial > 0:
            try:
                best_val = study_cat.best_value
                print(f"    📂 Resuming CatBoost tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming CatBoost tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new CatBoost tuning ({N_TRIALS} trials)")
    
    if start_trial >= N_TRIALS:
        print(f"    ✅ CatBoost tuning already complete ({start_trial}/{N_TRIALS} trials)")
        try:
            best_val = study_cat.best_value
            print(f"    Best CV F1: {best_val:.4f}")
        except:
            pass
    else:
        progress.update_stage(f"Tuning CatBoost ({N_TRIALS} trials)")
        study_cat.optimize(
            objective_cat, 
            n_trials=N_TRIALS - start_trial,
            n_jobs=min(8, multiprocessing.cpu_count()), 
            show_progress_bar=True
        )
    if start_trial < N_TRIALS:
        elapsed = time.time() - start_time
        print(f"\n    ✅ CatBoost tuning complete: {elapsed/60:.1f} min | Best CV F1: {study_cat.best_value:.4f}")
    best_params_cat = study_cat.best_params
    # B1: Override tree count if specified
    n_iter_cat = CAT_TREES if CAT_TREES > 0 else 4000
    # B2: Apply regularization multiplier
    if REG_MULTIPLIER != 1.0:
        if 'l2_leaf_reg' in best_params_cat:
            best_params_cat['l2_leaf_reg'] *= REG_MULTIPLIER
    best_params_cat.update({'iterations': n_iter_cat, 'random_seed': SEED, 'verbose': False, 'allow_writing_files': False})
else:
    best_params_cat = {
        'iterations': 1000, 'depth': 6, 'learning_rate': 0.05,
        'l2_leaf_reg': 3, 'scale_pos_weight': scale_pos_weight,
        'random_seed': SEED, 'verbose': False, 'allow_writing_files': False
    }

# Tune LightGBM if available
if LIGHTGBM_AVAILABLE and OPTUNA_AVAILABLE:
    print(f"  Tuning LightGBM ({N_TRIALS} trials - optimized for speed)...")
    print(f"    ⚡ Parallel: {min(8, multiprocessing.cpu_count())} workers | CV folds: {N_SPLITS_TUNING} (tuning) / {N_SPLITS} (final) | Estimators: 1000 (tuning) / 2000 (final)")
    start_time = time.time()
    def objective_lgb(trial):
        # Phase A: Tightened search space (LightGBM is middle-performing but adds diversity)
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 100, 300),  # Tightened (best ~257)
            'max_depth': trial.suggest_int('max_depth', 8, 12),  # Tightened (best ~12)
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),  # Tightened (best ~0.037)
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),  # Tightened (best ~44)
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),  # Tightened (best ~0.92)
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),  # Tightened (best ~0.61)
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 1e-5, log=True),  # Tightened (best ~1e-7)
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.1, log=True),  # Tightened (best ~0.049)
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                   scale_pos_weight * 0.8,  # Tightened (best ~16.2)
                                                   scale_pos_weight * 1.2,
                                                   log=True),
            'random_state': SEED,
            'verbosity': -1,
            'n_estimators': 1000  # Reduced for faster tuning
        }
        # Phase A: Report intermediate values for pruning (early stopping per fold)
        gkf_tuning = GroupKFold(n_splits=N_SPLITS_TUNING)
        oof = np.zeros(len(X))
        fold_f1s = []
        
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf_tuning.split(X, y, groups)):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            
            # Calculate F1 for this fold (for early stopping/pruning)
            fold_precision, fold_recall, fold_thresholds = precision_recall_curve(y.iloc[va_idx], oof[va_idx])
            fold_f1_scores = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall + 1e-10)
            fold_f1 = np.max(fold_f1_scores)
            fold_f1s.append(fold_f1)
            
            # Report intermediate value for pruning (after first fold)
            if fold_idx == 0 and USE_PRUNER:
                trial.report(fold_f1, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Final: OOF F1 after threshold selection on all folds
        precision, recall, thresholds = precision_recall_curve(y, oof)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores)
        
        # Report final value for pruning
        if USE_PRUNER:
            trial.report(best_f1, step=N_SPLITS_TUNING)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_f1
    
    # Use Optuna SQLite storage for automatic persistence
    storage_url = f"sqlite:///{CHECKPOINT_DIR}/lgb_study.db"
    try:
        study_lgb = optuna.load_study(study_name='lgb_f1_fine', storage=storage_url)
        start_trial = len(study_lgb.trials)
        if start_trial > 0:
            try:
                best_val = study_lgb.best_value
                print(f"    📂 Resuming LightGBM tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming LightGBM tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new LightGBM tuning ({N_TRIALS} trials)")
    except:
        # Phase A: Add pruner for efficient early stopping
        pruner = None
        if USE_PRUNER and OPTUNA_AVAILABLE:
            if PRUNER_TYPE == 'hyperband':
                pruner = HyperbandPruner(
                    min_resource=1,  # Start pruning after 1 fold
                    max_resource=N_SPLITS_TUNING,  # Max resource = number of folds
                    reduction_factor=3  # Aggressiveness
                )
            else:  # median
                pruner = MedianPruner(
                    n_startup_trials=5,  # Don't prune first 5 trials
                    n_warmup_steps=1,  # Wait 1 step before pruning
                    interval_steps=1
                )
        
        study_lgb = optuna.create_study(
            direction='maximize', 
            study_name='lgb_f1_fine',
            storage=storage_url,
            load_if_exists=True,
            pruner=pruner
        )
        start_trial = len(study_lgb.trials)
        if start_trial > 0:
            try:
                best_val = study_lgb.best_value
                print(f"    📂 Resuming LightGBM tuning from trial {start_trial}/{N_TRIALS} (Best: {best_val:.4f})")
            except:
                print(f"    📂 Resuming LightGBM tuning from trial {start_trial}/{N_TRIALS}")
        else:
            print(f"    🆕 Starting new LightGBM tuning ({N_TRIALS} trials)")
    
    if start_trial >= N_TRIALS:
        print(f"    ✅ LightGBM tuning already complete ({start_trial}/{N_TRIALS} trials)")
        try:
            best_val = study_lgb.best_value
            print(f"    Best CV F1: {best_val:.4f}")
        except:
            pass
    else:
        progress.update_stage(f"Tuning LightGBM ({N_TRIALS} trials)")
        study_lgb.optimize(
            objective_lgb, 
            n_trials=N_TRIALS - start_trial,
            n_jobs=min(8, multiprocessing.cpu_count()), 
            show_progress_bar=True
        )
    if start_trial < N_TRIALS:
        elapsed = time.time() - start_time
        print(f"\n    ✅ LightGBM tuning complete: {elapsed/60:.1f} min | Best CV F1: {study_lgb.best_value:.4f}")
    else:
        print(f"\n    ✅ LightGBM tuning already complete | Best CV F1: {study_lgb.best_value:.4f}")
    best_params_lgb = study_lgb.best_params
    best_params_lgb.update({'n_estimators': 2000, 'random_state': SEED, 'verbosity': -1})
elif LIGHTGBM_AVAILABLE:
    best_params_lgb = {
        'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
        'n_estimators': 2000, 'scale_pos_weight': scale_pos_weight,
        'random_state': SEED, 'verbosity': -1
    }
else:
    best_params_lgb = None

# Get initial predictions with GroupKFold
oof_initial_xgb = np.zeros(len(X))
test_proba_initial_xgb = np.zeros(len(X_test))
oof_initial_cat = np.zeros(len(X))
test_proba_initial_cat = np.zeros(len(X_test))
oof_initial_lgb = np.zeros(len(X))
test_proba_initial_lgb = np.zeros(len(X_test))

print(f"  Training {N_SPLITS} folds...")
fold_times = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    fold_start = time.time()
    print(f"  Fold {fold}/{N_SPLITS}: Training XGBoost, CatBoost, LightGBM...", end='\r')
    model_xgb = xgb.XGBClassifier(**best_params_xgb)
    model_xgb.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        verbose=False
    )
    oof_initial_xgb[va_idx] = model_xgb.predict_proba(X.iloc[va_idx])[:, 1]
    test_proba_initial_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Ensure allow_writing_files is set
    params_cat_safe = best_params_cat.copy()
    params_cat_safe['allow_writing_files'] = False
    model_cat = cb.CatBoostClassifier(**params_cat_safe)
    model_cat.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
        early_stopping_rounds=100,
        verbose=False
    )
    oof_initial_cat[va_idx] = model_cat.predict_proba(X.iloc[va_idx])[:, 1]
    test_proba_initial_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS
    
    if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
        model_lgb = lgb.LGBMClassifier(**best_params_lgb)
        model_lgb.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_initial_lgb[va_idx] = model_lgb.predict_proba(X.iloc[va_idx])[:, 1]
        test_proba_initial_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS
    
    fold_time = time.time() - fold_start
    fold_times.append(fold_time)
    avg_time = np.mean(fold_times)
    remaining = avg_time * (N_SPLITS - fold)
    print(f"  Fold {fold}/{N_SPLITS} complete: {fold_time:.1f}s | Avg: {avg_time:.1f}s | ETA: {remaining/60:.1f}m{' ' * 20}")

total_time = sum(fold_times)
print(f"\n  ✅ Initial models trained: {total_time/60:.1f} min total")
print()

# Evaluate individual models
print("[2.5] Individual Model OOF F1 Scores:")
print("=" * 70)

def evaluate_model_oof(oof_proba, y, name):
    """Optimized threshold search using precision_recall_curve"""
    # Use PR curve for faster, more accurate threshold finding
    precision, recall, thresholds = precision_recall_curve(y, oof_proba)
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"  {name:15s}: F1={best_f1:.4f} (thresh={best_t:.3f})")
    return best_f1, best_t

f1_xgb, t_xgb = evaluate_model_oof(oof_initial_xgb, y, "XGBoost")
f1_cat, t_cat = evaluate_model_oof(oof_initial_cat, y, "CatBoost")
if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
    f1_lgb, t_lgb = evaluate_model_oof(oof_initial_lgb, y, "LightGBM")
else:
    f1_lgb, t_lgb = 0, 0.5

# Calculate correlations
corr_xgb_cat = np.corrcoef(oof_initial_xgb, oof_initial_cat)[0, 1]
if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
    corr_xgb_lgb = np.corrcoef(oof_initial_xgb, oof_initial_lgb)[0, 1]
    corr_cat_lgb = np.corrcoef(oof_initial_cat, oof_initial_lgb)[0, 1]
else:
    corr_xgb_lgb = 0
    corr_cat_lgb = 0

print(f"\n  OOF Probability Correlations:")
print(f"    XGB-Cat: {corr_xgb_cat:.3f}")
if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
    print(f"    XGB-LGB: {corr_xgb_lgb:.3f}")
    print(f"    Cat-LGB: {corr_cat_lgb:.3f}")
print()

# Systematic ensemble weight optimization with diversity metrics
print("[2.6] Systematic Ensemble Weight Optimization (Diversity-Based)...")
print("="*70)
try:
    from ensemble_diversity_optimizer import (
        compute_diversity_report,
        generate_all_variants,
        summarize_corr
    )
    
    # Prepare OOF and test predictions dictionaries
    oof_dict_initial = {
        "xgb": oof_initial_xgb,
        "cat": oof_initial_cat
    }
    test_dict_initial = {
        "xgb": test_proba_initial_xgb,
        "cat": test_proba_initial_cat
    }
    
    if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
        oof_dict_initial["lgb"] = oof_initial_lgb
        test_dict_initial["lgb"] = test_proba_initial_lgb
        names = ("xgb", "cat", "lgb")
    else:
        names = ("xgb", "cat")
    
    # Generate all variants (initial models)
    variants_df_initial, corr_logit, corr_spearman, pair_df = generate_all_variants(
        y=y,
        oof_dict=oof_dict_initial,
        test_dict=test_dict_initial,
        variant_prefix="initial",
        out_dir=str(DATA_DIR),
        n_optuna_trials=50,  # Lower for initial models (will do more after pseudo-labeling)
        names=names  # Pass correct names based on LGB availability
    )
    
    # Print diversity analysis
    print("  Diversity Analysis:")
    print(f"    Mean correlation (logit): {variants_df_initial['mean_abs_corr_logit'].iloc[0]:.3f}")
    print(f"    Min correlation (logit): {variants_df_initial['min_abs_corr_logit'].iloc[0]:.3f}")
    print(f"    Mean double-fault: {variants_df_initial['mean_double_fault'].iloc[0]:.3f}")
    print()
    
    # Check if LightGBM adds value (diversity-based)
    if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
        # Find best with LGB (w_lgb > 0.05) vs without LGB (w_lgb <= 0.05)
        variants_with_lgb = variants_df_initial[variants_df_initial['w_lgb'] > 0.05]
        variants_without_lgb = variants_df_initial[variants_df_initial['w_lgb'] <= 0.05]
        
        if len(variants_with_lgb) > 0 and len(variants_without_lgb) > 0:
            best_with_lgb = variants_with_lgb['f1_oof'].max()
            best_without_lgb = variants_without_lgb['f1_oof'].max()
            print(f"  LightGBM Value Analysis:")
            print(f"    Best CV F1 with LGB (w_lgb > 0.05): {best_with_lgb:.4f}")
            print(f"    Best CV F1 without LGB (w_lgb <= 0.05): {best_without_lgb:.4f}")
            print(f"    LGB adds value: {best_with_lgb > best_without_lgb}")
            if best_with_lgb > best_without_lgb:
                print(f"    ✅ LightGBM helps ensemble (diversity benefit)")
            else:
                print(f"    ⚠️  LightGBM doesn't help (too correlated/redundant)")
            print()
        
        # Check correlations with LGB
        lgb_corr_with_cat = abs(corr_logit.loc['lgb', 'cat'])
        lgb_corr_with_xgb = abs(corr_logit.loc['lgb', 'xgb'])
        print(f"  LightGBM Correlations:")
        print(f"    LGB-Cat (abs): {lgb_corr_with_cat:.3f} {'✅ Diverse' if lgb_corr_with_cat < 0.85 else '⚠️  Correlated'}")
        print(f"    LGB-XGB (abs): {lgb_corr_with_xgb:.3f} {'✅ Diverse' if lgb_corr_with_xgb < 0.85 else '⚠️  Correlated'}")
        
        # Check double-fault with LGB
        lgb_cat_pair = pair_df[(pair_df['model_A'] == 'lgb') & (pair_df['model_B'] == 'cat')]
        if len(lgb_cat_pair) > 0:
            double_fault_lgb_cat = lgb_cat_pair['double_fault'].iloc[0]
            print(f"    LGB-Cat double-fault: {double_fault_lgb_cat:.3f} {'✅ Low' if double_fault_lgb_cat < 0.15 else '⚠️  High'}")
        print()
    
except ImportError as e:
    print(f"  ⚠️  Diversity optimizer not available: {e}")
    print("  Skipping systematic weight optimization (using manual weights only)")
    variants_df_initial = None
except Exception as e:
    print(f"  ⚠️  Error in diversity optimization: {e}")
    import traceback
    traceback.print_exc()
    print("  Continuing with manual weights only")
    variants_df_initial = None
print()

# Generate focused variants
print("[3] Generating focused variants...")
print(f"    Pseudo-thresholds: {len(PSEUDO_THRESHOLDS)} | Weight combinations: {len(WEIGHT_COMBINATIONS_3MODEL)}")
print(f"    Total variants: {len(PSEUDO_THRESHOLDS) * len(WEIGHT_COMBINATIONS_3MODEL)}")
cleanup_temp_files()  # Clean before pseudo-labeling training
print()

all_results = []
variant_start_time = time.time()

for idx, (tde_thresh, non_tde_thresh) in enumerate(PSEUDO_THRESHOLDS, 1):
    print(f"\n{'='*70}")
    print(f"VARIANT {idx}/{len(PSEUDO_THRESHOLDS)}: Pseudo-labeling TDE>{tde_thresh:.2f}, non-TDE<{non_tde_thresh:.2f}")
    print(f"{'='*70}")
    variant_time = time.time()
    
    # C3: Pseudo-label source selection (ensemble vs CatBoost-only)
    if PSEUDO_SOURCE == 'catboost':
        # C3: Use CatBoost-only for pseudo-labels (simpler, since CatBoost dominates ensemble)
        ensemble_test_proba = test_proba_initial_cat
        print(f"    Pseudo-labeling source: CatBoost-only (F1={f1_cat:.4f})")
    else:
        # Default: Use ensemble consensus (more robust than single model)
        # Weight by actual model performance (data-driven, not arbitrary)
        total_f1 = f1_xgb + f1_cat
        if total_f1 > 0:
            w_xgb_pseudo = f1_xgb / total_f1
            w_cat_pseudo = f1_cat / total_f1
        else:
            w_xgb_pseudo, w_cat_pseudo = 0.5, 0.5  # Fallback to equal if no F1 scores
        ensemble_test_proba = w_xgb_pseudo * test_proba_initial_xgb + w_cat_pseudo * test_proba_initial_cat
        print(f"    Pseudo-labeling ensemble: {w_xgb_pseudo:.1%} XGB (F1={f1_xgb:.4f}) + {w_cat_pseudo:.1%} Cat (F1={f1_cat:.4f})")
    
    high_conf_tde = ensemble_test_proba > tde_thresh
    high_conf_non_tde = ensemble_test_proba < non_tde_thresh
    
    if high_conf_tde.sum() == 0:
        print(f"    ⚠️  No high-confidence TDEs, skipping...")
        continue
    
    # C1: Iterative pseudo-labeling support
    # Start with initial predictions
    current_test_proba_xgb = test_proba_initial_xgb.copy()
    current_test_proba_cat = test_proba_initial_cat.copy()
    
    # Define orig_n before using it (needed for iterative pseudo-labeling)
    orig_n = len(X)
    
    # Initialize final predictions (will be updated each iteration)
    oof_pseudo_xgb = np.zeros(orig_n)
    test_proba_pseudo_xgb = np.zeros(len(X_test))
    oof_pseudo_cat = np.zeros(orig_n)
    test_proba_pseudo_cat = np.zeros(len(X_test))
    oof_pseudo_lgb = np.zeros(orig_n)
    test_proba_pseudo_lgb = np.zeros(len(X_test))
    
    # C1: Iterate pseudo-labeling PSEUDO_ITERATIONS times
    for pseudo_iter in range(1, PSEUDO_ITERATIONS + 1):
        if PSEUDO_ITERATIONS > 1:
            print(f"    [Iteration {pseudo_iter}/{PSEUDO_ITERATIONS}]")
        
        # Create pseudo-labels from current predictions
        if pseudo_iter == 1:
            # First iteration: use initial predictions
            ensemble_test_proba_iter = ensemble_test_proba.copy()
        else:
            # Later iterations: use updated predictions from previous iteration
            if PSEUDO_SOURCE == 'catboost':
                ensemble_test_proba_iter = current_test_proba_cat.copy()
            else:
                total_f1_iter = f1_xgb + f1_cat
                if total_f1_iter > 0:
                    w_xgb_iter = f1_xgb / total_f1_iter
                    w_cat_iter = f1_cat / total_f1_iter
                else:
                    w_xgb_iter, w_cat_iter = 0.5, 0.5
                ensemble_test_proba_iter = w_xgb_iter * current_test_proba_xgb + w_cat_iter * current_test_proba_cat
        
        high_conf_tde_iter = ensemble_test_proba_iter > tde_thresh
        high_conf_non_tde_iter = ensemble_test_proba_iter < non_tde_thresh
        
        if high_conf_tde_iter.sum() == 0:
            print(f"    ⚠️  No high-confidence TDEs in iteration {pseudo_iter}, stopping iterations")
            break
        
        pseudo_test_tde = test[high_conf_tde_iter].copy()
        pseudo_test_tde['target'] = 1
        pseudo_test_tde['_pseudo_prob'] = ensemble_test_proba_iter[high_conf_tde_iter]
        
        pseudo_test_non_tde = test[high_conf_non_tde_iter].copy()
        pseudo_test_non_tde['target'] = 0
        pseudo_test_non_tde['_pseudo_prob'] = ensemble_test_proba_iter[high_conf_non_tde_iter]
        
        pseudo_train = pd.concat([train, pseudo_test_tde, pseudo_test_non_tde], ignore_index=True)
        X_pseudo = pseudo_train[feature_cols]
        y_pseudo = pseudo_train['target'].astype(int)
        
        orig_n = len(X)
        orig_idx = np.arange(orig_n)
        pseudo_idx = np.arange(orig_n, len(X_pseudo))
        
        # Retrain with GroupKFold
        oof_iter_xgb = np.zeros(orig_n)
        test_proba_iter_xgb = np.zeros(len(X_test))
        oof_iter_cat = np.zeros(orig_n)
        test_proba_iter_cat = np.zeros(len(X_test))
        oof_iter_lgb = np.zeros(orig_n)
        test_proba_iter_lgb = np.zeros(len(X_test))
        
        print(f"    Retraining with pseudo-labels ({len(pseudo_idx)} added)...")
        pseudo_fold_times = []
        for fold, (tr_o, va_o) in enumerate(gkf.split(X, y, groups), 1):
            pseudo_fold_start = time.time()
            print(f"    Fold {fold}/{N_SPLITS}...", end='\r')
            tr_idx = np.concatenate([tr_o, pseudo_idx])
            va_idx = va_o

            # ENHANCED: Weight pseudo-labels lower than real labels (prevents overfitting)
            # Real labels: weight=1.0
            # Pseudo-labels: weight based on confidence (higher prob → higher weight, but capped)
            # Research: 0.5-0.7x is typical, we use confidence-based: 0.4-0.7x range
            sample_weights = np.ones(len(tr_idx))
            pseudo_mask_in_tr = tr_idx >= orig_n
            if pseudo_mask_in_tr.sum() > 0:
                # Get stored probabilities from pseudo_train
                pseudo_probs = pseudo_train.iloc[tr_idx[pseudo_mask_in_tr]]['_pseudo_prob'].values
                if len(pseudo_probs) > 0 and not np.isnan(pseudo_probs).all():
                    # Weight by confidence or fixed based on PSEUDO_WEIGHT_METHOD
                    if PSEUDO_WEIGHT_METHOD == 'fixed':
                        # Fixed 0.6x weight for all pseudo-labels
                        sample_weights[pseudo_mask_in_tr] = 0.6
                    else:
                        # Confidence-based: prob near 0 or 1 → higher weight (0.7x), prob near 0.5 → lower weight (0.4x)
                        # For TDEs (prob > tde_thresh): weight = 0.4 + 0.3 * (prob - tde_thresh) / (1.0 - tde_thresh)
                        # For non-TDEs (prob < non_tde_thresh): weight = 0.4 + 0.3 * (non_tde_thresh - prob) / non_tde_thresh
                        pseudo_weights = np.zeros(len(pseudo_probs))
                        tde_mask = pseudo_probs > tde_thresh
                        non_tde_mask = pseudo_probs < non_tde_thresh
                        if tde_mask.sum() > 0:
                            pseudo_weights[tde_mask] = 0.4 + 0.3 * (pseudo_probs[tde_mask] - tde_thresh) / (1.0 - tde_thresh + 1e-10)
                        if non_tde_mask.sum() > 0:
                            pseudo_weights[non_tde_mask] = 0.4 + 0.3 * (non_tde_thresh - pseudo_probs[non_tde_mask]) / (non_tde_thresh + 1e-10)
                        pseudo_weights = np.clip(pseudo_weights, 0.4, 0.7)  # Cap at 0.4-0.7x
                        sample_weights[pseudo_mask_in_tr] = pseudo_weights
                else:
                    # Fallback: use fixed 0.6x if we can't get probabilities
                    sample_weights[pseudo_mask_in_tr] = 0.6

            # XGBoost
            model_xgb = xgb.XGBClassifier(**best_params_xgb)
            model_xgb.fit(
                X_pseudo.iloc[tr_idx], y_pseudo.iloc[tr_idx],
                sample_weight=sample_weights,  # Weight pseudo-labels lower
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                verbose=False
            )
            oof_iter_xgb[va_idx] = model_xgb.predict_proba(X.iloc[va_idx])[:, 1]
            test_proba_iter_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

            # CatBoost
            params_cat_safe = best_params_cat.copy()
            params_cat_safe['allow_writing_files'] = False
            model_cat = cb.CatBoostClassifier(**params_cat_safe)
            model_cat.fit(
                X_pseudo.iloc[tr_idx], y_pseudo.iloc[tr_idx],
                sample_weight=sample_weights,  # Weight pseudo-labels lower
                eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
                early_stopping_rounds=100,
                verbose=False
            )
            oof_iter_cat[va_idx] = model_cat.predict_proba(X.iloc[va_idx])[:, 1]
            test_proba_iter_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS

            # LightGBM
            if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
                model_lgb = lgb.LGBMClassifier(**best_params_lgb)
                model_lgb.fit(
                    X_pseudo.iloc[tr_idx], y_pseudo.iloc[tr_idx],
                    sample_weight=sample_weights,  # Weight pseudo-labels lower
                    eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                oof_iter_lgb[va_idx] = model_lgb.predict_proba(X.iloc[va_idx])[:, 1]
                test_proba_iter_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

            pseudo_fold_time = time.time() - pseudo_fold_start
            pseudo_fold_times.append(pseudo_fold_time)
            avg_time = np.mean(pseudo_fold_times)
            remaining = avg_time * (N_SPLITS - fold) if fold < N_SPLITS else 0
            print(f"    Fold {fold}/{N_SPLITS} complete: {pseudo_fold_time:.1f}s | ETA: {remaining/60:.1f}m{' ' * 20}")
        
        # Update current predictions for next iteration
        current_test_proba_xgb = test_proba_iter_xgb.copy()
        current_test_proba_cat = test_proba_iter_cat.copy()
        
        # Accumulate to final predictions (average across iterations)
        oof_pseudo_xgb += oof_iter_xgb / PSEUDO_ITERATIONS
        test_proba_pseudo_xgb += test_proba_iter_xgb / PSEUDO_ITERATIONS
        oof_pseudo_cat += oof_iter_cat / PSEUDO_ITERATIONS
        test_proba_pseudo_cat += test_proba_iter_cat / PSEUDO_ITERATIONS
        if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
            oof_pseudo_lgb += oof_iter_lgb / PSEUDO_ITERATIONS
            test_proba_pseudo_lgb += test_proba_iter_lgb / PSEUDO_ITERATIONS
    
    if PSEUDO_ITERATIONS > 1:
        print(f"    ✅ Iterative pseudo-labeling complete ({PSEUDO_ITERATIONS} iterations): {sum(pseudo_fold_times)/60:.1f} min")
    else:
        print(f"    ✅ Pseudo-labeling training complete: {sum(pseudo_fold_times)/60:.1f} min")
    
    # ============================================================================
    # ENSEMBLE: STACKING (meta-learner) + Manual weights (fallback/comparison)
    # ============================================================================
    if USE_STACKING:
        print("    Creating ensembles: STACKING (primary) + Manual weights (comparison)...")
    else:
        print("    Creating ensembles: Manual weights only (stacking disabled)...")
    weight_results = []  # Collect all ensemble results
    
    # STACKING: Train meta-learner on OOF predictions (learns optimal combination)
    if USE_STACKING:
        print("      [1] STACKING: Training meta-learner on OOF predictions...")
        try:
            # Prepare OOF features (each model's predictions)
            if LIGHTGBM_AVAILABLE and best_params_lgb is not None:
                oof_stack_features = np.column_stack([oof_pseudo_xgb, oof_pseudo_cat, oof_pseudo_lgb])
                test_stack_features = np.column_stack([test_proba_pseudo_xgb, test_proba_pseudo_cat, test_proba_pseudo_lgb])
                n_models = 3
            else:
                oof_stack_features = np.column_stack([oof_pseudo_xgb, oof_pseudo_cat])
                test_stack_features = np.column_stack([test_proba_pseudo_xgb, test_proba_pseudo_cat])
                n_models = 2
            # Train meta-learner with GroupKFold (same splits as base models)
            # Use original train groups (not pseudo-labeled), but only train indices from orig_n
            oof_stacked = np.zeros(len(y))
            test_stacked = np.zeros(len(X_test))
            meta_coefs = []
            
            # ENHANCED: Use XGBoost as meta-learner (more powerful than LogisticRegression)
            # Also add feature interactions (products, ratios) for better learning
            def create_meta_features(base_features):
                """Create enhanced meta-features with interactions"""
                if USE_ENHANCED_STACKING:
                    n = base_features.shape[0]
                    enhanced = [base_features]
                    
                    # Add pairwise products (captures interactions)
                    for i in range(n_models):
                        for j in range(i+1, n_models):
                            enhanced.append((base_features[:, i] * base_features[:, j]).reshape(-1, 1))
                    
                    # Add ratios (captures relative strengths)
                    for i in range(n_models):
                        for j in range(n_models):
                            if i != j:
                                ratio = base_features[:, i] / (base_features[:, j] + 1e-10)
                                enhanced.append(ratio.reshape(-1, 1))
                    
                    # Add statistics (mean, std, max, min across models)
                    enhanced.append(base_features.mean(axis=1).reshape(-1, 1))
                    enhanced.append(base_features.std(axis=1).reshape(-1, 1))
                    enhanced.append(base_features.max(axis=1).reshape(-1, 1))
                    enhanced.append(base_features.min(axis=1).reshape(-1, 1))
                    
                    return np.hstack(enhanced)
                else:
                    # Simple: just use base features (no interactions)
                    return base_features
            
            if USE_ENHANCED_STACKING:
                oof_stack_enhanced = create_meta_features(oof_stack_features)
                test_stack_enhanced = create_meta_features(test_stack_features)
            else:
                oof_stack_enhanced = oof_stack_features
                test_stack_enhanced = test_stack_features
            
            for fold, (tr_o, va_o) in enumerate(gkf.split(X, y, groups)):
                # Train on original train indices only (not pseudo-labeled data)
                if USE_ENHANCED_STACKING:
                    # Use XGBoost meta-learner (more powerful than LR)
                    meta_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        random_state=SEED+fold,
                        verbosity=0,
                        tree_method='hist'
                    )
                else:
                    # Use LogisticRegression meta-learner (simpler, less overfitting risk)
                    meta_model = LogisticRegression(
                        max_iter=1000,
                        random_state=SEED+fold,
                        class_weight='balanced'
                    )
                meta_model.fit(oof_stack_enhanced[tr_o], y.iloc[tr_o])
                oof_stacked[va_o] = meta_model.predict_proba(oof_stack_enhanced[va_o])[:, 1]
                test_stacked += meta_model.predict_proba(test_stack_enhanced)[:, 1] / N_SPLITS
                
                # Extract feature importance as "coefficients" (for interpretability)
                if hasattr(meta_model, 'feature_importances_'):
                    meta_coefs.append(meta_model.feature_importances_[:n_models])  # Only base model importances
            
            # Average feature importances (learned weights from XGBoost meta-learner)
            if len(meta_coefs) > 0:
                avg_coefs = np.mean(meta_coefs, axis=0)
                if n_models == 3:
                    w_xgb_stack, w_cat_stack, w_lgb_stack = avg_coefs / (avg_coefs.sum() + 1e-10)
                else:
                    w_xgb_stack, w_cat_stack = avg_coefs / (avg_coefs.sum() + 1e-10)
                    w_lgb_stack = 0.0
            else:
                # Fallback to equal weights if no importances available
                if n_models == 3:
                    w_xgb_stack, w_cat_stack, w_lgb_stack = 1/3, 1/3, 1/3
                else:
                    w_xgb_stack, w_cat_stack = 0.5, 0.5
                    w_lgb_stack = 0.0
            
            # Find best threshold for stacked ensemble
            precisions, recalls, thresholds = precision_recall_curve(y, oof_stacked)
            f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores_pr)
            best_t_stack = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1_stack = f1_scores_pr[best_idx]
            
            # Fine-grained threshold search
            search_range = max(0.05, min(0.95, best_t_stack + 0.05))
            search_min = max(0.05, best_t_stack - 0.05)
            for t in np.linspace(search_min, search_range, 101):
                f1 = f1_score(y, (oof_stacked >= t).astype(int))
                if f1 > best_f1_stack:
                    best_f1_stack = f1
                    best_t_stack = t
            
            weight_results.append({
                'method': 'STACKING',
                'weights': (w_xgb_stack, w_cat_stack, w_lgb_stack),
                'cv_f1': best_f1_stack,
                'threshold': best_t_stack,
                'oof': oof_stacked.copy(),
                'test_proba': test_stacked.copy()
            })
            print(f"        ✅ Stacking: weights=({w_xgb_stack:.3f}, {w_cat_stack:.3f}, {w_lgb_stack:.3f}), CV F1={best_f1_stack:.4f}")
        except Exception as e:
            print(f"        ⚠️  Stacking failed: {e}, using manual weights only")
    else:
        print("      [1] STACKING: Disabled (USE_STACKING=false)")
    
    # MANUAL WEIGHTS: Test known good combinations (w28 family + baselines)
    print("      [2] MANUAL WEIGHTS: Testing known good combinations...")
    for w_xgb, w_cat, w_lgb in WEIGHT_COMBINATIONS_3MODEL:
        if not LIGHTGBM_AVAILABLE or best_params_lgb is None:
            # Fallback to 2-model if LightGBM not available
            if w_lgb > 0:
                continue
            oof_weighted = w_xgb * oof_pseudo_xgb + w_cat * oof_pseudo_cat
            test_proba_weighted = w_xgb * test_proba_pseudo_xgb + w_cat * test_proba_pseudo_cat
        else:
            oof_weighted = w_xgb * oof_pseudo_xgb + w_cat * oof_pseudo_cat + w_lgb * oof_pseudo_lgb
            test_proba_weighted = w_xgb * test_proba_pseudo_xgb + w_cat * test_proba_pseudo_cat + w_lgb * test_proba_pseudo_lgb
        
        # Find best threshold using PR curve (faster and more accurate)
        precisions, recalls, thresholds = precision_recall_curve(y, oof_weighted)
        f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores_pr)
        best_t_pr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1_this = f1_scores_pr[best_idx]
        
        # Fine-grained search around PR optimum
        search_range = max(0.05, min(0.95, best_t_pr + 0.05))
        search_min = max(0.05, best_t_pr - 0.05)
        fine_thresholds = np.linspace(search_min, search_range, 101)
        best_t_this = best_t_pr
        for t in fine_thresholds:
            f1 = f1_score(y, (oof_weighted >= t).astype(int))
            if f1 > best_f1_this:
                best_f1_this = f1
                best_t_this = t
        
        weight_results.append({
            'method': 'MANUAL',
            'weights': (w_xgb, w_cat, w_lgb),
            'cv_f1': best_f1_this,
            'threshold': best_t_this,
            'oof': oof_weighted.copy(),
            'test_proba': test_proba_weighted.copy()
        })
        
        print(f"        Weights ({w_xgb:.2f}, {w_cat:.2f}, {w_lgb:.2f}): CV F1 = {best_f1_this:.4f}")
    
    # Sort by CV F1 and take TOP 3 (including stacking if available)
    weight_results.sort(key=lambda x: x['cv_f1'], reverse=True)
    top_n = min(3, len(weight_results))
    
    print(f"    ✅ Top {top_n} ensembles (saving all, CV F1 not reliable for LB):")
    for i, wr in enumerate(weight_results[:top_n], 1):
        w_xgb, w_cat, w_lgb = wr['weights']
        method = wr.get('method', 'MANUAL')
        print(f"      #{i} [{method}]: ({w_xgb:.3f}, {w_cat:.3f}, {w_lgb:.3f}) - CV F1={wr['cv_f1']:.4f}")
    
    # Save TOP N ensembles (stacking + manual weights)
    for i, wr in enumerate(weight_results[:top_n]):
        w_xgb, w_cat, w_lgb = wr['weights']
        method = wr.get('method', 'MANUAL')
        oof_ensemble = wr['oof']
        test_proba_ensemble = wr['test_proba']
        best_t = wr['threshold']
        best_f1 = wr['cv_f1']
        
        # Create submission
        test_pred = (test_proba_ensemble >= best_t).astype(int)
        n_tdes = test_pred.sum()
        
        # Filename with optional test identifier
        test_prefix = f"test_{TEST_ID}_" if TEST_ID else ""
        if method == 'STACKING':
            filename = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_stacking.csv"
        elif i == 0:
            # First manual weight uses standard name
            filename = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_w{int(w_xgb*10):d}{int(w_cat*10):d}{int(w_lgb*10):d}.csv"
        else:
            # Others get rank suffix
            filename = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_w{int(w_xgb*10):d}{int(w_cat*10):d}{int(w_lgb*10):d}_rank{i+1}.csv"
        
        sub = pd.DataFrame({
            'object_id': test['object_id'],
            'prediction': test_pred
        })
        sub.to_csv(DATA_DIR / filename, index=False)
        
        # 346tdes version (only for CV-best to avoid too many files)
        if i == 0:
            for t in np.linspace(0.1, 0.9, 161):
                n = (test_proba_ensemble >= t).sum()
                if 340 <= n <= 355:
                    test_pred_target = (test_proba_ensemble >= t).astype(int)
                    sub_target = pd.DataFrame({
                        'object_id': test['object_id'],
                        'prediction': test_pred_target
                    })
                    filename_346 = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_w{int(w_xgb*10):d}{int(w_cat*10):d}{int(w_lgb*10):d}_346tdes.csv"
                    sub_target.to_csv(DATA_DIR / filename_346, index=False)
                    break
        
        # Record results
        all_results.append({
            'pseudo_tde': tde_thresh,
            'pseudo_non_tde': non_tde_thresh,
            'method': method,
            'weight_xgb': w_xgb,
            'weight_cat': w_cat,
            'weight_lgb': w_lgb,
            'cv_f1': best_f1,
            'threshold': best_t,
            'n_tdes': n_tdes,
            'filename': filename,
            'rank': i + 1
        })
    
    # Use CV-best for main result tracking
    best_result = weight_results[0]
    best_weights = best_result['weights']
    best_f1 = best_result['cv_f1']
    best_t = best_result['threshold']
    best_method = best_result.get('method', 'MANUAL')
    n_tdes = (best_result['test_proba'] >= best_t).sum()
    test_prefix = f"test_{TEST_ID}_" if TEST_ID else ""
    if best_method == 'STACKING':
        filename = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_stacking.csv"
    else:
        filename = f"{test_prefix}focused_p{int(tde_thresh*100):02d}_{int(non_tde_thresh*100):02d}_w{int(best_weights[0]*10):d}{int(best_weights[1]*10):d}{int(best_weights[2]*10):d}.csv"
    
    elapsed_variant = time.time() - variant_time
    print(f"    ✅ Variant complete: {elapsed_variant/60:.1f} min | Saved {top_n} weight combinations | CV F1={best_f1:.4f} | TDEs={n_tdes}")

total_time = time.time() - variant_start_time
print()
print("="*70)
print("FOCUSED VARIANTS COMPLETE")
print("="*70)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
print()
print("📊 Results (sorted by CV F1):")
print()
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('cv_f1', ascending=False)
for _, row in results_df.iterrows():
    print(f"  {row['filename']:50s} CV F1={row['cv_f1']:.4f}, TDEs={row['n_tdes']:3d}, "
          f"Pseudo={row['pseudo_tde']:.2f}/{row['pseudo_non_tde']:.2f}, "
          f"Weights={row['weight_xgb']:.1f}/{row['weight_cat']:.1f}/{row.get('weight_lgb', 0):.1f}")

print()
print(f"✅ Generated {len(all_results)} focused variants")
print("="*70)
