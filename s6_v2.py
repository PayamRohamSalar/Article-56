
# -*- coding: utf-8 -*-

"""
Chapter 6 — Advanced Analytics (s6.py)
Matches the style of s3/s4/s5:
- Persian RTL text rendering (arabic_reshaper + bidi)
- Persian digits on all ticks/labels
- Same font setup (Vazirmatn path) and I/O conventions
Outputs:
  fig/chart_6_1_dendrogram.png
  fig/chart_6_2_kmeans_scatter.png
  fig/chart_6_3_corr_heatmap.png
  fig/chart_6_4_regression_forecast.png
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Persian RTL
import arabic_reshaper
from bidi.algorithm import get_display

# ML & Stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm

# Font handling
from matplotlib import font_manager

# ==============================================================================
# Font Configuration (aligned with s3/s4/s5)
# ==============================================================================
font_path = Path(r'D:\OneDrive\AI-Project\Article56\fonts\ttf\Vazirmatn-Regular.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Vazirmatn'
else:
    plt.rcParams['font.family'] = 'Vazirmatn'
    try:
        fam = plt.rcParams['font.family']
        if isinstance(fam, str):
            fam_ok = fam.lower() == 'vazirmatn'
        else:
            fam_ok = ('Vazirmatn' in fam)
        if not fam_ok:
            print(f"Warning: Font not found at {font_path}")
            print("Falling back to default font...")
            plt.rcParams['font.family'] = 'Tahoma'
    except Exception:
        plt.rcParams['font.family'] = 'Tahoma'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

# ==============================================================================
# Helpers (same pattern as s5)
# ==============================================================================
def fix_persian_text(text: str) -> str:
    if text is None or str(text).strip() == '':
        return ''
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return str(text)

def convert_to_persian_number(x) -> str:
    english_digits = '0123456789'
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    trans = str.maketrans(english_digits, persian_digits)
    return str(x).translate(trans)

def format_number_with_separator(number, use_persian: bool=True) -> str:
    if isinstance(number, (int, float, np.integer, np.floating)):
        formatted = f'{number:,.0f}'
    else:
        formatted = str(number)
    return convert_to_persian_number(formatted) if use_persian else formatted

def find_first_existing_column(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"ستون(های) مورد انتظار یافت نشد: {candidates}")

def normalize_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    replacements = {'ي':'ی','ك':'ک','ۀ':'ه','ة':'ه','‌':'','\u200c':''}
    for a,b in replacements.items():
        s = s.replace(a,b)
    return s

# ==============================================================================
# Paths & IO
# ==============================================================================
base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Sample_Data.xlsx'
if not data_file.exists():
    alt = base_dir / 'Q_Sample_Data.xlsx'
    if alt.exists():
        data_file = alt
    else:
        raise FileNotFoundError(f"Expected data file not found at {data_file} or {alt}")

fig_dir = base_dir / 'fig/S6'
fig_dir.mkdir(exist_ok=True)

print(f"✓ Output directory: {fig_dir}")

# ==============================================================================
# Load & Prepare Data
# ==============================================================================
df = pd.read_excel(data_file)

# Detect columns
year_col    = find_first_existing_column(df, ['سال'])
device_col  = find_first_existing_column(df, ['نام سازمان'])
credit_col  = find_first_existing_column(df, ['اعتبار'])
rtype_col   = None
for c in ['نوع پژوهش','نوع طرح','ماهیت طرح']:
    if c in df.columns:
        rtype_col = c
        break

# Coerce types
df[year_col]   = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
df[credit_col] = pd.to_numeric(df[credit_col], errors='coerce')

# Filter year range 1398..1403 and drop invalid credit
df = df[df[year_col].between(1398, 1403, inclusive='both')]
df = df[df[credit_col].notna()]

# Normalize device names
df[device_col] = df[device_col].map(normalize_name)

# ------------------------------------------------------------------------------
# Build device-level features
# ------------------------------------------------------------------------------
MIN_N = 3  # min projects per device to be included

grp = df.groupby(device_col, dropna=True)
df_dev = pd.DataFrame({
    'device'     : grp.size().index,
    'n_projects' : grp.size().values,
    'sum_credit' : grp[credit_col].sum().values,
    'mean_credit': grp[credit_col].mean().values
})

# Research type ratio (Fundamental/Applied)
def _is_fundamental(x):
    if pd.isna(x): return False
    s = normalize_name(x)
    return ('بنیادی' in s) or ('پایه' in s)
def _is_applied(x):
    if pd.isna(x): return False
    s = normalize_name(x)
    return ('کاربردی' in s)

if rtype_col is not None:
    tgrp = df.groupby(device_col)[rtype_col].apply(lambda s: list(s.dropna()))
    fund_counts = tgrp.apply(lambda lst: sum(_is_fundamental(x) for x in lst) if isinstance(lst, list) else 0)
    appl_counts = tgrp.apply(lambda lst: sum(_is_applied(x) for x in lst) if isinstance(lst, list) else 0)
    ratio = (fund_counts / (appl_counts.replace(0, np.nan))).fillna(np.nan)
    df_dev = df_dev.merge(ratio.rename('ratio_fund_applied').reset_index()
                          .rename(columns={device_col:'device'}), on='device', how='left')
else:
    print("⚠ ستون نوع پژوهش یافت نشد؛ نسبت بنیادی/کاربردی محاسبه نخواهد شد.")
    df_dev['ratio_fund_applied'] = np.nan

# Filter by MIN_N projects
df_dev = df_dev[df_dev['n_projects'] >= MIN_N].reset_index(drop=True)

# Handle NaNs by column medians (for clustering)
for col in ['n_projects','sum_credit','mean_credit','ratio_fund_applied']:
    if col in df_dev.columns:
        if df_dev[col].isna().all():
            pass
        else:
            df_dev[col] = df_dev[col].fillna(df_dev[col].median())

# Standardize features for clustering
cluster_features = ['n_projects','sum_credit','mean_credit']
if not df_dev['ratio_fund_applied'].isna().all():
    cluster_features.append('ratio_fund_applied')

if len(df_dev) < 2 or len(cluster_features) < 2:
    print("⚠ داده کافی برای خوشه‌بندی وجود ندارد (کمتر از 2 دستگاه یا کمتر از 2 ویژگی).")
    can_cluster = False
else:
    can_cluster = True
    X = df_dev[cluster_features].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

# ==============================================================================
# Chart 6-1: Hierarchical Clustering Dendrogram
# ==============================================================================
try:
    if not can_cluster:
        raise RuntimeError("insufficient data for clustering")
    D = pdist(X_std, metric='euclidean')
    Z = linkage(D, method='ward')

    fig, ax = plt.subplots(figsize=(14, 8))
    labels = [fix_persian_text(d) for d in df_dev['device'].tolist()]
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=9, ax=ax, color_threshold=None)
    ax.set_title(fix_persian_text('خوشه‌بندی سلسله‌مراتبی دستگاه‌های اجرایی'), fontsize=15, fontweight='bold', pad=12)
    ax.set_ylabel(fix_persian_text('فاصله خوشه‌ای (Ward)'), fontsize=12)
    ax.set_yticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_yticklabels()])
    plt.tight_layout()
    outpath = fig_dir / 'chart_6_1_dendrogram.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 6-1 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 6-1 (dendrogram) failed: {e}")

# ==============================================================================
# Chart 6-2: KMeans Scatter (choose k=3 or 4 by silhouette)
# ==============================================================================
try:
    if not can_cluster:
        raise RuntimeError("insufficient data for clustering")
    best_k = 3
    best_score = -1
    best_labels = None
    for k in [3,4]:
        if len(df_dev) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km.fit_predict(X_std)
        if len(set(labels_k)) < 2:
            continue
        score = silhouette_score(X_std, labels_k)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels_k
    if best_labels is None:
        km = KMeans(n_clusters=min(3, max(1, len(df_dev)-1)), n_init=10, random_state=42)
        best_labels = km.fit_predict(X_std)
        best_k = km.n_clusters

    df_dev['cluster'] = best_labels

    fig, ax = plt.subplots(figsize=(12, 8))
    for cl in sorted(df_dev['cluster'].unique()):
        sub = df_dev[df_dev['cluster'] == cl]
        ax.scatter(sub['n_projects'], sub['mean_credit'], s=70, label=convert_to_persian_number(f'خوشه {cl+1}'))
        for _, r in sub.iterrows():
            ax.text(r['n_projects'], r['mean_credit'],
                    fix_persian_text(r['device']), fontsize=8, alpha=0.9, ha='left', va='center')

    ax.set_xlabel(fix_persian_text('تعداد طرح‌ها'), fontsize=12, fontweight='bold')
    ax.set_ylabel(fix_persian_text('میانگین اعتبار (میلیون ریال)'), fontsize=12, fontweight='bold')
    ax.set_title(fix_persian_text('خوشه‌بندی دستگاه‌ها (K-Means)'), fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_xticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_yticklabels()])  # wrong axis intentionally?
    # fix: correct x/y ticks Persianization
    ax.set_xticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_xticklabels()])
    ax.set_yticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_yticklabels()])
    leg = ax.legend(title=None, frameon=True)
    for txt in leg.get_texts():
        txt.set_fontsize(10)
    plt.tight_layout()
    outpath = fig_dir / 'chart_6_2_kmeans_scatter.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 6-2 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 6-2 (kmeans scatter) failed: {e}")

# ==============================================================================
# Chart 6-3: Correlation Heatmap
# ==============================================================================
try:
    corr_cols = ['n_projects','sum_credit','mean_credit','ratio_fund_applied']
    corr_cols = [c for c in corr_cols if c in df_dev.columns and not df_dev[c].isna().all()]
    if len(corr_cols) >= 2:
        corr = df_dev[corr_cols].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlOrBr', linewidths=0.5, linecolor='white', ax=ax)
        ax.set_xticklabels([fix_persian_text(col) for col in corr.columns], rotation=0, fontsize=10)
        ax.set_yticklabels([fix_persian_text(col) for col in corr.index], rotation=0, fontsize=10)
        for t in ax.texts:
            t.set_text(convert_to_persian_number(t.get_text()))
        ax.set_title(fix_persian_text('ماتریس همبستگی متغیرهای کلیدی'), fontsize=15, fontweight='bold', pad=12)
        plt.tight_layout()
        outpath = fig_dir / 'chart_6_3_corr_heatmap.png'
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Chart 6-3 saved: {outpath}")
    else:
        print("⚠ برای محاسبه همبستگی حداقل به ۲ متغیر معتبر نیاز است.")
except Exception as e:
    print(f"⚠ Chart 6-3 (correlation heatmap) failed: {e}")

# ==============================================================================
# Chart 6-4: Linear Regression + Forecast 1404
# ==============================================================================
try:
    df_year = (df.groupby(year_col)[credit_col].sum()
                 .reindex(range(1398, 1404), fill_value=np.nan)
                 .reset_index())
    df_year.columns = ['year', 'total_credit']
    fit_df = df_year.dropna()
    if len(fit_df) >= 2:
        X = sm.add_constant(fit_df['year'].values.astype(float))
        y = fit_df['total_credit'].values.astype(float)
        model = sm.OLS(y, X).fit()

        years_pred = np.arange(1398, 1405, 1).astype(float)
        Xp = sm.add_constant(years_pred)
        pred = model.get_prediction(Xp).summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(df_year['year'], df_year['total_credit'], s=60, label=fix_persian_text('مشاهدات'))
        ax.plot(years_pred, pred['mean'].values, lw=2.5, label=fix_persian_text('رگرسیون خطی'))
        ax.fill_between(years_pred, pred['mean_ci_lower'].values, pred['mean_ci_upper'].values,
                        alpha=0.2, label=fix_persian_text('بازه اطمینان ۹۵٪'))

        idx_1404 = np.where(years_pred == 1404)[0][0]
        yhat_1404 = pred['mean'].iloc[idx_1404]
        lo_1404 = pred['mean_ci_lower'].iloc[idx_1404]
        hi_1404 = pred['mean_ci_upper'].iloc[idx_1404]
        ax.scatter([1404], [yhat_1404], s=90, zorder=5, color='tab:red')
        ann = f"{convert_to_persian_number('۱۴۰۴')}: {format_number_with_separator(yhat_1404)}\n" \
              f"{convert_to_persian_number('بازه ۹۵٪')}: {format_number_with_separator(lo_1404)} – {format_number_with_separator(hi_1404)}"
        ax.text(1404 + 0.05, yhat_1404, fix_persian_text(ann), fontsize=9, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel(fix_persian_text('سال'), fontsize=12, fontweight='bold')
        ax.set_ylabel(fix_persian_text('اعتبارات (میلیون ریال)'), fontsize=12, fontweight='bold')
        ax.set_title(fix_persian_text('روند و پیش‌بینی اعتبارات سال ۱۴۰۴'), fontsize=15, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.set_xticks(range(1398, 1405))
        ax.set_xticklabels([convert_to_persian_number(str(y)) for y in range(1398, 1405)])
        ax.set_yticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_yticklabels()])
        ax.legend()
        plt.tight_layout()
        outpath = fig_dir / 'chart_6_4_regression_forecast.png'
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Chart 6-4 saved: {outpath}")
    else:
        print("⚠ داده کافی برای برازش رگرسیون خطی وجود ندارد.")
except Exception as e:
    print(f"⚠ Chart 6-4 (regression) failed: {e}")

print("✓ فصل ۶ — نمودارها تولید شد.")
