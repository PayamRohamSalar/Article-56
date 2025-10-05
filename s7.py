# -*- coding: utf-8 -*-
"""
Outputs:
  fig/S7/chart_7_1_comparison_1402_1403.png
  fig/S7/chart_7_2_growth_rate.png
  fig/S7/chart_7_3_radar_chart.png
  fig/S7/chart_7_4_boxplot_outliers.png
  fig/S7/chart_7_5_trend_line.png
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

# Font handling
from matplotlib import font_manager

# =============================================================================
# Font Configuration
# =============================================================================
font_path = Path(r'D:\OneDrive\AI-Project\Article56\fonts\ttf\Vazirmatn-Regular.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Vazirmatn'
else:
    plt.rcParams['font.family'] = 'Vazirmatn'
    try:
        fam = plt.rcParams['font.family']
        fam_ok = (fam.lower() == 'vazirmatn') if isinstance(fam, str) else ('Vazirmatn' in fam)
        if not fam_ok:
            print(f"Warning: Font not found at {font_path}")
            print("Falling back to default font...")
            plt.rcParams['font.family'] = 'Tahoma'
    except Exception:
        plt.rcParams['font.family'] = 'Tahoma'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

# =============================================================================
# Helpers
# =============================================================================
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

# =============================================================================
# Paths & IO
# =============================================================================
base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Useful_Source_Data.xlsx'
if not data_file.exists():
    alt = base_dir / 'Q_Useful_Source_Data.xlsx'
    if alt.exists():
        data_file = alt
    else:
        raise FileNotFoundError(f"Expected data file not found at {data_file} or {alt}")

fig_dir = base_dir / 'fig' / 'S7'
fig_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {fig_dir}")

# =============================================================================
# Load & Prepare Data
# =============================================================================
df = pd.read_excel(data_file)

year_col = 'سال'
credit_col = 'اعتبار'

df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
df[credit_col] = pd.to_numeric(df[credit_col], errors='coerce')
df = df[df[year_col].between(1398, 1403, inclusive='both')]
df = df[df[credit_col].notna()]

print(f"✓ Total records: {len(df):,}")
print(f"✓ Years: {df[year_col].min()} – {df[year_col].max()}")

# =============================================================================
# نمودار ۷-۱: مقایسه ۱۴۰۳ با ۱۴۰۲ (ستونی دوگانه)
# =============================================================================
try:
    # محاسبه شاخص‌ها
    def calc_metrics(year_data):
        return {
            'تعداد طرح': len(year_data),
            'اعتبار کل (میلیارد ریال)': year_data[credit_col].sum() / 1000,
            'میانگین اعتبار': year_data[credit_col].mean(),
            'تعداد دستگاه': year_data['نام سازمان'].nunique() if 'نام سازمان' in year_data.columns else 0,
            'تعداد استان': year_data['استان اجرا'].nunique() if 'استان اجرا' in year_data.columns else 0,
        }
    
    df_1402 = df[df[year_col] == 1402]
    df_1403 = df[df[year_col] == 1403]
    
    metrics_1402 = calc_metrics(df_1402)
    metrics_1403 = calc_metrics(df_1403)
    
    # رسم نمودار
    fig, ax = plt.subplots(figsize=(14, 8))
    
    indicators = list(metrics_1402.keys())
    x = np.arange(len(indicators))
    width = 0.35
    
    values_1402 = [metrics_1402[k] for k in indicators]
    values_1403 = [metrics_1403[k] for k in indicators]
    
    bars1 = ax.bar(x - width/2, values_1402, width, label=fix_persian_text('۱۴۰۲'), 
                   color='#3498DB', edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x + width/2, values_1403, width, label=fix_persian_text('۱۴۰۳'), 
                   color='#2ECC71', edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر روی ستون‌ها
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_number_with_separator(height, use_persian=True),
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(fix_persian_text('شاخص‌ها'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(fix_persian_text('مقدار'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('مقایسه شاخص‌های کلیدی: سال ۱۴۰۲ و ۱۴۰۳'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.set_xticks(x)
    ax.set_xticklabels([fix_persian_text(ind) for ind in indicators], rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=13, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_7_1_comparison_1402_1403.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 7-1 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 7-1 failed: {e}")

# =============================================================================
# نمودار ۷-۲: نرخ رشد شاخص‌ها (ستونی افقی)
# =============================================================================
try:
    growth_rates = {}
    for key in metrics_1402.keys():
        val_1402 = metrics_1402[key]
        val_1403 = metrics_1403[key]
        if val_1402 > 0:
            growth_rates[key] = ((val_1403 - val_1402) / val_1402) * 100
        else:
            growth_rates[key] = 0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    indicators_growth = list(growth_rates.keys())
    values_growth = list(growth_rates.values())
    colors_growth = ['#2ECC71' if v >= 0 else '#E74C3C' for v in values_growth]
    
    y_pos = np.arange(len(indicators_growth))
    bars = ax.barh(y_pos, values_growth, color=colors_growth, edgecolor='black', 
                   linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر
    for i, (bar, val) in enumerate(zip(bars, values_growth)):
        ax.text(val + (max(values_growth) - min(values_growth)) * 0.02, i, 
               convert_to_persian_number(f'{val:.1f}') + '%',
               va='center', ha='left' if val >= 0 else 'right', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([fix_persian_text(ind) for ind in indicators_growth], fontsize=11)
    ax.set_xlabel(fix_persian_text('نرخ رشد (%)'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('نرخ رشد شاخص‌های سال ۱۴۰۳ نسبت به ۱۴۰۲'), 
                 fontsize=16, fontweight='bold', pad=25)
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_7_2_growth_rate.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 7-2 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 7-2 failed: {e}")

# =============================================================================
# نمودار ۷-۳: Radar Chart مقایسه با میانگین ۵ ساله
# =============================================================================
try:
    # محاسبه میانگین ۵ ساله
    years_all = [1398, 1399, 1400, 1402, 1403]
    metrics_all_years = {}
    for yr in years_all:
        df_yr = df[df[year_col] == yr]
        metrics_all_years[yr] = calc_metrics(df_yr)
    
    # محاسبه میانگین
    metrics_avg = {}
    for key in metrics_1403.keys():
        vals = [metrics_all_years[yr][key] for yr in years_all if yr in metrics_all_years]
        metrics_avg[key] = np.mean(vals) if vals else 0
    
    # نرمال‌سازی به بازه 0-100
    def normalize_values(metrics_dict, metrics_avg_dict):
        normalized = {}
        for key in metrics_dict.keys():
            val = metrics_dict[key]
            avg_val = metrics_avg_dict[key]
            max_val = max(val, avg_val) if max(val, avg_val) > 0 else 1
            normalized[key] = (val / max_val) * 100
        return normalized
    
    values_1403_norm = normalize_values(metrics_1403, metrics_avg)
    values_avg_norm = normalize_values(metrics_avg, metrics_avg)
    
    categories = list(values_1403_norm.keys())
    values_1403_list = [values_1403_norm[k] for k in categories]
    values_avg_list = [values_avg_norm[k] for k in categories]
    
    # تکمیل دایره
    values_1403_list += values_1403_list[:1]
    values_avg_list += values_avg_list[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values_1403_list, 'o-', linewidth=2.5, 
           label=fix_persian_text('سال ۱۴۰۳'), color='#E74C3C', markersize=8)
    ax.fill(angles, values_1403_list, alpha=0.25, color='#E74C3C')
    
    ax.plot(angles, values_avg_list, 'o-', linewidth=2.5, 
           label=fix_persian_text('میانگین ۵ ساله'), color='#3498DB', markersize=8)
    ax.fill(angles, values_avg_list, alpha=0.15, color='#3498DB')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([fix_persian_text(cat) for cat in categories], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([convert_to_persian_number(str(i)) for i in [20, 40, 60, 80, 100]], fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True)
    ax.set_title(fix_persian_text('مقایسه چندبعدی سال ۱۴۰۳ با میانگین پنج‌ساله'), 
                size=16, weight='bold', pad=30)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_7_3_radar_chart.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 7-3 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 7-3 failed: {e}")

# =============================================================================
# نمودار ۷-۴: Boxplot شناسایی Outliers در ۱۴۰۳
# =============================================================================
try:
    credits_1403 = df_1403[credit_col].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bp = ax.boxplot([credits_1403], vert=True, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='#3498DB', alpha=0.7, linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2.5, color='red'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6))
    
    ax.set_yscale('log')
    ax.set_ylabel(fix_persian_text('اعتبار (میلیون ریال - مقیاس لگاریتمی)'), 
                 fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('توزیع آماری اعتبارات و شناسایی طرح‌های غیرمتعارف (۱۴۰۳)'), 
                fontsize=16, fontweight='bold', pad=25)
    ax.set_xticklabels([fix_persian_text('سال ۱۴۰۳')], fontsize=13, fontweight='bold')
    
    def format_y_log(x, p):
        return format_number_with_separator(x, use_persian=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_log))
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    # محاسبه و نمایش آمار
    q1, median, q3 = np.percentile(credits_1403, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = credits_1403[(credits_1403 < lower_fence) | (credits_1403 > upper_fence)]
    
    stats_text = f"میانه: {format_number_with_separator(median)}\n"
    stats_text += f"تعداد Outliers: {convert_to_persian_number(str(len(outliers)))}"
    ax.text(0.02, 0.98, fix_persian_text(stats_text), transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_7_4_boxplot_outliers.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 7-4 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 7-4 failed: {e}")

# =============================================================================
# نمودار ۷-۵: نمودار خطی تطبیقی (۱۴۰۳ vs تاریخی)
# =============================================================================
try:
    yearly_credit = df.groupby(year_col)[credit_col].sum() / 1000  # میلیارد ریال
    years = yearly_credit.index.tolist()
    credits = yearly_credit.values.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # خط روند واقعی
    ax.plot(years, credits, marker='o', linewidth=3, markersize=12,
           color='#2E86AB', markerfacecolor='#2E86AB', 
           markeredgewidth=2, markeredgecolor='white',
           label=fix_persian_text('روند واقعی'), zorder=3)
    
    # میانگین متحرک (اگر داده کافی باشد)
    if len(credits) >= 3:
        moving_avg = pd.Series(credits).rolling(window=3, center=True).mean()
        ax.plot(years, moving_avg, linestyle='--', linewidth=2.5, 
               color='#F39C12', alpha=0.7,
               label=fix_persian_text('میانگین متحرک'), zorder=2)
    
    # برجسته‌سازی ۱۴۰۳
    if 1403 in years:
        idx_1403 = years.index(1403)
        ax.scatter([1403], [credits[idx_1403]], s=300, zorder=5, 
                  color='#E74C3C', edgecolors='white', linewidths=3)
        ax.text(1403, credits[idx_1403] * 1.05, 
               fix_persian_text(f'۱۴۰۳: {format_number_with_separator(credits[idx_1403])}'),
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # افزودن مقادیر
    for x, y in zip(years, credits):
        if x != 1403:  # برای ۱۴۰۳ قبلاً اضافه شده
            ax.text(x, y + max(credits) * 0.02, 
                   format_number_with_separator(y, use_persian=True), 
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(fix_persian_text('اعتبارات (میلیارد ریال)'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('موقعیت سال ۱۴۰۳ در روند پنج‌ساله اعتبارات'), 
                fontsize=16, fontweight='bold', pad=25)
    
    ax.set_xticks(years)
    ax.set_xticklabels([convert_to_persian_number(str(y)) for y in years], fontsize=12, fontweight='bold')
    ax.set_yticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_yticklabels()])
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_7_5_trend_line.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 7-5 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 7-5 failed: {e}")

print("✓ فصل ۷ – نمودارها تولید شد.")