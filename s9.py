# -*- coding: utf-8 -*-
"""
فصل ۹ - تحلیل جغرافیایی تفصیلی سال ۱۴۰۳ (s9.py)
مطابق با سبک فصول قبلی:
- رندر متن فارسی RTL (arabic_reshaper + bidi)
- اعداد فارسی در تمام برچسب‌ها
- تنظیمات فونت Vazirmatn
خروجی‌ها:
  fig/S9/chart_9_1_heatmap_iran.png
  fig/S9/chart_9_2_top_bottom_provinces.png
  fig/S9/chart_9_3_top_provinces_comparison.png
  fig/S9/chart_9_4_provincial_growth.png
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

# Geo imports
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠ GeoPandas not installed. Choropleth will use simple heatmap instead.")

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

def normalize_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    for a,b in {'ي':'ی','ك':'ک','ۀ':'ه','ة':'ه','‌':'','\u200c':''}.items():
        s = s.replace(a,b)
    return s

def find_province_column(df: pd.DataFrame) -> str:
    candidates = ['استان', 'استان اجرا', 'نام استان', 'استان محل اجرا']
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("ستون «استان» در فایل داده یافت نشد.")

# =============================================================================
# Paths & IO
# =============================================================================
base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Sample_Data.xlsx'
if not data_file.exists():
    alt = base_dir / 'Q_Sample_Data.xlsx'
    if alt.exists():
        data_file = alt
    else:
        raise FileNotFoundError(f"Expected data file not found at {data_file} or {alt}")

fig_dir = base_dir / 'fig' / 'S9'
fig_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {fig_dir}")

# GeoJSON path
geojson_candidates = [
    Path(r'D:\OneDrive\AI-Project\Article56\iran-geojson\iran_geo.json'),
    base_dir / 'data' / 'iran_geo.json',
    base_dir / 'iran_geo.json'
]
geojson_file = None
for p in geojson_candidates:
    if p.exists():
        geojson_file = p
        break

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

province_col = find_province_column(df)
df[province_col] = df[province_col].map(normalize_name)

df_1402 = df[df[year_col] == 1402]
df_1403 = df[df[year_col] == 1403]

print(f"✓ Records 1402: {len(df_1402):,}")
print(f"✓ Records 1403: {len(df_1403):,}")
print(f"✓ Province column: {province_col}")

# =============================================================================
# نمودار ۹-۱: Heatmap نقشه ایران (۱۴۰۳) - Choropleth
# =============================================================================
if GEOPANDAS_AVAILABLE and geojson_file:
    try:
        gdf_prov = gpd.read_file(geojson_file)
        
        # استانداردسازی نام‌ها
        if 'NAME_1' not in gdf_prov.columns and 'name' in gdf_prov.columns:
            gdf_prov = gdf_prov.rename(columns={'name': 'NAME_1'})
        gdf_prov['prov_norm'] = gdf_prov['NAME_1'].map(normalize_name)
        
        # داده سال ۱۴۰۳
        df_1403['prov_norm'] = df_1403[province_col].map(normalize_name)
        province_counts = df_1403.groupby('prov_norm').size().reset_index(name='project_count')
        
        # ترکیب
        gdf_plot = gdf_prov.merge(province_counts, on='prov_norm', how='left')
        gdf_plot['project_count'] = gdf_plot['project_count'].fillna(0)
        gdf_plot['centroid'] = gdf_plot.geometry.representative_point()
        
        # رسم نقشه
        fig, ax = plt.subplots(figsize=(14, 16))
        gdf_plot.plot(
            ax=ax, column='project_count', cmap='RdYlGn_r',
            linewidth=0.8, edgecolor='#333', legend=True,
            legend_kwds={'label': fix_persian_text('تعداد طرح‌های پژوهشی'), 
                        'orientation': 'horizontal', 'shrink': 0.8}
        )
        
        # برچسب‌گذاری نام استان‌ها
        for _, row in gdf_plot.iterrows():
            if row['project_count'] > 0:
                x, y = row['centroid'].x, row['centroid'].y
                ax.text(x, y, fix_persian_text(row['NAME_1']), 
                       ha='center', va='center', fontsize=8, 
                       color='#000', alpha=0.85, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.7, edgecolor='none'))
        
        ax.set_title(fix_persian_text('توزیع جغرافیایی طرح‌های پژوهشی در سال ۱۴۰۳'), 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_axis_off()
        
        # فارسی‌سازی اعداد رنگ‌نما
        try:
            cax = ax.get_figure().axes[-1]
            cax.set_xticklabels([convert_to_persian_number(t.get_text()) 
                                for t in cax.get_xticklabels()])
        except Exception:
            pass
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        outpath = fig_dir / 'chart_9_1_heatmap_iran.png'
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Chart 9-1 saved: {outpath}")
    except Exception as e:
        print(f"⚠ Chart 9-1 (Choropleth) failed: {e}")
        GEOPANDAS_AVAILABLE = False

# Fallback: Simple heatmap matrix
if not GEOPANDAS_AVAILABLE or not geojson_file:
    try:
        province_counts_1403 = df_1403.groupby(province_col).size().sort_values(ascending=False)
        
        # ایجاد ماتریس برای heatmap
        n_provinces = len(province_counts_1403)
        rows = int(np.ceil(n_provinces / 5))
        matrix = np.zeros((rows, 5))
        
        for i, count in enumerate(province_counts_1403.values):
            row = i // 5
            col = i % 5
            matrix[row, col] = count
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   linewidths=1, linecolor='white', cbar_kws={'label': ''}, ax=ax)
        
        # برچسب‌گذاری
        labels = []
        for i, prov in enumerate(province_counts_1403.index):
            row = i // 5
            col = i % 5
            labels.append((row, col, fix_persian_text(prov)))
        
        for row, col, label in labels:
            ax.text(col + 0.5, row + 0.7, label, ha='center', va='center', 
                   fontsize=9, color='black', fontweight='bold')
        
        # فارسی‌سازی اعداد
        for text in ax.texts:
            current_text = text.get_text()
            if current_text and current_text.replace('.', '').isdigit():
                text.set_text(convert_to_persian_number(current_text))
        
        ax.set_title(fix_persian_text('توزیع جغرافیایی طرح‌های پژوهشی در سال ۱۴۰۳'), 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        outpath = fig_dir / 'chart_9_1_heatmap_iran.png'
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Chart 9-1 (fallback heatmap) saved: {outpath}")
    except Exception as e:
        print(f"⚠ Chart 9-1 (fallback) failed: {e}")

# =============================================================================
# نمودار ۹-۲: Top 10 و Bottom 10 استان در ۱۴۰۳
# =============================================================================
try:
    province_counts_1403 = df_1403.groupby(province_col).size().sort_values(ascending=False)
    
    top10 = province_counts_1403.head(10)
    bottom10 = province_counts_1403.tail(10)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top 10
    y_pos_top = np.arange(len(top10))
    bars_top = ax1.barh(y_pos_top, top10.values, color='#2ECC71', 
                        edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, val) in enumerate(zip(bars_top, top10.values)):
        ax1.text(val + max(top10.values) * 0.01, i, 
                format_number_with_separator(val, use_persian=True),
                va='center', ha='left', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_yticks(y_pos_top)
    ax1.set_yticklabels([fix_persian_text(p) for p in top10.index], fontsize=12)
    ax1.set_xlabel(fix_persian_text('تعداد طرح'), fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title(fix_persian_text('۱۰ استان برتر در سال ۱۴۰۳'), 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('#F0FFF0')
    ax1.invert_yaxis()
    
    for spine in ax1.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    # Bottom 10
    y_pos_bottom = np.arange(len(bottom10))
    bars_bottom = ax2.barh(y_pos_bottom, bottom10.values, color='#E74C3C', 
                           edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, val) in enumerate(zip(bars_bottom, bottom10.values)):
        ax2.text(val + max(bottom10.values) * 0.01, i, 
                format_number_with_separator(val, use_persian=True),
                va='center', ha='left', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_yticks(y_pos_bottom)
    ax2.set_yticklabels([fix_persian_text(p) for p in bottom10.index], fontsize=12)
    ax2.set_xlabel(fix_persian_text('تعداد طرح'), fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title(fix_persian_text('۱۰ استان با کمترین طرح در سال ۱۴۰۳'), 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#FFF0F0')
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    fig.suptitle(fix_persian_text('استان‌های برتر و محروم در سال ۱۴۰۳'), 
                fontsize=18, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    outpath = fig_dir / 'chart_9_2_top_bottom_provinces.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 9-2 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 9-2 failed: {e}")

# =============================================================================
# نمودار ۹-۳: مقایسه استان‌های برتر (۱۴۰۲ vs ۱۴۰۳)
# =============================================================================
try:
    province_counts_1402 = df_1402.groupby(province_col).size()
    province_counts_1403 = df_1403.groupby(province_col).size()
    
    # انتخاب Top 10 از ۱۴۰۳
    top10_provinces = province_counts_1403.sort_values(ascending=False).head(10).index
    
    # داده‌های مقایسه
    counts_1402_top = [province_counts_1402.get(p, 0) for p in top10_provinces]
    counts_1403_top = [province_counts_1403.get(p, 0) for p in top10_provinces]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top10_provinces))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts_1402_top, width, 
                   label=fix_persian_text('۱۴۰۲'), color='#3498DB', 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x + width/2, counts_1403_top, width, 
                   label=fix_persian_text('۱۴۰۳'), color='#E74C3C', 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       convert_to_persian_number(str(int(height))),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    persian_provinces = [fix_persian_text(p) for p in top10_provinces]
    ax.set_xticks(x)
    ax.set_xticklabels(persian_provinces, rotation=25, ha='right', fontsize=11)
    
    ax.set_ylabel(fix_persian_text('تعداد طرح'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('تغییرات تعداد طرح‌ها در استان‌های برتر (۱۴۰۲ و ۱۴۰۳)'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.legend(fontsize=13, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_9_3_top_provinces_comparison.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 9-3 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 9-3 failed: {e}")

# =============================================================================
# نمودار ۹-۴: نمودار رشد استانی
# =============================================================================
try:
    province_counts_1402 = df_1402.groupby(province_col).size()
    province_counts_1403 = df_1403.groupby(province_col).size()
    
    # محاسبه رشد
    common_provinces = province_counts_1402.index.intersection(province_counts_1403.index)
    growth_data = []
    
    for prov in common_provinces:
        c_1402 = province_counts_1402[prov]
        c_1403 = province_counts_1403[prov]
        if c_1402 > 0:
            growth_pct = ((c_1403 - c_1402) / c_1402) * 100
            growth_data.append({'province': prov, 'growth': growth_pct})
    
    df_growth = pd.DataFrame(growth_data).sort_values('growth', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    colors_growth = ['#2ECC71' if g >= 0 else '#E74C3C' for g in df_growth['growth'].values]
    y_pos = np.arange(len(df_growth))
    
    bars = ax.barh(y_pos, df_growth['growth'].values, color=colors_growth, 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر
    for i, (bar, val) in enumerate(zip(bars, df_growth['growth'].values)):
        offset = max(abs(df_growth['growth'].values)) * 0.02
        ax.text(val + offset if val >= 0 else val - offset, i, 
               convert_to_persian_number(f'{val:.1f}') + '%',
               va='center', ha='left' if val >= 0 else 'right', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    persian_provinces = [fix_persian_text(p) for p in df_growth['province'].values]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(persian_provinces, fontsize=11)
    
    ax.set_xlabel(fix_persian_text('نرخ رشد (%)'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('نرخ رشد طرح‌های پژوهشی استان‌ها (۱۴۰۲ به ۱۴۰۳)'), 
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
    outpath = fig_dir / 'chart_9_4_provincial_growth.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 9-4 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 9-4 failed: {e}")

# =============================================================================
# خلاصه آماری فصل ۹
# =============================================================================
print("\n" + "="*70)
print("Statistical Summary for Chapter 9:")
print("="*70)

try:
    print(f"\n🗺️ تحلیل جغرافیایی سال ۱۴۰۳:")
    print(f"   • تعداد کل استان‌های فعال: {df_1403[province_col].nunique()}")
    
    top_province = province_counts_1403.idxmax()
    top_count = province_counts_1403.max()
    print(f"   • استان برتر: {top_province} ({top_count} طرح)")
    
    bottom_province = province_counts_1403.idxmin()
    bottom_count = province_counts_1403.min()
    print(f"   • استان با کمترین طرح: {bottom_province} ({bottom_count} طرح)")
    
    # محاسبه ضریب جینی
    counts_array = province_counts_1403.values
    sorted_counts = np.sort(counts_array)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    print(f"   • ضریب جینی (نابرابری جغرافیایی): {gini:.3f}")
    
    # تمرکز جغرافیایی
    top5_share = (province_counts_1403.head(5).sum() / province_counts_1403.sum()) * 100
    print(f"   • سهم ۵ استان برتر از کل: {top5_share:.1f}%")
    
    if len(df_1402) > 0:
        print(f"\n📈 مقایسه با سال ۱۴۰۲:")
        total_1402 = len(df_1402)
        total_1403 = len(df_1403)
        overall_growth = ((total_1403 - total_1402) / total_1402) * 100
        print(f"   • رشد کلی کشوری: {overall_growth:.1f}%")
        
        # استان‌های با بیشترین رشد
        if len(growth_data) > 0:
            df_growth_sorted = pd.DataFrame(growth_data).sort_values('growth', ascending=False)
            print(f"   • استان با بیشترین رشد: {df_growth_sorted.iloc[0]['province']} "
                  f"({df_growth_sorted.iloc[0]['growth']:.1f}%)")
            print(f"   • استان با بیشترین کاهش: {df_growth_sorted.iloc[-1]['province']} "
                  f"({df_growth_sorted.iloc[-1]['growth']:.1f}%)")

except Exception as e:
    print(f"⚠ Error in statistical summary: {e}")

print("\n✅ فصل ۹ – نمودارها تولید شد.")