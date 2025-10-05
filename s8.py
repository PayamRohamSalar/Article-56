# -*- coding: utf-8 -*-
"""
Outputs:
  fig/S8/chart_8_1_top20_devices_count.png
  fig/S8/chart_8_2_top20_devices_credit.png
  fig/S8/chart_8_3_emerging_devices.png
  fig/S8/chart_8_4_treemap.png
  fig/S8/chart_8_5_donut_areas.png
  fig/S8/chart_8_6_areas_comparison.png
  fig/S8/chart_8_7_sunburst.png
  fig/S8/chart_8_8_project_types_by_area.png
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

# Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly not installed. Treemap and Sunburst will be skipped.")

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

fig_dir = base_dir / 'fig' / 'S8'
fig_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {fig_dir}")

# =============================================================================
# Load & Prepare Data
# =============================================================================
df = pd.read_excel(data_file)

year_col = 'سال'
device_col = 'نام سازمان'
credit_col = 'اعتبار'
area_col = 'زمینه اولویت'
project_type_col = 'نوع طرح'

df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
df[credit_col] = pd.to_numeric(df[credit_col], errors='coerce')
df = df[df[year_col].between(1398, 1403, inclusive='both')]
df = df[df[credit_col].notna()]
df[device_col] = df[device_col].map(normalize_name)

df_1402 = df[df[year_col] == 1402]
df_1403 = df[df[year_col] == 1403]

print(f"✓ Records 1402: {len(df_1402):,}")
print(f"✓ Records 1403: {len(df_1403):,}")

# =============================================================================
# نمودار ۸-۱: Top 20 دستگاه در ۱۴۰۳ (تعداد)
# =============================================================================
try:
    device_counts_1403 = df_1403.groupby(device_col).size().sort_values(ascending=True).tail(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(device_counts_1403)))
    bars = ax.barh(range(len(device_counts_1403)), device_counts_1403.values, 
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر
    for i, (bar, value) in enumerate(zip(bars, device_counts_1403.values)):
        persian_value = format_number_with_separator(value, use_persian=True)
        ax.text(value + max(device_counts_1403.values) * 0.01, i, persian_value,
                va='center', ha='left', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    persian_orgs = [fix_persian_text(org) for org in device_counts_1403.index]
    ax.set_yticks(range(len(device_counts_1403)))
    ax.set_yticklabels(persian_orgs, fontsize=11)
    
    ax.set_xlabel(fix_persian_text('تعداد طرح‌های پژوهشی'), 
                  fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(fix_persian_text('نام دستگاه اجرایی'), 
                  fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('۲۰ دستگاه برتر سال ۱۴۰۳ از نظر تعداد طرح'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_8_1_top20_devices_count.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 8-1 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 8-1 failed: {e}")

# =============================================================================
# نمودار ۸-۲: Top 20 دستگاه در ۱۴۰۳ (اعتبار)
# =============================================================================
try:
    device_credit_1403 = (df_1403.groupby(device_col)[credit_col].sum() / 1000).sort_values(ascending=True).tail(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(device_credit_1403)))
    bars = ax.barh(range(len(device_credit_1403)), device_credit_1403.values, 
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, value) in enumerate(zip(bars, device_credit_1403.values)):
        persian_value = format_number_with_separator(value, use_persian=True)
        ax.text(value + max(device_credit_1403.values) * 0.01, i, persian_value,
                va='center', ha='left', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    persian_orgs = [fix_persian_text(org) for org in device_credit_1403.index]
    ax.set_yticks(range(len(device_credit_1403)))
    ax.set_yticklabels(persian_orgs, fontsize=11)
    
    ax.set_xlabel(fix_persian_text('حجم اعتبارات (میلیارد ریال)'), 
                  fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(fix_persian_text('نام دستگاه اجرایی'), 
                  fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('۲۰ دستگاه برتر سال ۱۴۰۳ از نظر حجم اعتبارات'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_8_2_top20_devices_credit.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 8-2 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 8-2 failed: {e}")

# =============================================================================
# نمودار ۸-۳: دستگاه‌های نوظهور (رشد >50%)
# =============================================================================
try:
    counts_1402 = df_1402.groupby(device_col).size()
    counts_1403 = df_1403.groupby(device_col).size()
    
    # محاسبه رشد
    common_devices = counts_1402.index.intersection(counts_1403.index)
    growth_data = []
    for dev in common_devices:
        c_1402 = counts_1402[dev]
        c_1403 = counts_1403[dev]
        if c_1402 > 0:
            growth_pct = ((c_1403 - c_1402) / c_1402) * 100
            if growth_pct > 50:
                growth_data.append({
                    'device': dev,
                    'count_1402': c_1402,
                    'count_1403': c_1403,
                    'growth': growth_pct
                })
    
    if len(growth_data) > 0:
        df_growth = pd.DataFrame(growth_data).sort_values('growth', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        x = np.arange(len(df_growth))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_growth['count_1402'].values, width, 
                       label=fix_persian_text('۱۴۰۲'), color='#3498DB', 
                       edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax.bar(x + width/2, df_growth['count_1403'].values, width, 
                       label=fix_persian_text('۱۴۰۳'), color='#2ECC71', 
                       edgecolor='black', linewidth=1.2, alpha=0.85)
        
        # افزودن مقادیر
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       format_number_with_separator(int(height), use_persian=True),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        persian_devices = [fix_persian_text(d[:30] + '...' if len(d) > 30 else d) 
                          for d in df_growth['device'].values]
        ax.set_xticks(x)
        ax.set_xticklabels(persian_devices, rotation=45, ha='right', fontsize=10)
        
        ax.set_ylabel(fix_persian_text('تعداد طرح'), fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(fix_persian_text('دستگاه‌های با رشد چشمگیر در سال ۱۴۰۳ (رشد بیش از ۵۰٪)'), 
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
        outpath = fig_dir / 'chart_8_3_emerging_devices.png'
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Chart 8-3 saved: {outpath}")
    else:
        print("⚠ Chart 8-3: No devices with >50% growth found")
except Exception as e:
    print(f"⚠ Chart 8-3 failed: {e}")

# =============================================================================
# نمودار ۸-۴: Treemap دستگاه‌ها بر اساس اعتبار
# =============================================================================
if PLOTLY_AVAILABLE:
    try:
        org_credit = df_1403.groupby(device_col)[credit_col].sum().reset_index()
        org_credit = org_credit.sort_values(credit_col, ascending=False).head(20)
        org_credit['device_persian'] = org_credit[device_col].apply(fix_persian_text)
        
        fig = px.treemap(org_credit, 
                        path=['device_persian'], 
                        values=credit_col,
                        title='توزیع سلسله‌مراتبی اعتبارات بین دستگاه‌ها (۱۴۰۳)',
                        color=credit_col,
                        color_continuous_scale='RdYlGn')
        
        fig.update_traces(textposition='middle center', 
                         textfont_size=11,
                         marker=dict(line=dict(width=2, color='white')))
        
        fig.update_layout(title_font_size=18,
                         title_font_family='Vazirmatn',
                         font=dict(family='Vazirmatn', size=12),
                         width=1400,
                         height=900)
        
        outpath = fig_dir / 'chart_8_4_treemap.png'
        fig.write_image(str(outpath), width=1400, height=900)
        print(f"✓ Chart 8-4 saved: {outpath}")
    except Exception as e:
        print(f"⚠ Chart 8-4 failed: {e}")
else:
    print("⚠ Chart 8-4 skipped (Plotly not available)")

# =============================================================================
# نمودار ۸-۵: Donut Chart حوزه‌های راهبردی ۱۴۰۳
# =============================================================================
try:
    area_budget = df_1403.groupby(area_col)[credit_col].sum().sort_values(ascending=False)
    top_areas = area_budget.head(10)
    other_sum = area_budget[10:].sum()
    
    if other_sum > 0:
        top_areas = pd.concat([top_areas, pd.Series({'سایر': other_sum})])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_areas)))
    
    wedges, texts, autotexts = ax.pie(top_areas.values, 
                                        labels=[fix_persian_text(label) for label in top_areas.index],
                                        autopct='%1.1f%%',
                                        startangle=90, 
                                        colors=colors,
                                        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    
    # فارسی‌سازی درصدها
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        current_text = autotext.get_text()
        persian_pct = convert_to_persian_number(current_text.replace('%', '')) + '%'
        autotext.set_text(persian_pct)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    
    ax.set_title(fix_persian_text('توزیع اعتبارات سال ۱۴۰۳ بر اساس حوزه‌های راهبردی'), 
                 fontsize=16, fontweight='bold', pad=25)
    ax.axis('equal')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_8_5_donut_areas.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 8-5 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 8-5 failed: {e}")

# =============================================================================
# نمودار ۸-۶: مقایسه سهم حوزه‌ها (۱۴۰۲ vs ۱۴۰۳)
# =============================================================================
try:
    area_budget_1402 = df_1402.groupby(area_col)[credit_col].sum()
    area_budget_1403 = df_1403.groupby(area_col)[credit_col].sum()
    
    total_1402 = area_budget_1402.sum()
    total_1403 = area_budget_1403.sum()
    
    area_pct_1402 = (area_budget_1402 / total_1402 * 100).sort_values(ascending=False).head(10)
    area_pct_1403 = (area_budget_1403 / total_1403 * 100)
    
    # ترکیب
    df_compare = pd.DataFrame({
        '1402': area_pct_1402,
        '1403': [area_pct_1403.get(idx, 0) for idx in area_pct_1402.index]
    }).sort_values('1403', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y = np.arange(len(df_compare))
    width = 0.35
    
    bars1 = ax.barh(y - width/2, df_compare['1402'].values, width, 
                    label=fix_persian_text('۱۴۰۲'), color='#3498DB', 
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.barh(y + width/2, df_compare['1403'].values, width, 
                    label=fix_persian_text('۱۴۰۳'), color='#E74C3C', 
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # افزودن مقادیر
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            width_val = bar.get_width()
            ax.text(width_val + max(df_compare.max()) * 0.01, bar.get_y() + bar.get_height()/2,
                   convert_to_persian_number(f'{width_val:.1f}') + '%',
                   va='center', ha='left', fontsize=10, fontweight='bold')
    
    persian_areas = [fix_persian_text(area) for area in df_compare.index]
    ax.set_yticks(y)
    ax.set_yticklabels(persian_areas, fontsize=11)
    
    ax.set_xlabel(fix_persian_text('سهم درصدی از کل اعتبارات'), 
                  fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('تغییرات سهم حوزه‌های راهبردی (۱۴۰۲ به ۱۴۰۳)'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.legend(fontsize=13, loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_8_6_areas_comparison.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 8-6 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 8-6 failed: {e}")

# =============================================================================
# نمودار ۸-۷: Sunburst Chart (حوزه > دستگاه)
# =============================================================================
if PLOTLY_AVAILABLE:
    try:
        # آماده‌سازی داده
        df_sunburst = df_1403.groupby([area_col, device_col])[credit_col].sum().reset_index()
        df_sunburst = df_sunburst.sort_values(credit_col, ascending=False).head(50)
        
        df_sunburst['area_persian'] = df_sunburst[area_col].apply(fix_persian_text)
        df_sunburst['device_persian'] = df_sunburst[device_col].apply(lambda x: fix_persian_text(x[:25] + '...' if len(str(x)) > 25 else x))
        
        fig = px.sunburst(df_sunburst,
                         path=['area_persian', 'device_persian'],
                         values=credit_col,
                         title='توزیع سلسله‌مراتبی: حوزه‌ها و دستگاه‌ها (۱۴۰۳)',
                         color=credit_col,
                         color_continuous_scale='Viridis')
        
        fig.update_traces(textfont_size=11,
                         marker=dict(line=dict(width=2, color='white')))
        
        fig.update_layout(title_font_size=18,
                         title_font_family='Vazirmatn',
                         font=dict(family='Vazirmatn', size=11),
                         width=1200,
                         height=1200)
        
        outpath = fig_dir / 'chart_8_7_sunburst.png'
        fig.write_image(str(outpath), width=1200, height=1200)
        print(f"✓ Chart 8-7 saved: {outpath}")
    except Exception as e:
        print(f"⚠ Chart 8-7 failed: {e}")
else:
    print("⚠ Chart 8-7 skipped (Plotly not available)")

# =============================================================================
# نمودار ۸-۸: تحلیل موضوعی حوزه‌های اصلی (چندستونی)
# =============================================================================
try:
    # انتخاب حوزه‌های اصلی
    top_areas_list = df_1403.groupby(area_col)[credit_col].sum().sort_values(ascending=False).head(6).index.tolist()
    
    df_filtered = df_1403[df_1403[area_col].isin(top_areas_list)]
    
    # محاسبه تعداد بر اساس نوع طرح
    project_type_dist = df_filtered.groupby([area_col, project_type_col]).size().unstack(fill_value=0)
    
    # مرتب‌سازی
    project_type_order = ['بنیادی', 'کاربردی', 'توسعه ای']
    project_type_dist = project_type_dist[[col for col in project_type_order if col in project_type_dist.columns]]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(project_type_dist))
    width = 0.25
    colors_types = {'بنیادی': '#9B59B6', 'کاربردی': '#3498DB', 'توسعه ای': '#2ECC71'}
    
    for i, ptype in enumerate(project_type_dist.columns):
        offset = (i - len(project_type_dist.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, project_type_dist[ptype].values, width,
                     label=fix_persian_text(ptype), 
                     color=colors_types.get(ptype, '#95A5A6'),
                     edgecolor='black', linewidth=1.2, alpha=0.85)
        
        # افزودن مقادیر
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       convert_to_persian_number(str(int(height))),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    persian_areas = [fix_persian_text(area) for area in project_type_dist.index]
    ax.set_xticks(x)
    ax.set_xticklabels(persian_areas, rotation=20, ha='right', fontsize=11)
    
    ax.set_ylabel(fix_persian_text('تعداد طرح'), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(fix_persian_text('توزیع نوع طرح در حوزه‌های راهبردی اصلی (۱۴۰۳)'), 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # ادامه نمودار ۸-۸
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    outpath = fig_dir / 'chart_8_8_project_types_by_area.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Chart 8-8 saved: {outpath}")
except Exception as e:
    print(f"⚠ Chart 8-8 failed: {e}")

# =============================================================================
# خلاصه آماری فصل ۸
# =============================================================================
print("\n" + "="*70)
print("Statistical Summary for Chapter 8:")
print("="*70)

try:
    print(f"\n📊 تحلیل دستگاه‌ها در سال ۱۴۰۳:")
    print(f"   • تعداد کل دستگاه‌ها: {df_1403[device_col].nunique()}")
    top_device = df_1403.groupby(device_col).size().idxmax()
    top_device_count = df_1403.groupby(device_col).size().max()
    print(f"   • دستگاه برتر (تعداد): {top_device} ({top_device_count} طرح)")
    
    top_device_credit = df_1403.groupby(device_col)[credit_col].sum().idxmax()
    top_credit_value = df_1403.groupby(device_col)[credit_col].sum().max() / 1000
    print(f"   • دستگاه برتر (اعتبار): {top_device_credit} ({top_credit_value:,.0f} میلیارد ریال)")
    
    print(f"\n🎯 تحلیل حوزه‌های راهبردی در سال ۱۴۰۳:")
    print(f"   • تعداد کل حوزه‌ها: {df_1403[area_col].nunique()}")
    top_area = df_1403.groupby(area_col)[credit_col].sum().idxmax()
    top_area_credit = df_1403.groupby(area_col)[credit_col].sum().max() / 1000
    top_area_pct = (top_area_credit / (df_1403[credit_col].sum() / 1000)) * 100
    print(f"   • حوزه برتر: {top_area}")
    print(f"   • اعتبار حوزه برتر: {top_area_credit:,.0f} میلیارد ریال ({top_area_pct:.1f}%)")
    
    if project_type_col in df_1403.columns:
        print(f"\n📑 توزیع نوع طرح در سال ۱۴۰۳:")
        type_dist = df_1403.groupby(project_type_col).size()
        for ptype, count in type_dist.items():
            pct = (count / len(df_1403)) * 100
            print(f"   • {ptype}: {count} طرح ({pct:.1f}%)")
    
    # تحلیل رشد دستگاه‌ها
    if len(df_1402) > 0:
        new_devices = set(df_1403[device_col].unique()) - set(df_1402[device_col].unique())
        print(f"\n🆕 دستگاه‌های جدید در سال ۱۴۰۳: {len(new_devices)}")
        
        if len(new_devices) > 0 and len(new_devices) <= 10:
            print("   دستگاه‌های جدید:")
            for dev in list(new_devices)[:10]:
                print(f"     - {dev}")
    
except Exception as e:
    print(f"⚠ Error in statistical summary: {e}")

print("\n✅ فصل ۸ – نمودارها تولید شد.")