import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# To show Farsi Font
import arabic_reshaper
from bidi.algorithm import get_display

# ==============================================================================
# Font Configuration for Persian (Farsi) Text
# ==============================================================================

# Add custom font path
font_path = Path(r'D:\OneDrive\AI-Project\Article56\fonts\ttf\Vazirmatn-Regular.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Vazirmatn'
else:
    print(f"Warning: Font not found at {font_path}")
    print("Falling back to default font...")
    plt.rcParams['font.family'] = 'Tahoma'

# Configure matplotlib for RTL (Right-to-Left) text
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

# ==============================================================================
# Helper Functions
# ==============================================================================

def fix_persian_text(text):
    """
    Convert Persian/Arabic text to displayable format in matplotlib
    """
    if text is None or str(text).strip() == '':
        return ''
    
    try:
        reshaped_text = arabic_reshaper.reshape(str(text))
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        print(f"Warning: Could not reshape text '{text}': {e}")
        return str(text)

def convert_to_persian_number(number):
    """
    Convert English numbers to Persian numbers
    """
    english_digits = '0123456789'
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    translation_table = str.maketrans(english_digits, persian_digits)
    return str(number).translate(translation_table)

def format_number_with_separator(number, use_persian=True):
    """
    Format numbers with thousand separator and optional Persian conversion
    """
    formatted = f'{number:,.0f}' if isinstance(number, (int, float)) else str(number)
    
    if use_persian:
        return convert_to_persian_number(formatted)
    return formatted

def herfindahl_index(market_shares):
    """
    Calculate Herfindahl-Hirschman Index (HHI)
    
    Parameters:
    -----------
    market_shares : array-like
        List or array of market shares (between 0 and 1)
        
    Returns:
    --------
    float
        HHI value (0 to 1)
    """
    return sum([s**2 for s in market_shares])

# ==============================================================================
# Setup Output Directory
# ==============================================================================

output_dir = Path.cwd() / 'fig'
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# ==============================================================================
# Load Data
# ==============================================================================

base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Sample_Data.xlsx'
if not data_file.exists():
    raise FileNotFoundError(f"Expected data file not found: {data_file}")

df = pd.read_excel(data_file)

# Data cleaning and preparation
df['سال'] = df['سال'].astype(int)
df['اعتبار'] = pd.to_numeric(df['اعتبار'], errors='coerce')
df = df[df['اعتبار'].notna()]

print(f"Total records: {len(df):,}")
print(f"Year range: {df['سال'].min()} to {df['سال'].max()}")

# ==============================================================================
# Chart 1: Top 15 Organizations by Number of Projects
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# Calculate top 15 organizations by project count
org_counts = df.groupby('نام سازمان').size().sort_values(ascending=True).tail(15)

# Create gradient colors
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(org_counts)))

# Plot horizontal bar chart
bars = ax.barh(range(len(org_counts)), org_counts.values, 
               color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# Add values on bars with Persian numbers
for i, (bar, value) in enumerate(zip(bars, org_counts.values)):
    persian_value = format_number_with_separator(value, use_persian=True)
    ax.text(value + max(org_counts.values) * 0.01, i, persian_value,
            va='center', ha='left', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Configure Y-axis with Persian organization names
persian_orgs = [fix_persian_text(org) for org in org_counts.index]
ax.set_yticks(range(len(org_counts)))
ax.set_yticklabels(persian_orgs, fontsize=11)

# Configure axes
ax.set_xlabel(fix_persian_text('تعداد طرح‌های پژوهشی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('نام دستگاه اجرایی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('۱۵ دستگاه برتر از نظر تعداد طرح‌های پژوهشی'), 
             fontsize=16, fontweight='bold', pad=25)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
ax.set_axisbelow(True)

# Styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 1 saved: {output_path}")

# ==============================================================================
# Chart 2: Top 15 Organizations by Budget
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# Calculate top 15 organizations by budget (in billion Rials)
org_budget = (df.groupby('نام سازمان')['اعتبار'].sum() / 1000).sort_values(ascending=True).tail(15)

# Create gradient colors
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(org_budget)))

# Plot horizontal bar chart
bars = ax.barh(range(len(org_budget)), org_budget.values, 
               color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# Add values on bars with Persian numbers
for i, (bar, value) in enumerate(zip(bars, org_budget.values)):
    persian_value = format_number_with_separator(value, use_persian=True)
    ax.text(value + max(org_budget.values) * 0.01, i, persian_value,
            va='center', ha='left', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Configure Y-axis with Persian organization names
persian_orgs = [fix_persian_text(org) for org in org_budget.index]
ax.set_yticks(range(len(org_budget)))
ax.set_yticklabels(persian_orgs, fontsize=11)

# Configure axes
ax.set_xlabel(fix_persian_text('حجم اعتبارات (میلیارد ریال)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('نام دستگاه اجرایی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('۱۵ دستگاه برتر از نظر حجم اعتبارات'), 
             fontsize=16, fontweight='bold', pad=25)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
ax.set_axisbelow(True)

# Styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 2 saved: {output_path}")

# ==============================================================================
# Chart 3: Pareto Chart - Concentration of Projects
# ==============================================================================

fig, ax1 = plt.subplots(figsize=(16, 9))

# Calculate project counts and sort
org_counts_pareto = df.groupby('نام سازمان').size().sort_values(ascending=False)

# Calculate cumulative percentage
cumulative_pct = (org_counts_pareto.cumsum() / org_counts_pareto.sum()) * 100

# Take top 20 for better visualization
top_n = 20
org_counts_pareto = org_counts_pareto.head(top_n)
cumulative_pct = cumulative_pct.head(top_n)

# First axis: Bar chart (project counts)
color1 = '#3498DB'
x_pos = np.arange(len(org_counts_pareto))
bars = ax1.bar(x_pos, org_counts_pareto.values, 
               color=color1, alpha=0.7, edgecolor='black', linewidth=1.2)

ax1.set_xlabel(fix_persian_text('دستگاه‌های اجرایی'), 
               fontsize=14, fontweight='bold', labelpad=10)
ax1.set_ylabel(fix_persian_text('تعداد طرح‌ها'), 
               fontsize=14, fontweight='bold', color=color1, labelpad=10)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)

# Format Y-axis with Persian numbers
def format_y_pareto(x, p):
    return format_number_with_separator(int(x), use_persian=True)

ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_y_pareto))

# Second axis: Line chart (cumulative percentage)
ax2 = ax1.twinx()
color2 = '#E74C3C'
line = ax2.plot(x_pos, cumulative_pct.values, 
                color=color2, marker='o', linewidth=2.5, markersize=8,
                label=fix_persian_text('درصد تجمعی'), zorder=3)

ax2.set_ylabel(fix_persian_text('درصد تجمعی'), 
               fontsize=14, fontweight='bold', color=color2, labelpad=10)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
ax2.set_ylim([0, 105])

# Add reference line at 80%
ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(len(org_counts_pareto) * 0.5, 82, 
         fix_persian_text('۸۰٪'), 
         fontsize=11, color='gray', fontweight='bold')

# Format second Y-axis with Persian numbers
def format_y2_pareto(x, p):
    return convert_to_persian_number(f'{int(x)}') + '٪'

ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_y2_pareto))

# Configure X-axis
persian_orgs_short = [fix_persian_text(org[:20] + '...' if len(org) > 20 else org) 
                      for org in org_counts_pareto.index]
ax1.set_xticks(x_pos)
ax1.set_xticklabels(persian_orgs_short, rotation=45, ha='right', fontsize=9)

# Title
ax1.set_title(fix_persian_text('نمودار پارتو: تمرکز طرح‌ها در دستگاه‌ها'), 
              fontsize=16, fontweight='bold', pad=25)

# Grid
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax1.set_axisbelow(True)

# Styling
ax1.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_3.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3 saved: {output_path}")

# ==============================================================================
# Chart 4: Heatmap - Organizations × Years
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 12))

# Get top 20 organizations by total project count
top_orgs = df.groupby('نام سازمان').size().sort_values(ascending=False).head(20).index

# Create pivot table: organizations × years
heatmap_data = df[df['نام سازمان'].isin(top_orgs)].pivot_table(
    index='نام سازمان', 
    columns='سال', 
    values='اعتبار',
    aggfunc='count',
    fill_value=0
)

# Sort by total count
heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).index]

# Plot heatmap
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', 
            linewidths=0.5, linecolor='gray', cbar_kws={'label': ''}, 
            ax=ax, vmin=0)

# Configure colorbar with Persian label
cbar = ax.collections[0].colorbar
cbar.set_label(fix_persian_text('تعداد طرح‌ها'), fontsize=13, fontweight='bold')

# Format annotations with Persian numbers
for text in ax.texts:
    current_text = text.get_text()
    if current_text and current_text != '0':
        text.set_text(convert_to_persian_number(current_text))
        text.set_fontsize(9)
        text.set_fontweight('bold')

# Configure axes
persian_orgs = [fix_persian_text(org) for org in heatmap_data.index]
ax.set_yticklabels(persian_orgs, rotation=0, fontsize=10)

persian_years = [convert_to_persian_number(year) for year in heatmap_data.columns]
ax.set_xticklabels(persian_years, rotation=0, fontsize=11, fontweight='bold')

ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('دستگاه اجرایی'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('توزیع زمانی طرح‌های دستگاه‌های برتر'), 
             fontsize=16, fontweight='bold', pad=25)

# Styling
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_4.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 4 saved: {output_path}")

# ==============================================================================
# Chart 5: HHI Trend - Concentration Index
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate HHI for each year
years = sorted(df['سال'].unique())
hhi_values = []

for year in years:
    year_data = df[df['سال'] == year]
    shares = year_data.groupby('نام سازمان')['اعتبار'].sum()
    shares_normalized = shares / shares.sum()
    hhi = herfindahl_index(shares_normalized)
    hhi_values.append(hhi)

# Plot line chart
ax.plot(years, hhi_values, marker='o', linewidth=3, markersize=12,
        color='#E74C3C', markerfacecolor='#E74C3C', 
        markeredgewidth=2, markeredgecolor='white',
        label=fix_persian_text('ضریب تمرکز (HHI)'), zorder=3)

# Add reference lines
ax.axhline(y=0.15, color='#F39C12', linestyle='--', linewidth=2, alpha=0.7,
           label=fix_persian_text('آستانه تمرکز متوسط (۰.۱۵)'))
ax.axhline(y=0.25, color='#C0392B', linestyle='--', linewidth=2, alpha=0.7,
           label=fix_persian_text('آستانه تمرکز بالا (۰.۲۵)'))

# Add values on data points
for x, y in zip(years, hhi_values):
    persian_value = convert_to_persian_number(f'{y:.3f}')
    ax.text(x, y + 0.005, persian_value, 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#E74C3C', alpha=0.8))

# Configure axes
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('شاخص هرفیندال-هیرشمن (HHI)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('روند ضریب تمرکز اعتبارات (HHI)'), 
             fontsize=16, fontweight='bold', pad=25)

# X-axis configuration
ax.set_xticks(years)
persian_years = [convert_to_persian_number(year) for year in years]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Y-axis configuration
def format_y_hhi(x, p):
    return convert_to_persian_number(f'{x:.2f}')

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_hhi))

# Grid and legend
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)

# Styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_5.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 5 saved: {output_path}")

# ==============================================================================
# Chart 6: Donut Chart - Strategic Areas Distribution
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 12))

# Calculate budget distribution by strategic area
area_budget = df.groupby('زمینه اولویت')['اعتبار'].sum().sort_values(ascending=False)

# Take top 10 and group others
top_areas = area_budget.head(10)
other_sum = area_budget[10:].sum()

if other_sum > 0:
    top_areas = pd.concat([top_areas, pd.Series({'سایر': other_sum})])

# Define color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(top_areas)))

# Create donut chart
wedges, texts, autotexts = ax.pie(top_areas.values, 
                                    labels=[fix_persian_text(label) for label in top_areas.index],
                                    autopct='%1.1f%%',
                                    startangle=90, 
                                    colors=colors,
                                    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))

# Format percentage labels with Persian numbers
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
    # Convert percentage to Persian
    current_text = autotext.get_text()
    persian_pct = convert_to_persian_number(current_text.replace('%', '')) + '٪'
    autotext.set_text(persian_pct)

# Format area labels
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')

# Title
ax.set_title(fix_persian_text('توزیع اعتبارات بر اساس حوزه‌های راهبردی'), 
             fontsize=16, fontweight='bold', pad=25)

# Equal aspect ratio ensures circular donut
ax.axis('equal')

# Styling
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_6.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 6 saved: {output_path}")

# ==============================================================================
# Chart 7: Multi-line Trend - Major Strategic Areas
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Identify top 6-7 strategic areas by total budget
top_areas = df.groupby('زمینه اولویت')['اعتبار'].sum().sort_values(ascending=False).head(7).index

# Calculate yearly budget for each top area (in billion Rials)
area_yearly = df[df['زمینه اولویت'].isin(top_areas)].pivot_table(
    index='سال',
    columns='زمینه اولویت',
    values='اعتبار',
    aggfunc='sum',
    fill_value=0
) / 1000

# Define color palette
colors_areas = plt.cm.tab10(np.linspace(0, 1, len(area_yearly.columns)))

# Plot multi-line chart
for i, area in enumerate(area_yearly.columns):
    ax.plot(area_yearly.index, area_yearly[area].values,
            marker='o', linewidth=2.5, markersize=8,
            color=colors_areas[i], label=fix_persian_text(area),
            markeredgewidth=1.5, markeredgecolor='white')

# Configure axes
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('حجم اعتبارات (میلیارد ریال)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('روند تخصیص اعتبارات به حوزه‌های راهبردی'), 
             fontsize=16, fontweight='bold', pad=25)

# X-axis configuration
ax.set_xticks(area_yearly.index)
persian_years = [convert_to_persian_number(year) for year in area_yearly.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Y-axis formatting
def format_y_area(x, p):
    return format_number_with_separator(x, use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_area))

# Grid and legend
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc='upper left', frameon=True, 
          fancybox=True, shadow=True, ncol=1)

# Styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_7.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 7 saved: {output_path}")

# ==============================================================================
# Chart 8: Stacked Area Chart - Relative Share of Strategic Areas
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate percentage share for each area per year
area_yearly_pct = df.pivot_table(
    index='سال',
    columns='زمینه اولویت',
    values='اعتبار',
    aggfunc='sum',
    fill_value=0
)

# Select top 8 areas and group others
top_8_areas = area_yearly_pct.sum().sort_values(ascending=False).head(8).index
area_yearly_pct_top = area_yearly_pct[top_8_areas].copy()

# Add 'Others' column
other_cols = [col for col in area_yearly_pct.columns if col not in top_8_areas]
if other_cols:
    area_yearly_pct_top['سایر'] = area_yearly_pct[other_cols].sum(axis=1)

# Calculate percentages
area_yearly_pct_top = area_yearly_pct_top.div(area_yearly_pct_top.sum(axis=1), axis=0) * 100

# Define color palette
colors_stacked = plt.cm.Spectral(np.linspace(0.1, 0.9, len(area_yearly_pct_top.columns)))

# Plot stacked area chart
ax.stackplot(area_yearly_pct_top.index, 
             [area_yearly_pct_top[col].values for col in area_yearly_pct_top.columns],
             labels=[fix_persian_text(col) for col in area_yearly_pct_top.columns],
             colors=colors_stacked, alpha=0.8, edgecolor='white', linewidth=1.5)

# Configure axes
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('سهم نسبی (درصد)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('سهم نسبی حوزه‌های راهبردی در طول زمان'), 
             fontsize=16, fontweight='bold', pad=25)

# X-axis configuration
ax.set_xticks(area_yearly_pct_top.index)
persian_years = [convert_to_persian_number(year) for year in area_yearly_pct_top.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Y-axis formatting with Persian numbers
def format_y_pct(x, p):
    return convert_to_persian_number(f'{int(x)}') + '٪'

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_pct))
ax.set_ylim([0, 100])

# Grid and legend
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5), 
          frameon=True, fancybox=True, shadow=True)

# Styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_8.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 8 saved: {output_path}")

# ==============================================================================
# Statistical Summary for Chapter 4
# ==============================================================================

print("\n" + "="*70)
print("Statistical Summary for Chapter 4:")
print("="*70)

# Organization statistics
print(f"\n📊 Organization Analysis:")
print(f"   • Total unique organizations: {df['نام سازمان'].nunique()}")
print(f"   • Top organization by projects: {df.groupby('نام سازمان').size().idxmax()}")
print(f"   • Top organization by budget: {df.groupby('نام سازمان')['اعتبار'].sum().idxmax()}")

# HHI statistics
print(f"\n📈 Concentration Analysis:")
for year in sorted(df['سال'].unique()):
    year_data = df[df['سال'] == year]
    shares = year_data.groupby('نام سازمان')['اعتبار'].sum()
    shares_normalized = shares / shares.sum()
    hhi = herfindahl_index(shares_normalized)
    print(f"   • HHI {year}: {hhi:.4f}")

# Strategic area statistics
print(f"\n🎯 Strategic Areas:")
top_5_areas = df.groupby('زمینه اولویت')['اعتبار'].sum().sort_values(ascending=False).head(5)
for area, budget in top_5_areas.items():
    print(f"   • {area}: {budget/1000:,.0f} billion Rials")

print("\n✅ All Chapter 4 charts saved successfully!")