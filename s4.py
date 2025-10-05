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

output_dir = Path.cwd() / 'fig/S4'
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
# Chart 9: Donut Chart - Distribution by Academic Field
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 12))

# محاسبه توزیع بر اساس گروه عمده تحصیلی
field_counts = df.groupby('گروه عمده تحصیلی').size().sort_values(ascending=False)

# گروه‌بندی موارد کوچک در "سایر" (اختیاری)
top_fields = field_counts.head(8)
other_sum = field_counts[8:].sum()

if other_sum > 0:
    top_fields = pd.concat([top_fields, pd.Series({'سایر': other_sum})])

# پالت رنگی
colors = plt.cm.Paired(np.linspace(0, 1, len(top_fields)))

# رسم Donut Chart
wedges, texts, autotexts = ax.pie(top_fields.values, 
                                    labels=[fix_persian_text(label) for label in top_fields.index],
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

# فرمت برچسب‌ها
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')

ax.set_title(fix_persian_text('توزیع طرح‌های پژوهشی بر اساس گروه عمده تحصیلی (۱۳۹۸-۱۴۰۳)'), 
             fontsize=16, fontweight='bold', pad=25)
ax.axis('equal')
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_9.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 9 saved: {output_path}")

# ==============================================================================
# Chart 10: Multi-line Trend - Academic Fields Over Time
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# شناسایی 5-6 گروه تحصیلی برتر
top_fields = df.groupby('گروه عمده تحصیلی')['اعتبار'].sum().sort_values(ascending=False).head(6).index

# محاسبه اعتبار سالانه برای هر گروه (میلیارد ریال)
field_yearly = df[df['گروه عمده تحصیلی'].isin(top_fields)].pivot_table(
    index='سال',
    columns='گروه عمده تحصیلی',
    values='اعتبار',
    aggfunc='sum',
    fill_value=0
) / 1000

# پالت رنگی
colors_fields = plt.cm.tab10(np.linspace(0, 1, len(field_yearly.columns)))

# رسم نمودار چندخطی
for i, field in enumerate(field_yearly.columns):
    ax.plot(field_yearly.index, field_yearly[field].values,
            marker='o', linewidth=2.5, markersize=8,
            color=colors_fields[i], label=fix_persian_text(field),
            markeredgewidth=1.5, markeredgecolor='white')

ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('حجم اعتبارات (میلیارد ریال)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('روند تخصیص اعتبارات به گروه‌های عمده تحصیلی'), 
             fontsize=16, fontweight='bold', pad=25)

# تنظیم محور X
ax.set_xticks(field_yearly.index)
persian_years = [convert_to_persian_number(year) for year in field_yearly.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# فرمت محور Y
def format_y_field(x, p):
    return format_number_with_separator(x, use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_field))

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc='upper left', frameon=True, 
          fancybox=True, shadow=True, ncol=1)

ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_10.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 10 saved: {output_path}")

# ==============================================================================
# Chart 11: Horizontal Bar - Distribution by Specialized Commission
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# محاسبه توزیع بر اساس کمیسیون تخصصی
commission_counts = df.groupby('کمیسیون تخصصی').size().sort_values(ascending=True).tail(15)

# گرادیانت رنگی
colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(commission_counts)))

# رسم نمودار ستونی افقی
bars = ax.barh(range(len(commission_counts)), commission_counts.values, 
               color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# افزودن مقادیر
for i, (bar, value) in enumerate(zip(bars, commission_counts.values)):
    persian_value = format_number_with_separator(value, use_persian=True)
    ax.text(value + max(commission_counts.values) * 0.01, i, persian_value,
            va='center', ha='left', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# برچسب‌های فارسی
persian_commissions = [fix_persian_text(comm) for comm in commission_counts.index]
ax.set_yticks(range(len(commission_counts)))
ax.set_yticklabels(persian_commissions, fontsize=11)

ax.set_xlabel(fix_persian_text('تعداد طرح‌های پژوهشی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('کمیسیون تخصصی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('توزیع طرح‌های پژوهشی بر اساس کمیسیون‌های تخصصی شورای عالی عتف'), 
             fontsize=16, fontweight='bold', pad=25)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
ax.set_axisbelow(True)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()
output_path = output_dir / 'chart_4_11.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 11 saved: {output_path}")

# ==============================================================================
# Chart 12: Heatmap - Commission × Year
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 12))

# شناسایی کمیسیون‌های برتر
top_commissions = df.groupby('کمیسیون تخصصی').size().sort_values(ascending=False).head(15).index

# ایجاد جدول محوری
heatmap_commission = df[df['کمیسیون تخصصی'].isin(top_commissions)].pivot_table(
    index='کمیسیون تخصصی', 
    columns='سال', 
    values='اعتبار',
    aggfunc='count',
    fill_value=0
)

# مرتب‌سازی بر اساس مجموع
heatmap_commission = heatmap_commission.loc[heatmap_commission.sum(axis=1).sort_values(ascending=False).index]

# رسم Heatmap
sns.heatmap(heatmap_commission, annot=True, fmt='g', cmap='YlGnBu', 
            linewidths=0.5, linecolor='gray', cbar_kws={'label': ''}, 
            ax=ax, vmin=0)

# تنظیم colorbar
cbar = ax.collections[0].colorbar
cbar.set_label(fix_persian_text('تعداد طرح‌ها'), fontsize=13, fontweight='bold')

# فارسی‌سازی اعداد
for text in ax.texts:
    current_text = text.get_text()
    if current_text and current_text != '0':
        text.set_text(convert_to_persian_number(current_text))
        text.set_fontsize(9)
        text.set_fontweight('bold')

# برچسب‌ها
persian_commissions = [fix_persian_text(comm) for comm in heatmap_commission.index]
ax.set_yticklabels(persian_commissions, rotation=0, fontsize=10)

persian_years = [convert_to_persian_number(year) for year in heatmap_commission.columns]
ax.set_xticklabels(persian_years, rotation=0, fontsize=11, fontweight='bold')

ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('کمیسیون تخصصی'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('توزیع زمانی فعالیت کمیسیون‌های تخصصی'), 
             fontsize=16, fontweight='bold', pad=25)

fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_12.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 12 saved: {output_path}")

# ==============================================================================
# Chart 13: Radar Chart - Priority Areas Profile
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# انتخاب ۱۰ زمینه اولویت برتر
top_priorities = df.groupby('زمینه اولویت')['اعتبار'].sum().sort_values(ascending=False).head(10)

# نرمال‌سازی به بازه ۰-۱۰۰
values_normalized = (top_priorities.values / top_priorities.values.max()) * 100

# تکمیل دایره
categories = [fix_persian_text(cat) for cat in top_priorities.index]
values_list = list(values_normalized) + [values_normalized[0]]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# رسم Radar Chart
ax.plot(angles, values_list, 'o-', linewidth=2.5, 
       color='#E74C3C', markersize=8, label=fix_persian_text('سهم نرمال‌شده'))
ax.fill(angles, values_list, alpha=0.25, color='#E74C3C')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels([convert_to_persian_number(str(i)) for i in [20, 40, 60, 80, 100]], fontsize=10)

ax.set_title(fix_persian_text('پروفایل زمینه‌های اولویت پژوهشی ماده ۵۶ (۱۳۹۸-۱۴۰۳)'), 
            size=16, weight='bold', pad=30)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_13.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 13 saved: {output_path}")

# ==============================================================================
# Chart 14: Multi-layer Radar Chart - Comparison Across Years
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

# انتخاب ۱۰ زمینه برتر از کل داده
top_priorities_all = df.groupby('زمینه اولویت')['اعتبار'].sum().sort_values(ascending=False).head(10).index

# محاسبه برای سه سال مختلف
years_to_compare = [1398, 1401, 1403]
colors_years = {'1398': '#3498DB', '1401': '#2ECC71', '1403': '#E74C3C'}
alphas = {'1398': 0.2, '1401': 0.2, '1403': 0.3}

categories = [fix_persian_text(cat) for cat in top_priorities_all]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for year in years_to_compare:
    df_year = df[df['سال'] == year]
    priorities_year = df_year.groupby('زمینه اولویت')['اعتبار'].sum()
    
    # مقادیر برای هر زمینه
    values = []
    for priority in top_priorities_all:
        val = priorities_year.get(priority, 0)
        values.append(val)
    
    # نرمال‌سازی
    max_val = max(values) if max(values) > 0 else 1
    values_norm = [(v / max_val) * 100 for v in values]
    values_norm += values_norm[:1]
    
    # رسم
    ax.plot(angles, values_norm, 'o-', linewidth=2.5, markersize=7,
           color=colors_years[str(year)], 
           label=fix_persian_text(f'سال {convert_to_persian_number(str(year))}'))
    ax.fill(angles, values_norm, alpha=alphas[str(year)], color=colors_years[str(year)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels([convert_to_persian_number(str(i)) for i in [20, 40, 60, 80, 100]], fontsize=9)

ax.set_title(fix_persian_text('تغییرات الگوی اولویت‌بندی در طول زمان'), 
            size=16, weight='bold', pad=35)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=12, frameon=True)
fig.patch.set_facecolor('white')

plt.tight_layout()
output_path = output_dir / 'chart_4_14.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 14 saved: {output_path}")

# ==============================================================================
# Statistical Summary for Chapter 4
# ==============================================================================

print("\n" + "="*70)
print("Additional Statistical Summary:")
print("="*70)

print(f"\n📚 Academic Fields Analysis:")
field_dist = df.groupby('گروه عمده تحصیلی').size()
for field, count in field_dist.sort_values(ascending=False).head(5).items():
    pct = (count / len(df)) * 100
    print(f"   • {field}: {count} طرح ({pct:.1f}%)")

print(f"\n🔬 Specialized Commissions:")
comm_dist = df.groupby('کمیسیون تخصصی').size()
print(f"   • تعداد کل کمیسیون‌ها: {df['کمیسیون تخصصی'].nunique()}")
top_comm = comm_dist.idxmax()
print(f"   • کمیسیون برتر: {top_comm} ({comm_dist.max()} طرح)")

print(f"\n🎯 Priority Areas:")
priority_dist = df.groupby('زمینه اولویت')['اعتبار'].sum() / 1000
print(f"   • تعداد کل زمینه‌ها: {df['زمینه اولویت'].nunique()}")
for area, budget in priority_dist.sort_values(ascending=False).head(3).items():
    print(f"   • {area}: {budget:,.0f} میلیارد ریال")

print("\n✅ All additional charts (9-14) saved successfully!")

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