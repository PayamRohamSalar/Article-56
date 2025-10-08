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
# Helper function to convert Persian text to a displayable format
# ==============================================================================

def fix_persian_text(text):
    """
    تبدیل متن فارسی/عربی به فرمت قابل نمایش در matplotlib
    
    Parameters:
    -----------
    text : str
        متن فارسی یا عربی
        
    Returns:
    --------
    str
        متن اصلاح‌شده برای نمایش صحیح از راست به چپ
    """
    if text is None or str(text).strip() == '':
        return ''
    
    try:
        # اصلاح شکل حروف (Reshape)
        reshaped_text = arabic_reshaper.reshape(str(text))
        # تبدیل به فرمت راست به چپ (Bidi)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        print(f"Warning: Could not reshape text '{text}': {e}")
        return str(text)

# ==============================================================================
# Helper function to convert English numbers to Persian
# ==============================================================================

def convert_to_persian_number(number):
    """
    تبدیل اعداد انگلیسی به اعداد فارسی
    
    Parameters:
    -----------
    number : int, float, str
        عدد یا رشته شامل اعداد انگلیسی
        
    Returns:
    --------
    str
        رشته شامل اعداد فارسی
    """
    # جدول تبدیل اعداد انگلیسی به فارسی
    english_digits = '0123456789'
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    
    # ایجاد جدول ترجمه
    translation_table = str.maketrans(english_digits, persian_digits)
    
    # تبدیل عدد به رشته و سپس به فارسی
    return str(number).translate(translation_table)

def format_number_with_separator(number, use_persian=True):
    """
    قالب‌بندی اعداد با جداکننده هزارگان و تبدیل اختیاری به فارسی
    
    Parameters:
    -----------
    number : int, float
        عدد برای قالب‌بندی
    use_persian : bool
        آیا اعداد به فارسی تبدیل شوند؟ (پیش‌فرض: True)
        
    Returns:
    --------
    str
        عدد قالب‌بندی‌شده
    """
    # قالب‌بندی با جداکننده هزارگان
    formatted = f'{number:,.0f}' if isinstance(number, (int, float)) else str(number)
    
    # تبدیل به اعداد فارسی در صورت نیاز
    if use_persian:
        return convert_to_persian_number(formatted)
    return formatted

# ==============================================================================
# Setup Output Directory
# ==============================================================================

# ایجاد فولدر fig در صورت عدم وجود
output_dir = Path.cwd() / 'fig/S3'
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# ==============================================================================
# Load Data
# ==============================================================================

base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Useful_Source_Data.xlsx'
if not data_file.exists():
    raise FileNotFoundError(f"Expected data file not found: {data_file}")

df = pd.read_excel(data_file)

# Data cleaning and preparation
df['سال'] = df['سال'].astype(int)
df['اعتبار'] = pd.to_numeric(df['اعتبار'], errors='coerce')

# Remove null values in budget column
df = df[df['اعتبار'].notna()]

print(f"Total records: {len(df):,}")
print(f"Year range: {df['سال'].min()} to {df['سال'].max()}")

# ==============================================================================
# Chart 1: Trend of Number of Projects (1398-1403)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate yearly project counts
yearly_counts = df.groupby('سال').size().sort_index()

# Plot line chart with enhanced styling
ax.plot(yearly_counts.index, yearly_counts.values, 
        marker='o', linewidth=3, markersize=12, 
        color='#2E86AB', markerfacecolor='#2E86AB',
        markeredgewidth=2, markeredgecolor='white',
        label=fix_persian_text('تعداد طرح‌ها'), zorder=3)

# Add values on data points با اعداد فارسی
for x, y in zip(yearly_counts.index, yearly_counts.values):
    # تبدیل عدد به فارسی با جداکننده هزارگان
    persian_number = format_number_with_separator(y, use_persian=True)
    
    ax.text(x, y + max(yearly_counts.values) * 0.02, 
            persian_number, 
            ha='center', va='bottom', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#2E86AB', alpha=0.8))

# Axis labels and title با استفاده از تابع fix_persian_text
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پژوهشی'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('روند تعداد طرح‌های پژوهشی ماده ۵۶ (۱۳۹۸-۱۴۰۳)'), 
             fontsize=16, fontweight='bold', pad=25)

# Enhanced grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)

# X-axis configuration با اعداد فارسی
ax.set_xticks(yearly_counts.index)
persian_years = [convert_to_persian_number(year) for year in yearly_counts.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Y-axis formatting با اعداد فارسی
def format_y_axis(x, p):
    """فرمت محور Y با اعداد فارسی"""
    return format_number_with_separator(int(x), use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))

# Add background color
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# اضافه کردن legend با متن فارسی اصلاح‌شده
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

plt.tight_layout()

# ذخیره در فولدر fig
output_path = output_dir / 'chart_3_1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 1 saved: {output_path}")

# ==============================================================================
# Chart 2: Annual Budget Trends
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate total budget per year (convert million Rials to billion Rials)
yearly_budget = df.groupby('سال')['اعتبار'].sum() / 1000

# Define colors with emphasis on year 1403
colors = ['#3498DB' if year != 1403 else '#E74C3C' 
          for year in yearly_budget.index]

# Plot bar chart
bars = ax.bar(yearly_budget.index, yearly_budget.values, 
              color=colors, edgecolor='black', linewidth=1.5, 
              alpha=0.85, width=0.6)

# Add values on bars with Persian numbers and thousand separators
for bar in bars:
    height = bar.get_height()
    # Convert number to Persian format with thousand separator
    persian_value = format_number_with_separator(height, use_persian=True)
    
    ax.text(bar.get_x() + bar.get_width()/2., height,
            persian_value,
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.7))

# Axis labels and title with Persian RTL text
ax.set_xlabel(fix_persian_text('سال'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('حجم اعتبارات (میلیارد ریال)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('حجم اعتبارات تخصیص‌یافته به تفکیک سال'), 
             fontsize=16, fontweight='bold', pad=25)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# Y-axis formatting with Persian numbers and comma separator
def format_y_axis_budget(x, p):
    """Format Y-axis with Persian numbers and thousand separator"""
    return format_number_with_separator(x, use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis_budget))

# X-axis configuration with Persian year numbers
ax.set_xticks(yearly_budget.index)
persian_years = [convert_to_persian_number(year) for year in yearly_budget.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Apply background styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Configure border spines
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 2 saved: {output_path}")

# ==============================================================================
# Chart 3: Average Budget per Project (Combination Chart)
# ==============================================================================

fig, ax1 = plt.subplots(figsize=(14, 8))

# Calculate average budget and project counts per year
yearly_avg = df.groupby('سال')['اعتبار'].mean()
yearly_counts = df.groupby('سال').size()

# First axis: Average budget (bar chart)
color1 = '#F39C12'
bars = ax1.bar(yearly_avg.index, yearly_avg.values, 
               color=color1, alpha=0.75, 
               label=fix_persian_text('میانگین اعتبار'), 
               edgecolor='black', linewidth=1.5, width=0.5)

# Configure first Y-axis (average budget)
ax1.set_xlabel(fix_persian_text('سال'), 
               fontsize=14, fontweight='bold', labelpad=10)
ax1.set_ylabel(fix_persian_text('میانگین اعتبار هر طرح (میلیون ریال)'), 
               fontsize=14, fontweight='bold', color=color1, labelpad=10)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)

# Format first Y-axis with Persian numbers
def format_y1_axis(x, p):
    """Format first Y-axis with Persian numbers and thousand separator"""
    return format_number_with_separator(x, use_persian=True)

ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_y1_axis))

# Second axis: Number of projects (line chart)
ax2 = ax1.twinx()
color2 = '#2E86AB'
line = ax2.plot(yearly_counts.index, yearly_counts.values, 
                marker='o', linewidth=3, markersize=12,
                color=color2, label=fix_persian_text('تعداد طرح‌ها'),
                markerfacecolor=color2, markeredgewidth=2, 
                markeredgecolor='white', zorder=3)

# Configure second Y-axis (project count)
ax2.set_ylabel(fix_persian_text('تعداد کل طرح‌ها'), 
               fontsize=14, fontweight='bold', 
               color=color2, labelpad=10)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)

# Format second Y-axis with Persian numbers
def format_y2_axis(x, p):
    """Format second Y-axis with Persian numbers"""
    return format_number_with_separator(int(x), use_persian=True)

ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_y2_axis))

# Chart title with Persian RTL text
ax1.set_title(fix_persian_text('میانگین اعتبار هر طرح و تعداد کل طرح‌ها'), 
              fontsize=16, fontweight='bold', pad=25)

# Add values on bars with Persian numbers
for bar in bars:
    height = bar.get_height()
    persian_value = format_number_with_separator(height, use_persian=True)
    
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             persian_value,
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Add values on line with Persian numbers
for x, y in zip(yearly_counts.index, yearly_counts.values):
    persian_count = format_number_with_separator(y, use_persian=True)
    
    ax2.text(x, y + max(yearly_counts.values) * 0.02, 
             persian_count, ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color=color2,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Grid configuration
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax1.set_axisbelow(True)

# Combined legend with Persian labels
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', fontsize=12, frameon=True, 
          fancybox=True, shadow=True)

# X-axis configuration with Persian year numbers
ax1.set_xticks(yearly_avg.index)
persian_years = [convert_to_persian_number(year) for year in yearly_avg.index]
ax1.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Apply background styling
ax1.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_3.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3 saved: {output_path}")

# ==============================================================================
# Chart 4: Distribution of Project Types (Stacked Bar)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate distribution of project types per year
type_counts = df.groupby(['سال', 'نوع طرح']).size().unstack(fill_value=0)

# Order columns according to project type
type_order = ['بنیادی', 'کاربردی', 'توسعه ای']
type_counts = type_counts[[col for col in type_order if col in type_counts.columns]]

# Define color palette for each project type
colors_palette = {
    'بنیادی': '#9B59B6', 
    'کاربردی': '#3498DB', 
    'توسعه ای': '#2ECC71'
}
colors = [colors_palette.get(col, '#95A5A6') for col in type_counts.columns]

# Plot stacked bar chart
type_counts.plot(kind='bar', stacked=True, ax=ax, color=colors, 
                 edgecolor='black', linewidth=1.2, alpha=0.85, width=0.7)

# Configure axes with Persian RTL text
ax.set_xlabel(fix_persian_text('سال'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پژوهشی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('توزیع طرح‌ها بر اساس نوع (۱۳۹۸-۱۴۰۳)'), 
             fontsize=16, fontweight='bold', pad=25)

# Configure X-axis with Persian year numbers
persian_years = [convert_to_persian_number(year) for year in type_counts.index]
ax.set_xticklabels(persian_years, rotation=0, fontsize=12, fontweight='bold')

# Configure legend with Persian labels
legend_labels = [fix_persian_text(label) for label in type_counts.columns]
ax.legend(legend_labels, title=fix_persian_text('نوع طرح'), 
          fontsize=12, title_fontsize=13, 
          loc='upper left', frameon=True, fancybox=True, shadow=True)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# Add total values on top of each stacked bar with Persian numbers
for i, (idx, row) in enumerate(type_counts.iterrows()):
    total = row.sum()
    persian_total = format_number_with_separator(total, use_persian=True)
    
    ax.text(i, total + max(type_counts.sum(axis=1)) * 0.01, 
            persian_total, 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.7))

# Apply background styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Configure border spines
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_4.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 4 saved: {output_path}")


# ==============================================================================
# Chart 5: Boxplot of Budget Distribution
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for boxplot by year
years_sorted = sorted(df['سال'].unique())
data_for_box = [df[df['سال'] == year]['اعتبار'].values 
                for year in years_sorted]

# Create labels with Persian year numbers
persian_year_labels = [convert_to_persian_number(year) for year in years_sorted]

# Plot boxplot with custom styling
bp = ax.boxplot(data_for_box, labels=persian_year_labels,
                patch_artist=True, notch=True, widths=0.6,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2, color='red'))

# Apply color palette to boxes
colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
for patch, color in zip(bp['boxes'], colors_box[:len(years_sorted)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Set logarithmic scale for Y-axis
ax.set_yscale('log')

# Configure axes with Persian RTL text
ax.set_xlabel(fix_persian_text('سال'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('اعتبار (میلیون ریال - مقیاس لگاریتمی)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('توزیع آماری اعتبارات طرح‌ها به تفکیک سال'), 
             fontsize=16, fontweight='bold', pad=25)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# Format Y-axis with Persian numbers
def format_y_axis_log(x, p):
    """Format Y-axis with Persian numbers for logarithmic scale"""
    return format_number_with_separator(x, use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis_log))
ax.tick_params(axis='both', labelsize=11)

# Apply background styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Configure border spines
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_5.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 5 saved: {output_path}")


# ==============================================================================
# Chart 6: Academic vs Non-Academic Projects Comparison
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate distribution of academic vs non-academic projects
academic_counts = df.groupby(['سال', 'نوع دانشگاهی']).size().unstack(fill_value=0)

# Setup grouped bar chart positions
x = np.arange(len(academic_counts.index))
width = 0.35

# Plot first group (Academic projects)
bars1 = ax.bar(x - width/2, academic_counts['دانشگاهی'], width, 
               label=fix_persian_text('دانشگاهی'), color='#3D5A80', 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Plot second group (Non-academic projects)
bars2 = ax.bar(x + width/2, academic_counts['غیر دانشگاهی'], width, 
               label=fix_persian_text('غیر دانشگاهی'), color='#EE6C4D',
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add values on bars with Persian numbers
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        persian_height = format_number_with_separator(int(height), use_persian=True)
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                persian_height,
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Configure axes with Persian RTL text
ax.set_xlabel(fix_persian_text('سال'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پژوهشی'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('مقایسه طرح‌های دانشگاهی و غیردانشگاهی'), 
             fontsize=16, fontweight='bold', pad=25)

# Configure X-axis with Persian year numbers
ax.set_xticks(x)
persian_years = [convert_to_persian_number(year) for year in academic_counts.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# Configure legend
ax.legend(fontsize=13, loc='upper left', frameon=True, 
          fancybox=True, shadow=True)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# Format Y-axis with Persian numbers
def format_y_axis_count(x, p):
    """Format Y-axis with Persian numbers"""
    return format_number_with_separator(int(x), use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis_count))

# Apply background styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Configure border spines
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_6.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 6 saved: {output_path}")

# ==============================================================================
# Chart 7: Budget Comparison between Academic and Non-Academic Projects (Box Plot)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for boxplot: Academic vs Non-Academic
academic_data = df[df['نوع دانشگاهی'] == 'دانشگاهی']['اعتبار'].values
non_academic_data = df[df['نوع دانشگاهی'] == 'غیر دانشگاهی']['اعتبار'].values

# Combine data for boxplot
data_for_comparison = [academic_data, non_academic_data]

# Create labels with Persian text
labels = [fix_persian_text('دانشگاهی'), fix_persian_text('غیر دانشگاهی')]

# Plot boxplot with custom styling
bp = ax.boxplot(data_for_comparison, labels=labels,
                patch_artist=True, notch=True, widths=0.5,
                boxprops=dict(linewidth=1.8),
                whiskerprops=dict(linewidth=1.8),
                capprops=dict(linewidth=1.8),
                medianprops=dict(linewidth=2.5, color='red'),
                showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='gray', 
                               markersize=4, linestyle='none', alpha=0.3))

# Apply color palette to boxes
colors_box = ['#3D5A80', '#EE6C4D']  # Academic: Blue, Non-Academic: Orange
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')

# Set logarithmic scale for Y-axis
ax.set_yscale('log')

# Configure axes with Persian RTL text
ax.set_xlabel(fix_persian_text('نوع طرح'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('اعتبار (میلیون ریال - مقیاس لگاریتمی)'), 
              fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(fix_persian_text('مقایسه توزیع اعتبارات طرح‌های دانشگاهی و غیردانشگاهی'), 
             fontsize=16, fontweight='bold', pad=25)

# Grid configuration
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# Format Y-axis with Persian numbers
def format_y_axis_budget_comparison(x, p):
    """Format Y-axis with Persian numbers for logarithmic scale"""
    return format_number_with_separator(x, use_persian=True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis_budget_comparison))
ax.tick_params(axis='both', labelsize=12)

# Add statistical annotations with Persian text
# Calculate median, mean, and quartiles
stats_academic = {
    'median': np.median(academic_data),
    'mean': np.mean(academic_data),
    'q1': np.percentile(academic_data, 25),
    'q3': np.percentile(academic_data, 75)
}

stats_non_academic = {
    'median': np.median(non_academic_data),
    'mean': np.mean(non_academic_data),
    'q1': np.percentile(non_academic_data, 25),
    'q3': np.percentile(non_academic_data, 75)
}

# Add text box with statistics (optional - can be commented out if too crowded)
stats_text = f"""{fix_persian_text('دانشگاهی:')}
{fix_persian_text('میانه:')} {format_number_with_separator(stats_academic['median'], use_persian=True)}
{fix_persian_text('میانگین:')} {format_number_with_separator(stats_academic['mean'], use_persian=True)}

{fix_persian_text('غیردانشگاهی:')}
{fix_persian_text('میانه:')} {format_number_with_separator(stats_non_academic['median'], use_persian=True)}
{fix_persian_text('میانگین:')} {format_number_with_separator(stats_non_academic['mean'], use_persian=True)}
"""

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                 edgecolor='#CCCCCC', alpha=0.9, linewidth=1.5))

# Apply background styling
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Configure border spines
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

# Apply tight layout
plt.tight_layout()

# Save chart to 'fig' directory
output_path = output_dir / 'chart_3_7.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 7 saved: {output_path}")


# ==============================================================================
# SECTION 3-4: Accepted Projects Analysis (REVISED)
# ==============================================================================

print("\n" + "="*70)
print("SECTION 3-4: Accepted Projects Analysis")
print("="*70)

# ==============================================================================
# Data Preparation: Filter Accepted Projects Only
# ==============================================================================

# فیلتر کردن فقط طرح‌های پذیرفته‌شده
# توجه: بررسی تمام حالات ممکن نوشتاری "پذیرفته شده"
df_accepted = df[
    (df['وضعیت نهایی'] == 'پذيرفته شده') |  # با ی عربی
    (df['وضعیت نهایی'] == 'پذیرفته شده')    # با ی فارسی
].copy()

# بررسی وجود داده
if len(df_accepted) == 0:
    print("⚠ Warning: No accepted projects found in data!")
    print("Available status values:", df['وضعیت نهایی'].unique())
    print("\n⚠ Skipping Section 3-4 charts due to no data.")
    
    # ایجاد یک DataFrame خالی برای جلوگیری از خطا
    df_accepted = pd.DataFrame(columns=df.columns)
    df_accepted_payment = pd.DataFrame(columns=df.columns)
    
else:
    print(f"\n✓ Accepted projects: {len(df_accepted):,} out of {len(df):,} total projects")
    print(f"✓ Acceptance rate: {(len(df_accepted)/len(df)*100):.1f}%")
    
    # محاسبه نرخ جذب برای طرح‌های پذیرفته‌شده
    df_accepted['نرخ جذب (%)'] = (df_accepted['پرداخت سال جاری'] / df_accepted['اعتبار']) * 100
    df_accepted['نرخ جذب (%)'] = df_accepted['نرخ جذب (%)'].clip(0, 100)
    
    # حذف مقادیر null در ستون پرداخت
    df_accepted_payment = df_accepted[df_accepted['پرداخت سال جاری'].notna()].copy()
    
    print(f"✓ Accepted projects with payment data: {len(df_accepted_payment):,}")

# ==============================================================================
# Chart 3-8: Trend of Accepted Projects (Line Chart)
# ==============================================================================
# بررسی وجود داده
if len(df_accepted) == 0:
    print("⚠ Skipping Chart 3-8: No accepted projects data")
    # ایجاد نمودار خالی با پیام
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.text(0.5, 0.5, fix_persian_text('داده‌ای برای نمایش وجود ندارد'), 
            ha='center', va='center', fontsize=16, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    output_path = output_dir / 'chart_3_8.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
else:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # محاسبه تعداد طرح‌های پذیرفته‌شده در هر سال
    yearly_accepted = df_accepted.groupby('سال').size().sort_index()
    

fig, ax = plt.subplots(figsize=(14, 8))

# محاسبه تعداد طرح‌های پذیرفته‌شده در هر سال
yearly_accepted = df_accepted.groupby('سال').size().sort_index()

# رسم نمودار خطی
ax.plot(
    yearly_accepted.index,
    yearly_accepted.values,
    marker='o',
    linewidth=3.5,
    markersize=14,
    color='#27AE60',
    markerfacecolor='#27AE60',
    markeredgewidth=2,
    markeredgecolor='white',
    label=fix_persian_text('طرح‌های پذیرفته‌شده'),
    zorder=3
)

# افزودن مقادیر روی نقاط
for x, y in zip(yearly_accepted.index, yearly_accepted.values):
    persian_number = format_number_with_separator(y, use_persian=True)
    
    ax.text(
        x, y + max(yearly_accepted.values) * 0.02,
        persian_number,
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#27AE60', alpha=0.8)
    )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پذیرفته‌شده'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('روند تعداد طرح‌های پذیرفته‌شده ماده ۵۶ (۱۳۹۸-۱۴۰۳)'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور X
ax.set_xticks(yearly_accepted.index)
persian_years = [convert_to_persian_number(year) for year in yearly_accepted.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# تنظیم محور Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)

# legend
ax.legend(loc='upper left', fontsize=12, framealpha=0.9, shadow=True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_8.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-8 saved: {output_path}")

# ==============================================================================
# Chart 3-9: Acceptance Rate (Dual Line Chart)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# محاسبه تعداد کل طرح‌ها و طرح‌های پذیرفته‌شده
yearly_total = df.groupby('سال').size().sort_index()
yearly_accepted = df_accepted.groupby('سال').size().reindex(yearly_total.index, fill_value=0)

# محاسبه نسبت پذیرش
acceptance_rate = (yearly_accepted / yearly_total * 100)

# رسم خط اول: کل طرح‌ها
line1 = ax.plot(
    yearly_total.index,
    yearly_total.values,
    marker='o',
    linewidth=3,
    markersize=12,
    color='#3498DB',
    markerfacecolor='#3498DB',
    markeredgewidth=2,
    markeredgecolor='white',
    label=fix_persian_text('کل طرح‌ها'),
    zorder=2
)

# رسم خط دوم: طرح‌های پذیرفته‌شده
line2 = ax.plot(
    yearly_accepted.index,
    yearly_accepted.values,
    marker='s',
    linewidth=3,
    markersize=12,
    color='#27AE60',
    markerfacecolor='#27AE60',
    markeredgewidth=2,
    markeredgecolor='white',
    label=fix_persian_text('طرح‌های پذیرفته‌شده'),
    zorder=3
)

# افزودن درصد پذیرش بر روی هر نقطه
for x, y_total, y_accepted, rate in zip(yearly_total.index, yearly_total.values, yearly_accepted.values, acceptance_rate.values):
    # نمایش درصد
    persian_rate = fix_persian_text(f'{rate:.1f}%')
    
    ax.text(
        x, y_accepted + max(yearly_total.values) * 0.03,
        persian_rate,
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#27AE60', alpha=0.8)
    )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌ها'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('نسبت طرح‌های پذیرفته‌شده به کل طرح‌ها'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور X
ax.set_xticks(yearly_total.index)
persian_years = [convert_to_persian_number(year) for year in yearly_total.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# تنظیم محور Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)

# legend
ax.legend(loc='upper left', fontsize=12, framealpha=0.9, shadow=True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_9.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-9 saved: {output_path}")

# ==============================================================================
# Chart 3-10: Top 10 Commissions - Accepted Projects (Horizontal Bar)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# محاسبه تعداد طرح‌های پذیرفته‌شده به تفکیک کمیسیون
commission_accepted = df_accepted.groupby('کمیسیون تخصصی').size().sort_values(ascending=True).tail(10)

# رسم نمودار ستونی افقی
bars = ax.barh(
    range(len(commission_accepted)),
    commission_accepted.values,
    color=plt.cm.Greens(np.linspace(0.4, 0.9, len(commission_accepted))),
    edgecolor='black',
    linewidth=1.5,
    alpha=0.85
)

# افزودن مقادیر
for i, (bar, value) in enumerate(zip(bars, commission_accepted.values)):
    persian_value = format_number_with_separator(value, use_persian=True)
    
    ax.text(
        value + max(commission_accepted.values) * 0.01,
        bar.get_y() + bar.get_height()/2,
        persian_value,
        ha='left', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('تعداد طرح‌های پذیرفته‌شده'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('کمیسیون تخصصی'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('۱۰ کمیسیون برتر از نظر تعداد طرح‌های پذیرفته‌شده'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم برچسب‌های محور Y
commission_labels = [fix_persian_text(name) for name in commission_accepted.index]
ax.set_yticks(range(len(commission_accepted)))
ax.set_yticklabels(commission_labels, fontsize=11)

# تنظیم محور X
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_10.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-10 saved: {output_path}")

# ==============================================================================
# Chart 3-11: Heatmap - Commissions × Year (Accepted Projects)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# محاسبه توزیع زمانی به تفکیک کمیسیون
commission_year = df_accepted.groupby(['کمیسیون تخصصی', 'سال']).size().unstack(fill_value=0)

# انتخاب ۱۵ کمیسیون برتر
top_commissions = df_accepted.groupby('کمیسیون تخصصی').size().sort_values(ascending=False).head(15).index
commission_year_top = commission_year.loc[top_commissions]

# رسم Heatmap
sns.heatmap(
    commission_year_top,
    annot=True,
    fmt='d',
    cmap='Greens',
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': fix_persian_text('تعداد طرح')},
    ax=ax,
    vmin=0
)

# تنظیم عنوان
ax.set_title(
    fix_persian_text('توزیع زمانی طرح‌های پذیرفته‌شده در کمیسیون‌های تخصصی'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('کمیسیون تخصصی'), fontsize=14, fontweight='bold', labelpad=10)

# تنظیم برچسب‌های محور Y
y_labels = [fix_persian_text(name) for name in commission_year_top.index]
ax.set_yticklabels(y_labels, rotation=0, fontsize=10)

# تنظیم برچسب‌های محور X
x_labels = [convert_to_persian_number(year) for year in commission_year_top.columns]
ax.set_xticklabels(x_labels, rotation=0, fontsize=11, fontweight='bold')

# پس‌زمینه
fig.patch.set_facecolor('white')

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_11.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-11 saved: {output_path}")

# ==============================================================================
# Chart 3-12: Top 15 Provinces - Accepted Projects (Horizontal Bar)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# محاسبه تعداد طرح‌های پذیرفته‌شده به تفکیک استان
province_accepted = df_accepted.groupby('استان اجرا').size().sort_values(ascending=True).tail(15)

# رسم نمودار ستونی افقی
bars = ax.barh(
    range(len(province_accepted)),
    province_accepted.values,
    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(province_accepted))),
    edgecolor='black',
    linewidth=1.5,
    alpha=0.85
)

# افزودن مقادیر
for i, (bar, value) in enumerate(zip(bars, province_accepted.values)):
    persian_value = format_number_with_separator(value, use_persian=True)
    
    ax.text(
        value + max(province_accepted.values) * 0.01,
        bar.get_y() + bar.get_height()/2,
        persian_value,
        ha='left', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('تعداد طرح‌های پذیرفته‌شده'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('استان'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('۱۵ استان برتر از نظر تعداد طرح‌های پذیرفته‌شده'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم برچسب‌های محور Y
province_labels = [fix_persian_text(name) for name in province_accepted.index]
ax.set_yticks(range(len(province_accepted)))
ax.set_yticklabels(province_labels, fontsize=11)

# تنظیم محور X
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_12.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-12 saved: {output_path}")

# ==============================================================================
# Chart 3-13: Geographic Distribution (All Provinces - Bar Chart)
# ==============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

# محاسبه تعداد طرح‌های پذیرفته‌شده برای تمام ۳۱ استان
all_provinces_accepted = df_accepted.groupby('استان اجرا').size().sort_values(ascending=False)

# تنظیم رنگ‌ها بر اساس مقدار (Gradient)
colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(all_provinces_accepted)))

# رسم نمودار ستونی
bars = ax.bar(
    range(len(all_provinces_accepted)),
    all_provinces_accepted.values,
    color=colors,
    edgecolor='black',
    linewidth=1.2,
    alpha=0.85
)

# افزودن مقادیر روی ستون‌ها
for i, (bar, value) in enumerate(zip(bars, all_provinces_accepted.values)):
    if value > 0:  # فقط برای مقادیر غیرصفر
        persian_value = format_number_with_separator(value, use_persian=True)
        
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value,
            persian_value,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            rotation=0
        )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('استان'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پذیرفته‌شده'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('توزیع جغرافیایی طرح‌های پذیرفته‌شده در سطح کشور'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم برچسب‌های محور X
province_labels = [fix_persian_text(name) for name in all_provinces_accepted.index]
ax.set_xticks(range(len(all_provinces_accepted)))
ax.set_xticklabels(province_labels, rotation=90, fontsize=9, ha='right')

# تنظیم محور Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_13.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-13 saved: {output_path}")

# ==============================================================================
# Chart 3-14: Top 10 Strategic Areas (Donut Chart)
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

# محاسبه توزیع حوزه‌های راهبردی
priority_accepted = df_accepted.groupby('زمینه اولویت').size().sort_values(ascending=False).head(10)

# تعریف رنگ‌ها
colors_priority = plt.cm.Set3(np.linspace(0, 1, len(priority_accepted)))

# رسم نمودار Donut
wedges, texts, autotexts = ax.pie(
    priority_accepted.values,
    labels=[fix_persian_text(name) for name in priority_accepted.index],
    colors=colors_priority,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85,
    explode=[0.05 if i == 0 else 0 for i in range(len(priority_accepted))],
    shadow=True,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

# تنظیم استایل متن‌ها
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# افزودن دایره مرکزی
centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=2, edgecolor='#CCCCCC')
fig.gca().add_artist(centre_circle)

# عنوان
ax.set_title(
    fix_persian_text('توزیع طرح‌های پذیرفته‌شده بر اساس حوزه‌های راهبردی (زمینه اولویت)'),
    fontsize=16, fontweight='bold', pad=25
)

# افزودن جدول آماری
table_data = []
for priority, count in priority_accepted.items():
    percentage = (count / priority_accepted.sum()) * 100
    table_data.append([
        fix_persian_text(priority[:30] + '...' if len(priority) > 30 else priority),
        format_number_with_separator(count, use_persian=True),
        fix_persian_text(f'{percentage:.1f}%')
    ])

# رسم جدول
table = ax.table(
    cellText=table_data,
    colLabels=[fix_persian_text('حوزه راهبردی'), fix_persian_text('تعداد'), fix_persian_text('درصد')],
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.45, 1.0, 0.35],
    colColours=['#E8E8E8']*3
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# استایل‌دهی جدول
for key, cell in table.get_celld().items():
    cell.set_linewidth(1.5)
    cell.set_edgecolor('#CCCCCC')
    if key[0] == 0:
        cell.set_facecolor('#D5DBDB')
        cell.set_text_props(weight='bold')

# پس‌زمینه
fig.patch.set_facecolor('white')

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_14.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-14 saved: {output_path}")

# ==============================================================================
# Chart 3-15: Trend of Top Strategic Areas (Stacked Area Chart)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# انتخاب ۶-۷ حوزه برتر
top_priorities = df_accepted.groupby('زمینه اولویت').size().sort_values(ascending=False).head(7).index

# محاسبه توزیع زمانی
priority_year = df_accepted[df_accepted['زمینه اولویت'].isin(top_priorities)].groupby(['سال', 'زمینه اولویت']).size().unstack(fill_value=0)

# رسم Stacked Area Chart
ax.stackplot(
    priority_year.index,
    [priority_year[col].values for col in priority_year.columns],
    labels=[fix_persian_text(col[:25] + '...' if len(col) > 25 else col) for col in priority_year.columns],
    alpha=0.8,
    edgecolor='white',
    linewidth=1.5
)

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('تعداد طرح‌های پذیرفته‌شده'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('روند طرح‌های پذیرفته‌شده به تفکیک حوزه‌های راهبردی'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور X
ax.set_xticks(priority_year.index)
persian_years = [convert_to_persian_number(year) for year in priority_year.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# تنظیم محور Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(int(x), use_persian=True)))

# legend
ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True, ncol=1)

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_15.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-15 saved: {output_path}")

# ==============================================================================
# Chart 3-16: Budget vs Payment (Accepted Projects)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# محاسبه اعتبار مصوب و پرداخت واقعی طرح‌های پذیرفته‌شده (میلیارد ریال)
yearly_approved_accepted = df_accepted_payment.groupby('سال')['اعتبار'].sum() / 1000
yearly_payment_accepted = df_accepted_payment.groupby('سال')['پرداخت سال جاری'].sum() / 1000

# تنظیم موقعیت ستون‌ها
x = np.arange(len(yearly_approved_accepted.index))
width = 0.35

# رسم ستون‌ها
bars1 = ax.bar(
    x - width/2,
    yearly_approved_accepted.values,
    width,
    label=fix_persian_text('اعتبار مصوب'),
    color='#3498DB',
    edgecolor='black',
    linewidth=1.5,
    alpha=0.85
)

bars2 = ax.bar(
    x + width/2,
    yearly_payment_accepted.values,
    width,
    label=fix_persian_text('پرداخت واقعی'),
    color='#27AE60',
    edgecolor='black',
    linewidth=1.5,
    alpha=0.85
)

# افزودن مقادیر
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        persian_height = format_number_with_separator(height, use_persian=True)
        
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            persian_height,
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('اعتبار (میلیارد ریال)'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('اعتبار مصوب و پرداخت واقعی طرح‌های پذیرفته‌شده'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور X
ax.set_xticks(x)
persian_years = [convert_to_persian_number(year) for year in yearly_approved_accepted.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# تنظیم محور Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number_with_separator(x, use_persian=True)))

# legend
ax.legend(fontsize=13, loc='upper left', frameon=True, fancybox=True, shadow=True)

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_16.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-16 saved: {output_path}")

# ==============================================================================
# Chart 3-17: Absorption Rate - Accepted Projects (Line Chart)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# محاسبه نرخ جذب کلی هر سال برای طرح‌های پذیرفته‌شده
yearly_absorption_accepted = (yearly_payment_accepted / yearly_approved_accepted) * 100

# رسم نمودار خطی
ax.plot(
    yearly_absorption_accepted.index,
    yearly_absorption_accepted.values,
    marker='o',
    linewidth=3.5,
    markersize=14,
    color='#E67E22',
    markerfacecolor='#E67E22',
    markeredgewidth=2,
    markeredgecolor='white',
    label=fix_persian_text('نرخ جذب'),
    zorder=3
)

# خط راهنما: هدف ۸۰٪
ax.axhline(y=80, color='#27AE60', linestyle='--', linewidth=2.5, 
           label=fix_persian_text('هدف: ۸۰٪'), alpha=0.7, zorder=2)

# افزودن مقادیر روی نقاط
for x, y in zip(yearly_absorption_accepted.index, yearly_absorption_accepted.values):
    persian_rate = fix_persian_text(f'{y:.1f}%')
    
    ax.text(
        x, y + 3,
        persian_rate,
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        color='#E67E22',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#E67E22', alpha=0.8)
    )

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('نرخ جذب (%)'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('نرخ جذب اعتبارات طرح‌های پذیرفته‌شده'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور X
ax.set_xticks(yearly_absorption_accepted.index)
persian_years = [convert_to_persian_number(year) for year in yearly_absorption_accepted.index]
ax.set_xticklabels(persian_years, fontsize=12, fontweight='bold')

# تنظیم محور Y
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fix_persian_text(f'{int(x)}%')))

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)

# legend
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, shadow=True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_17.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-17 saved: {output_path}")

# ==============================================================================
# Chart 3-18: Boxplot - Absorption Rate Distribution (Accepted Projects)
# ==============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# آماده‌سازی داده برای boxplot
years_sorted = sorted(df_accepted_payment['سال'].unique())
data_for_box = [df_accepted_payment[df_accepted_payment['سال'] == year]['نرخ جذب (%)'].values 
                for year in years_sorted]

# برچسب‌های فارسی
persian_year_labels = [convert_to_persian_number(year) for year in years_sorted]

# رسم boxplot
bp = ax.boxplot(
    data_for_box,
    labels=persian_year_labels,
    patch_artist=True,
    notch=True,
    widths=0.6,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=2.5, color='red'),
    showfliers=True,
    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none', alpha=0.3)
)

# رنگ‌آمیزی
colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
for patch, color in zip(bp['boxes'], colors_box[:len(years_sorted)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')

# تنظیم محورها
ax.set_xlabel(fix_persian_text('سال'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(fix_persian_text('نرخ جذب (%)'), fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(
    fix_persian_text('توزیع آماری نرخ جذب در طرح‌های پذیرفته‌شده'),
    fontsize=16, fontweight='bold', pad=25
)

# تنظیم محور Y
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fix_persian_text(f'{int(x)}%')))
ax.tick_params(axis='both', labelsize=12)

# grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_axisbelow(True)

# پس‌زمینه
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# حاشیه
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

# ذخیره
output_path = output_dir / 'chart_3_18.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Chart 3-18 saved: {output_path}")

# ==============================================================================
# Statistical Summary for Section 3-4 (Accepted Projects)
# ==============================================================================

print("\n" + "="*70)
print("Section 3-4: Statistical Summary - Accepted Projects")
print("="*70)

# آمار کلی
total_accepted = len(df_accepted)
total_all = len(df)
overall_acceptance = (total_accepted / total_all) * 100

print(f"\n📊 Overall Statistics:")
print(f"   • Total accepted projects: {total_accepted:,}")
print(f"   • Total all projects: {total_all:,}")
print(f"   • Overall acceptance rate: {overall_acceptance:.1f}%")

# آمار سالانه
print(f"\n📈 Annual Accepted Projects:")
for year in sorted(df_accepted['سال'].unique()):
    year_accepted = len(df_accepted[df_accepted['سال'] == year])
    year_total = len(df[df['سال'] == year])
    year_rate = (year_accepted / year_total) * 100 if year_total > 0 else 0
    
    print(f"   • {year}: {year_accepted:,} projects ({year_rate:.1f}% acceptance rate)")

# آمار کمیسیون‌ها
print(f"\n🏛 Top 5 Commissions (Accepted Projects):")
top_5_commissions = df_accepted.groupby('کمیسیون تخصصی').size().sort_values(ascending=False).head(5)
for commission, count in top_5_commissions.items():
    percentage = (count / total_accepted) * 100
    print(f"   • {commission}: {count:,} projects ({percentage:.1f}%)")

# آمار استانی
print(f"\n🗺 Top 5 Provinces (Accepted Projects):")
top_5_provinces = df_accepted.groupby('استان اجرا').size().sort_values(ascending=False).head(5)
for province, count in top_5_provinces.items():
    percentage = (count / total_accepted) * 100
    print(f"   • {province}: {count:,} projects ({percentage:.1f}%)")

# آمار حوزه‌های راهبردی
print(f"\n🎯 Top 5 Strategic Areas (Accepted Projects):")
top_5_priorities = df_accepted.groupby('زمینه اولویت').size().sort_values(ascending=False).head(5)
for priority, count in top_5_priorities.items():
    percentage = (count / total_accepted) * 100
    priority_short = priority[:40] + '...' if len(priority) > 40 else priority
    print(f"   • {priority_short}: {count:,} projects ({percentage:.1f}%)")

# آمار مالی
if len(df_accepted_payment) > 0:
    print(f"\n💰 Financial Statistics (Accepted Projects):")
    
    total_budget_accepted = df_accepted_payment['اعتبار'].sum() / 1000
    total_payment_accepted = df_accepted_payment['پرداخت سال جاری'].sum() / 1000
    overall_absorption_accepted = (total_payment_accepted / total_budget_accepted) * 100
    
    print(f"   • Total approved budget: {total_budget_accepted:,.0f} billion Rials")
    print(f"   • Total actual payment: {total_payment_accepted:,.0f} billion Rials")
    print(f"   • Overall absorption rate: {overall_absorption_accepted:.1f}%")
    
    print(f"\n📊 Annual Absorption Rate (Accepted Projects):")
    for year, rate in yearly_absorption_accepted.items():
        print(f"   • {year}: {rate:.1f}%")
    
    # آمار توزیع نرخ جذب
    print(f"\n📈 Absorption Rate Distribution:")
    print(f"   • Mean: {df_accepted_payment['نرخ جذب (%)'].mean():.1f}%")
    print(f"   • Median: {df_accepted_payment['نرخ جذب (%)'].median():.1f}%")
    print(f"   • Std Dev: {df_accepted_payment['نرخ جذب (%)'].std():.1f}%")
    print(f"   • Min: {df_accepted_payment['نرخ جذب (%)'].min():.1f}%")
    print(f"   • Max: {df_accepted_payment['نرخ جذب (%)'].max():.1f}%")
    
    # تعداد طرح‌ها بر اساس بازه نرخ جذب
    print(f"\n📊 Projects by Absorption Rate Range:")
    absorption_ranges = [
        (0, 20, 'Very Low (0-20%)'),
        (20, 40, 'Low (20-40%)'),
        (40, 60, 'Medium (40-60%)'),
        (60, 80, 'Good (60-80%)'),
        (80, 100, 'Excellent (80-100%)')
    ]
    
    for min_val, max_val, label in absorption_ranges:
        count = len(df_accepted_payment[
            (df_accepted_payment['نرخ جذب (%)'] >= min_val) & 
            (df_accepted_payment['نرخ جذب (%)'] < max_val)
        ])
        percentage = (count / len(df_accepted_payment)) * 100
        print(f"   • {label}: {count:,} projects ({percentage:.1f}%)")

# محاسبه CAGR
first_year = df_accepted['سال'].min()
last_year = df_accepted['سال'].max()
first_count = len(df_accepted[df_accepted['سال'] == first_year])
last_count = len(df_accepted[df_accepted['سال'] == last_year])
years_diff = last_year - first_year

if years_diff > 0 and first_count > 0:
    cagr_accepted = ((last_count / first_count) ** (1/years_diff) - 1) * 100
    print(f"\n📊 CAGR (Accepted Projects): {cagr_accepted:.1f}%")
else:
    print(f"\n📊 CAGR calculation not available")

print("\n✅ All Section 3-4 charts (Accepted Projects) saved successfully!")
print("="*70)

# ==============================================================================
# Additional Analysis: Acceptance Rate by Project Type
# ==============================================================================

print("\n" + "="*70)
print("Additional Analysis: Acceptance by Project Type")
print("="*70)

if 'نوع طرح' in df.columns:
    print(f"\n📊 Acceptance Rate by Project Type:")
    
    for project_type in df['نوع طرح'].unique():
        if pd.notna(project_type):
            type_total = len(df[df['نوع طرح'] == project_type])
            type_accepted = len(df_accepted[df_accepted['نوع طرح'] == project_type])
            type_rate = (type_accepted / type_total) * 100 if type_total > 0 else 0
            
            print(f"   • {project_type}:")
            print(f"     - Total: {type_total:,} projects")
            print(f"     - Accepted: {type_accepted:,} projects")
            print(f"     - Acceptance rate: {type_rate:.1f}%")

# ==============================================================================
# Additional Analysis: Acceptance Rate by Academic Type
# ==============================================================================

if 'نوع دانشگاهی' in df.columns:
    print(f"\n📊 Acceptance Rate by Academic Type:")
    
    for academic_type in df['نوع دانشگاهی'].unique():
        if pd.notna(academic_type):
            academic_total = len(df[df['نوع دانشگاهی'] == academic_type])
            academic_accepted = len(df_accepted[df_accepted['نوع دانشگاهی'] == academic_type])
            academic_rate = (academic_accepted / academic_total) * 100 if academic_total > 0 else 0
            
            print(f"   • {academic_type}:")
            print(f"     - Total: {academic_total:,} projects")
            print(f"     - Accepted: {academic_accepted:,} projects")
            print(f"     - Acceptance rate: {academic_rate:.1f}%")

# ==============================================================================
# Summary Table Export (Optional - for report writing)
# ==============================================================================

print(f"\n📋 Creating summary tables for report...")

# جدول ۱: آمار سالانه
summary_yearly = pd.DataFrame({
    'سال': sorted(df['سال'].unique()),
    'کل طرح‌ها': [len(df[df['سال'] == y]) for y in sorted(df['سال'].unique())],
    'طرح‌های پذیرفته‌شده': [len(df_accepted[df_accepted['سال'] == y]) for y in sorted(df['سال'].unique())],
    'نسبت پذیرش (%)': [
        (len(df_accepted[df_accepted['سال'] == y]) / len(df[df['سال'] == y]) * 100) 
        if len(df[df['سال'] == y]) > 0 else 0
        for y in sorted(df['سال'].unique())
    ]
})

print(f"\n📊 Yearly Summary Table:")
print(summary_yearly.to_string(index=False))

# جدول ۲: کمیسیون‌های برتر
summary_commissions = pd.DataFrame({
    'کمیسیون': commission_accepted.index,
    'تعداد': commission_accepted.values
})

print(f"\n📊 Top Commissions Table:")
print(summary_commissions.to_string(index=False))

# جدول ۳: استان‌های برتر
summary_provinces = pd.DataFrame({
    'استان': province_accepted.index,
    'تعداد': province_accepted.values
})

print(f"\n📊 Top Provinces Table:")
print(summary_provinces.to_string(index=False))

# ذخیره جداول در فایل Excel (اختیاری)
try:
    output_excel = output_dir / 'summary_tables_section_3_4.xlsx'
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        summary_yearly.to_excel(writer, sheet_name='Yearly Summary', index=False)
        summary_commissions.to_excel(writer, sheet_name='Top Commissions', index=False)
        summary_provinces.to_excel(writer, sheet_name='Top Provinces', index=False)
    
    print(f"\n✓ Summary tables saved to: {output_excel}")
except Exception as e:
    print(f"\n⚠ Could not save Excel file: {e}")

print("\n" + "="*70)
print("✅ SECTION 3-4 ANALYSIS COMPLETE!")