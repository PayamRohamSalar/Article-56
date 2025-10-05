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

# Print comparison statistics
print("\n" + "="*70)
print("Chart 7: Budget Comparison Statistics")
print("="*70)
print(f"\n📊 Academic Projects:")
print(f"   • Count: {len(academic_data):,}")
print(f"   • Median: {stats_academic['median']:,.0f} million Rials")
print(f"   • Mean: {stats_academic['mean']:,.0f} million Rials")
print(f"   • Q1: {stats_academic['q1']:,.0f} million Rials")
print(f"   • Q3: {stats_academic['q3']:,.0f} million Rials")

print(f"\n📊 Non-Academic Projects:")
print(f"   • Count: {len(non_academic_data):,}")
print(f"   • Median: {stats_non_academic['median']:,.0f} million Rials")
print(f"   • Mean: {stats_non_academic['mean']:,.0f} million Rials")
print(f"   • Q1: {stats_non_academic['q1']:,.0f} million Rials")
print(f"   • Q3: {stats_non_academic['q3']:,.0f} million Rials")

# Statistical test
median_ratio = stats_non_academic['median'] / stats_academic['median']
print(f"\n📈 Median Ratio (Non-Academic/Academic): {median_ratio:.2f}x")


# ==============================================================================
# Statistical Summary for Report Text
# ==============================================================================

print("\n" + "="*70)
print("Statistical Summary for Report:")
print("="*70)

# General statistics
print(f"\n📊 Overall Statistics:")
print(f"   • Total projects: {len(df):,}")
print(f"   • Total budget: {df['اعتبار'].sum()/1000:,.0f} billion Rials")
print(f"   • Average budget: {df['اعتبار'].mean():,.0f} million Rials")
print(f"   • Median budget: {df['اعتبار'].median():,.0f} million Rials")

# Annual statistics
print(f"\n📈 Annual Trend:")
for year in sorted(df['سال'].unique()):
    year_data = df[df['سال'] == year]
    print(f"   • {year}: {len(year_data):,} projects, "
          f"{year_data['اعتبار'].sum()/1000:,.0f} billion Rials")

# Growth rate (CAGR)
first_year = df['سال'].min()
last_year = df['سال'].max()
first_count = df[df['سال'] == first_year].shape[0]
last_count = df[df['سال'] == last_year].shape[0]
years_diff = last_year - first_year

if years_diff > 0 and first_count > 0:
    growth_rate = ((last_count / first_count) ** (1/years_diff) - 1) * 100
    print(f"\n📊 Compound Annual Growth Rate (CAGR): {growth_rate:.1f}%")
else:
    print(f"\n📊 CAGR calculation not available (insufficient data)")

print("\n✅ All charts saved successfully!")
