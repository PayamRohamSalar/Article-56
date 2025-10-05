
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For proper Persian (Farsi) rendering
import arabic_reshaper
from bidi.algorithm import get_display

# NEW: geo imports for choropleth
try:
    import geopandas as gpd
except Exception as e:
    gpd = None

# ==============================================================================
# Font Configuration for Persian (Farsi) Text  (aligned with s3/s4)
# ==============================================================================
# NOTE: If Vazirmatn font is installed in a custom path, update the path below.
# Otherwise, the script falls back to "Tahoma".
font_path = Path(r'D:\OneDrive\AI-Project\Article56\fonts\ttf\Vazirmatn-Regular.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Vazirmatn'
else:
    # Try generic family first (if already installed on system)
    plt.rcParams['font.family'] = 'Vazirmatn'
    # Fallback message + fallback font
    if plt.rcParams['font.family'] != 'Vazirmatn':
        print(f"Warning: Font not found at {font_path}")
        print("Falling back to default font...")
        plt.rcParams['font.family'] = 'Tahoma'

# Matplotlib general config
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

# ==============================================================================
# Helper Functions (reused style from s3/s4)
# ==============================================================================
def fix_persian_text(text: str) -> str:
    """
    تبدیل متن فارسی/عربی به فرمت قابل نمایش در matplotlib (RTL + Reshape)
    """
    if text is None or str(text).strip() == '':
        return ''
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception as e:
        print(f"Warning: Could not reshape text '{text}': {e}")
        return str(text)

def convert_to_persian_number(x) -> str:
    """
    تبدیل اعداد انگلیسی به اعداد فارسی (درون متن و برچسب‌ها)
    """
    english_digits = '0123456789'
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    trans = str.maketrans(english_digits, persian_digits)
    return str(x).translate(trans)

def format_number_with_separator(number, use_persian: bool=True) -> str:
    """
    قالب‌بندی عدد با جداکننده هزارگان و تبدیل اختیاری به اعداد فارسی
    """
    if isinstance(number, (int, float, np.integer, np.floating)):
        formatted = f'{number:,.0f}'
    else:
        formatted = str(number)
    return convert_to_persian_number(formatted) if use_persian else formatted

def find_province_column(df: pd.DataFrame) -> str:
    """
    تشخیص نام ستون «استان»
    """
    candidates = ['استان', 'استان اجرا', 'نام استان', 'استان محل اجرا']
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("ستون «استان» در فایل داده یافت نشد. لطفاً یکی از ستون‌های "
                   "['استان','استان اجرا','نام استان','استان محل اجرا'] را اضافه/نام‌گذاری کنید.")

def safe_group_sum(df: pd.DataFrame, by: str, value_col: str) -> pd.Series:
    """
    گروه‌بندی و جمع اعداد با حذف رکوردهای NaN در ستون مقدار
    """
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors='coerce')
    tmp = tmp[tmp[value_col].notna()]
    return tmp.groupby(by)[value_col].sum().sort_values(ascending=False)

def gini_coefficient(x: np.ndarray) -> float:
    """
    محاسبه ضریب جینی برای آرایه‌ای از مقادیر (x ≥ 0)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.any(x < 0):
        raise ValueError("مقادیر منفی برای محاسبه ضریب جینی مجاز نیست.")
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = sorted_x.size
    cumx = np.cumsum(sorted_x)
    # فرمول استاندارد (۰ تا ۱)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)

def lorenz_curve_points(x: np.ndarray):
    """
    محاسبه نقاط منحنی لورنز (محور افقی: سهم تجمعی استان‌ها، محور عمودی: سهم تجمعی مقدار)
    خروجی: (cum_pop, cum_value) هر کدام به بازه [0,1] نرمال شده
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.array([0,1]), np.array([0,1])
    sorted_x = np.sort(x)
    cum_values = np.cumsum(sorted_x)
    cum_values = np.insert(cum_values, 0, 0.0)
    cum_values = cum_values / cum_values[-1]
    cum_pop = np.arange(0, n + 1) / n
    return cum_pop, cum_values

# NEW: name normalizer so DF names match Geo names
def normalize_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    # unify Arabic/Persian forms + remove ZWNJ
    replacements = {
        'ي': 'ی', 'ك': 'ک', 'ۀ': 'ه', 'ة': 'ه', '‌': '', '\u200c': ''
    }
    for a,b in replacements.items():
        s = s.replace(a,b)
    return s

# ==============================================================================
# Paths & IO
# ==============================================================================
base_dir = Path.cwd()
data_file = base_dir / 'data' / 'Q_Useful_Source_Data.xlsx'
if not data_file.exists():
    # fallback to current dir
    alt = base_dir / 'Q_Useful_Source_Data.xlsx'
    if alt.exists():
        data_file = alt
    else:
        raise FileNotFoundError(f"Expected data file not found at: {data_file} (or {alt})")

output_dir = base_dir / 'fig/S5'
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# NEW: GeoJSON candidate paths (your Windows path first)
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
if geojson_file is None:
    print('⚠ GeoJSON not found. Choropleth will be skipped unless path is corrected.')

# ==============================================================================
# Load & Prepare Data
# ==============================================================================
df = pd.read_excel(data_file)

# Standardize key columns
if 'سال' not in df.columns:
    raise KeyError("ستون «سال» در داده‌ها وجود ندارد.")

df['سال'] = pd.to_numeric(df['سال'], errors='coerce').astype('Int64')
df = df[df['سال'].notna()]

# اعتبار: بر اساس General Prompt واحد «میلیون ریال» است.
value_col = 'اعتبار'
if value_col not in df.columns:
    raise KeyError("ستون «اعتبار» در داده‌ها یافت نشد.")
df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df = df[df[value_col].notna()]

province_col = find_province_column(df)

# محدود کردن بازه سالی در صورت نیاز (۱۳۹۸ تا ۱۴۰۳)
df = df[(df['سال'] >= 1398) & (df['سال'] <= 1403)]

print(f"✓ Total records: {len(df):,}")
print(f"✓ Years: {df['سال'].min()} – {df['سال'].max()}")
print(f"✓ Province column: {province_col}")

# ==============================================================================
# Chart 5-1: Heatmap ساده استان × سال (اعتبار)  — (بدون تغییر)
# ==============================================================================
pivot_cred = (df
              .groupby([province_col, 'سال'])[value_col]
              .sum()
              .reset_index()
              .pivot(index=province_col, columns='سال', values=value_col)
              .fillna(0))

pivot_cred['__TOTAL__'] = pivot_cred.sum(axis=1)
pivot_cred = pivot_cred.sort_values('__TOTAL__', ascending=False)
pivot_cred = pivot_cred.drop(columns='__TOTAL__')

fig, ax = plt.subplots(figsize=(14, max(8, 0.35 * len(pivot_cred))))
sns.heatmap(pivot_cred,
            cmap='RdYlGn',  # سبز ↔ قرمز
            linewidths=0.3,
            linecolor='white',
            cbar_kws={'label': fix_persian_text('اعتبار (میلیون ریال)')},
            ax=ax)

ax.set_yticklabels([fix_persian_text(t.get_text()) for t in ax.get_yticklabels()], rotation=0, fontsize=10)
ax.set_xticklabels([convert_to_persian_number(t.get_text()) for t in ax.get_xticklabels()], rotation=0, fontsize=10)

ax.set_title(fix_persian_text('توزیع سالانه اعتبارات به تفکیک استان (Heatmap ساده)'),
             fontsize=16, fontweight='bold', pad=16)

plt.tight_layout()
out1 = output_dir / 'chart_5_1_heatmap.png'
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Chart 5-1 saved: {out1}")

# ==============================================================================
# Chart 5-2: Top 10 & Bottom 10 Provinces by Total Credit  — (بدون تغییر)
# ==============================================================================
province_credit_total = safe_group_sum(df, province_col, value_col)
top10 = province_credit_total.head(10)[::-1]
bottom10 = province_credit_total.tail(10)[::-1]

def plot_rank_bar(series: pd.Series, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(series)), series.values, color=sns.color_palette('RdYlGn', n_colors=len(series)))
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([fix_persian_text(i) for i in series.index], fontsize=11)
    ax.set_xlabel(fix_persian_text('اعتبار (میلیون ریال)'), fontsize=13, fontweight='bold', labelpad=8)
    ax.set_title(fix_persian_text(title), fontsize=15, fontweight='bold', pad=14)
    ax.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    maxv = series.values.max() if len(series) else 0
    for i, v in enumerate(series.values):
        ax.text(v + maxv * 0.01, i, format_number_with_separator(v, True),
                va='center', ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.7))
    plt.tight_layout()
    outp = output_dir / filename
    plt.savefig(outp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {outp}")

plot_rank_bar(top10, '۱۰ استان با بیشترین اعتبارات (۱۳۹۸–۱۴۰۳)', 'chart_5_2_top10.png')
plot_rank_bar(bottom10, '۱۰ استان با کمترین اعتبارات (۱۳۹۸–۱۴۰۳)', 'chart_5_2_bottom10.png')

# ==============================================================================
# Chart 5-3: Lorenz Curve + Gini for Provinces (Credits & Project Counts) — (بدون تغییر)
# ==============================================================================
cred_by_prov = safe_group_sum(df, province_col, value_col)
cnt_by_prov = df.groupby(province_col).size().sort_values(ascending=False)

def plot_lorenz_gini(values: pd.Series, title: str, filename: str):
    arr = values.values.astype(float)
    gini = gini_coefficient(arr)
    x, y = lorenz_curve_points(arr)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0,1], [0,1], linestyle='--', linewidth=1.5, color='gray', label=fix_persian_text('خط برابری'))
    ax.plot(x, y, linewidth=3)
    ax.fill_between(x, y, x, alpha=0.15)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(fix_persian_text('سهم تجمعی استان‌ها'), fontsize=13, fontweight='bold')
    ax.set_ylabel(fix_persian_text('سهم تجمعی مقدار'), fontsize=13, fontweight='bold')
    title_txt = f"{title}\n{fix_persian_text('ضریب جینی')}: {convert_to_persian_number(f'{gini:.3f}')}"
    ax.set_title(fix_persian_text(title_txt), fontsize=15, fontweight='bold', pad=14)
    ticks = np.linspace(0,1,6)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([convert_to_persian_number(f'{int(t*100)}%') for t in ticks])
    ax.set_yticklabels([convert_to_persian_number(f'{int(t*100)}%') for t in ticks])
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    outp = output_dir / filename
    plt.savefig(outp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {outp}")

plot_lorenz_gini(cred_by_prov, 'منحنی لورنز اعتبارات استان‌ها (۱۳۹۸–۱۴۰۳)', 'chart_5_3_lorenz_credit.png')
plot_lorenz_gini(cnt_by_prov, 'منحنی لورنز تعداد طرح‌های استان‌ها (۱۳۹۸–۱۴۰۳)', 'chart_5_3_lorenz_projects.png')

# ==============================================================================
# Chart 5-4 (NEW): Choropleth — اعتبار سال ۱۴۰۳ به تفکیک استان
# ==============================================================================
if gpd is None:
    print('⚠ geopandas is not installed. Run: pip install geopandas shapely fiona pyproj rtree')
elif geojson_file is None:
    print('⚠ GeoJSON path not found. Please update geojson_candidates to the correct path.')
else:
    try:
        gdf_prov = gpd.read_file(geojson_file)
        # استانداردسازی نام‌ها برای جوین
        if 'NAME_1' not in gdf_prov.columns and 'name' in gdf_prov.columns:
            gdf_prov = gdf_prov.rename(columns={'name': 'NAME_1'})
        gdf_prov['prov_norm'] = gdf_prov['NAME_1'].map(normalize_name)
        # داده سال ۱۴۰۳
        df_1403 = df[df['سال'] == 1403].copy()
        df_1403['prov_norm'] = df_1403[province_col].map(normalize_name)
        summ_1403 = (df_1403
                     .groupby('prov_norm')[value_col]
                     .sum()
                     .reset_index()
                     .rename(columns={value_col: 'credit_1403'}))
        gdf_plot = gdf_prov.merge(summ_1403, on='prov_norm', how='left')
        gdf_plot['credit_1403'] = gdf_plot['credit_1403'].fillna(0)
        gdf_plot['centroid'] = gdf_plot.geometry.representative_point()
        # رسم نقشه
        fig, ax = plt.subplots(figsize=(10, 12))
        gdf_plot.plot(
            ax=ax, column='credit_1403', cmap='YlGnBu',
            linewidth=0.6, edgecolor='#888', legend=True,
            legend_kwds={'label': fix_persian_text('اعتبار ۱۴۰۳ (میلیون ریال)')}
        )
        # برچسب‌گذاری فارسی (نام استان)
        for _, row in gdf_plot.iterrows():
            x, y = row['centroid'].x, row['centroid'].y
            ax.text(x, y, fix_persian_text(row['NAME_1']), ha='center', va='center', fontsize=7, color='#333', alpha=0.9)
        # تیتر و ظاهر
        ax.set_title(fix_persian_text('پراکنش اعتبارات سال ۱۴۰۳ به تفکیک استان (Choropleth)'), fontsize=16, fontweight='bold', pad=12)
        ax.set_axis_off()
        # فارسی‌سازی اعداد رنگ‌نما
        try:
            cax = ax.get_figure().axes[-1]
            cax.set_yticklabels([convert_to_persian_number(t.get_text()) for t in cax.get_yticklabels()])
        except Exception:
            pass
        out_map = output_dir / 'chart_5_4_choropleth_1403.png'
        plt.tight_layout()
        plt.savefig(out_map, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'✓ Chart 5-4 saved: {out_map}')
    except Exception as e:
        print(f'⚠ Choropleth failed: {e}')

print("✓ فصل ۵ — نمودارها تولید شد.")
