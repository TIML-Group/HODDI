# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc
from scipy.interpolate import make_interp_spline

# Read csv file
data = pd.read_csv(r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\SE_positive_samples.csv')

# Figure settings
fig_width = 5
fig_height = 3.8
left_margin = 0.5  
right_margin = 0.3
top_margin = 0.2
bottom_margin = 0.7
total_width = fig_width + left_margin + right_margin
total_height = fig_height + top_margin + bottom_margin

plt.figure(figsize=(total_width, total_height), dpi=300)
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": 'Times New Roman',
    'font.size': 23,
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})
rc('mathtext', fontset='stix')

plt.subplots_adjust(
    left=left_margin/total_width,
    right=1-right_margin/total_width,
    top=1-top_margin/total_height,
    bottom=bottom_margin/total_height
)

# Generate gradient colors
n_bars = len(data)
base_color = np.array([60, 184, 170])
colors = []

end_color = np.array([48, 147, 136])  
start_color = np.array([138, 212, 204])   
colors = []
for i in range(n_bars-1, -1, -1):
   factor = i / (n_bars - 1)
   color = start_color + (end_color - start_color) * factor
   colors.append('#%02x%02x%02x' % tuple(color.astype(int)))


plt.bar(range(len(data)), data['num_SE'], color=colors)

# Create a smooth line
x = range(len(data))
y = data['num_SE']
x_smooth = np.linspace(min(x), max(x), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)
plt.plot(x_smooth, y_smooth, '--', color='#2a8277', alpha=0.5)

# Add vertical lines
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)  # x=5 at first bar's center (index 0)
plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)  # x=50 between bars 41-50 and 51-100

plt.xticks(range(len(data)), data['num_occurences'], rotation=30, fontsize=20)

plt.xlabel('# Occurrence', labelpad=1)
plt.ylabel(r'# Side Effect ($\times 10^2$)')

y_max = max(data['num_SE'])
plt.ylim(0, 3000)  # 30 x 100
plt.yticks(np.arange(0, 3100, 500), [str(int(x/100)) for x in np.arange(0, 3100, 500)])

plt.savefig('SE_positive_zoom-in2.pdf', bbox_inches='tight', dpi=300)
plt.show()