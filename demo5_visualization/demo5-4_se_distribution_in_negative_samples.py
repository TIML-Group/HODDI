# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import ScalarFormatter

# Read csv file
data = pd.read_csv(r'dataset\condition123subsets\statistics\2014Q3_2024Q3\negative_samples_SE_statistics.csv')

# Figure settings
fig_width = 5
fig_height = 3.8
dpi = 300
left_margin = 0.5
right_margin = 0.3
top_margin = 0.2
bottom_margin = 0.5
total_width = fig_width + left_margin + right_margin
total_height = fig_height + top_margin + bottom_margin

plt.figure(figsize=(total_width, total_height), dpi=dpi)
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
for i in range(n_bars-1, -1, -1):  
   factor = 1 - (i / (n_bars - 1)) * 0.7
   color = base_color * factor
   colors.append('#%02x%02x%02x' % tuple(color.astype(int)))

plt.bar(data['num_occurence'], data['num_SE'], color=colors)

# Add smooth line
x = data['num_occurence']
y = data['num_SE']
x_smooth = np.linspace(min(x), max(x), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)
plt.plot(x_smooth, y_smooth, '--', color='#2a8277', alpha=0.5)


# Add vertical dash lines
plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

plt.xlim(0, 61)
plt.ylim(0, 401)
xticks = np.arange(10, 61, 10)  
plt.xticks(xticks)  
plt.yticks(np.arange(0, 401, 100), [f'{int(x/100)}' for x in np.arange(0, 401, 100)])
plt.xlabel('# Occurrence')
plt.ylabel(r'# Side Effect ($\times 10^2$)')
plt.savefig('negative_samples_se_statistics.pdf', bbox_inches='tight', dpi=dpi)
plt.show()