# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc
from scipy.interpolate import make_interp_spline

# Read csv file
data = pd.read_csv(r'dataset\condition123subsets\statistics\2014Q3_2024Q3\positive_samples_drug_statistics.csv')

# Figure settings
fig_width = 5
fig_height = 3.8
left_margin = 0.5
right_margin = 0.3
top_margin = 0.2
bottom_margin = 0.5
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

# Set gradient colors
n_bars = len(data)
base_color = np.array([196, 153, 187])  
colors = []

darkest = np.array([113, 74, 108])  # #714a6c
lightest = np.array([199, 165, 194])  # #c7a5c2

for i in range(n_bars):
   t = np.log(1 + 50 * i / (n_bars - 1)) / np.log(51)  
   color = darkest + (lightest - darkest) * t
   colors.append('#%02x%02x%02x' % tuple(color.astype(int)))


plt.bar(data['num_drug'], data['num_records'], color=colors)

x = data['num_drug']
y = data['num_records']
x_smooth = np.linspace(min(x), max(x), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)
plt.plot(x_smooth, y_smooth, '--', color='#815279', alpha=0.5)

plt.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.5)

plt.xlim(0, 101)
plt.ylim(0, 25000)

xticks = np.arange(10, 101, 10)  
plt.xticks(xticks)  


plt.yticks(np.arange(0, 30000, 5000), [f'{int(x/1000)}' if x != 0 else '0' for x in np.arange(0, 30000, 5000)])
plt.xlabel('# Drug / Record')
plt.ylabel('# Record (×10$^3$)')

plt.savefig('positive_drugs_statistics.pdf', bbox_inches='tight', dpi=300)
plt.show()