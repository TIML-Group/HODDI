# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from scipy.interpolate import make_interp_spline

# Read csv file
data = pd.read_csv('demo6_visualization_for_publication_revised_v3/1-1_big_quarterly_table.csv')

# Figure settings
fig_width, fig_height = 8, 10
margins = {'left': 0.5, 'right': 0.3, 'top': 0.2, 'bottom': 0.5}
total_width = fig_width + margins['left'] + margins['right']
total_height = fig_height + margins['top'] + margins['bottom']

fig = plt.figure(figsize=(total_width, total_height), dpi=300)
ax = fig.add_subplot(111)

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
   "font.family": 'Times New Roman',
   'font.size': 18,
   'text.color': '#000000',
   'axes.labelcolor': '#000000',
   'xtick.color': '#000000',
   'ytick.color': '#000000'
})
rc('mathtext', fontset='stix')

# Multiply the left bar lengths by 5
ax.barh(range(len(data)), -data['condition1']*5/1000, label='Condition 1', color='#d82027')
ax.barh(range(len(data)), -data['condition2']*5/1000, left=-data['condition1']*5/1000, label='Condition 2', color='#fc8d58')
ax.barh(range(len(data)), -data['condition3']*5/1000, left=-(data['condition1']+data['condition2'])*5/1000, label='Condition 3', color='#ffe090')

# Right bar
ax.barh(range(len(data)), data['below0.8']/1000, label='< 0.8', color='#4575b5')
ax.barh(range(len(data)), data['0.8to0.9']/1000, left=data['below0.8']/1000, label='0.8 - 0.9', color='#6baed6')
ax.barh(range(len(data)), data['above0.9']/1000, left=(data['below0.8']+data['0.8to0.9'])/1000, label='> 0.9', color='#91bedb')

# Add trend lines
y = np.arange(len(data))
for col, sign in [('total1', -5), ('total2', 1)]:  
   x = sign * data[col]/1000
   y_smooth = np.linspace(0, len(data)-1, 200)
   spl = make_interp_spline(y, x, k=3)
   x_smooth = spl(y_smooth)
   ax.plot(x_smooth, y_smooth, '--', color='#815279', alpha=0.5)



# X axis settings
max_tick = max(abs(min(ax.get_xlim())), abs(max(ax.get_xlim())))
ax.set_xlim(-max_tick, max_tick)
left_ticks = np.array([-20, -15, -10, -5, 0])  
right_ticks = np.arange(0, 21, 4)  
ax.set_xticks(np.concatenate([left_ticks, right_ticks]))
ax.set_xticklabels([4, 3, 2, 1, 0] + [i for i in range(0, 21, 4)], 
                   fontsize=18, fontfamily='Times New Roman')

# Y axis settings
ax.set_yticks(range(len(data)))
ax.set_yticklabels(data['Time'], fontsize=16, fontfamily='Times New Roman')
ax.set_ylim(-0.5, len(data)-0.5)

ax.set_xlabel('# Record (×10$^3$)', fontsize=20, fontfamily='Times New Roman', math_fontfamily='cm')
ax.set_ylabel('Time', fontsize=20, fontfamily='Times New Roman')

# Legend settings
legend_left = ax.legend(handles=ax.containers[:3], labels=['Condition 1', 'Condition 2', 'Condition 3'],
                      loc='upper left',
                      prop={'family': 'Times New Roman', 'size': 16},
                      frameon=True, edgecolor='#D3D3D3', facecolor='white', framealpha=1)
legend_left.get_frame().set_linewidth(1)

legend_right = ax.legend(handles=ax.containers[3:], labels=['< 0.8', '0.8 - 0.9', '> 0.9'],
                       loc='upper right',
                       prop={'family': 'Times New Roman', 'size': 16},
                       frameon=True, edgecolor='#D3D3D3', facecolor='white', framealpha=1)
legend_right.get_frame().set_linewidth(1)
ax.add_artist(legend_left)

plt.savefig('stacked_bar_graphs.pdf', bbox_inches='tight', dpi=200, orientation='portrait')
plt.show()