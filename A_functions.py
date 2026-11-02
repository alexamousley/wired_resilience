### Import functions ###

# Standard library
import os
import re
import math
import itertools
import textwrap
import warnings
from collections import Counter, defaultdict

# Numerical / data
import numpy as np
import pandas as pd
from numpy.linalg import svd

# SciPy
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr, norm, rankdata, ttest_ind, linregress, zscore

# Statistics
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.stats.multitest import multipletests

# Machine learning
from sklearn.cross_decomposition import PLSSVD
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pycirclize import Circos

# Parallel processing / progress bars
from joblib import Parallel, delayed
from tqdm import trange
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# Network analysis
import bct

########################################## COLORS FOR PLOTTING #############################################
colors = {
    'Adversity':     '#8B0000',   
    'Household/Community Instability':     '#8B0000',   
    'Physical/Sexual Abuse':     '#8B0000',
    'Parental Neglect':     '#8B0000',   
    'Financial Difficulties':     '#8B0000',  
    'Mental Health': '#1F6FB2',   
    'Genetics':      '#D9A300',  
    'Overlap':       '#7B3294',   
    'Cognition': "#17BDAD"
}

# Build colormaps associated with domains
color_maps = {
    'Adversity':     LinearSegmentedColormap.from_list('adv',  ['#FDECEC', '#E24B4A', '#8B0000']),
    'Mental Health': LinearSegmentedColormap.from_list('mh',   ['#E8F1FA', '#4A90C2', '#0C447C']),
    'Genetics':      LinearSegmentedColormap.from_list('gen',  ['#FBF3D9', '#E0B83C', '#9C6F00']),
    'Overlap':       LinearSegmentedColormap.from_list('ovl',  ['#F2E9F5', '#9D6BB3', '#4B1B66']),
}

######################################### GENERAL FUNCTIONS ################################################

def vectorize(matrix):
    "Convert a symmetric matrix to a vector by taking the upper triangle (excluding diagonal)"

    n = matrix.shape[0]                    # Number of nodes
    triu_indices = np.triu_indices(n, k=1) # Get indices of the upper triangle (excluding diagonal)
    vector = matrix[triu_indices]          # Turn matrix into vector
    return vector

def convert_to_matrix(vector,nnodes,poss_edges,zero_mask,removed_zeros=True):
    "Convert a vector back to a symmetric matrix, optionally restoring zeros if they were removed"

    if removed_zeros:
        restored_vector = np.zeros(poss_edges)     # Preallocate full vector with zeros
        restored_vector[~zero_mask] = vector       # Fill in the non-zero edges
    else:
        restored_vector = vector                   # If zeros were not removed, just use the original vector

    # Convert vector back to matrix
    triu_idx = np.triu_indices(nnodes, k=1)        # Get upper triangle indices (excluding diagonal)  
    matrix = np.zeros((nnodes, nnodes))            # Preallocate the symmetric matrix   
    matrix[triu_idx] = restored_vector             # Fill the upper triangle
    matrix = matrix + matrix.T                     # Mirror to lower triangle

    return matrix

############################################### FUNCTIONS FOR B_sample_diversity ##########################################

def descriptive_stats(df, vars):

    print("DESCRIPTIVE STATISTICS")

    # Total participants
    print(f"\nTotal N: {len(df)}\n")

    ### Statistics on continous variables ###

    print(f"{'Variable':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8} {'SD':>8}")
    print("-"*80)

    binary_vars = []
    for var in vars:
        if var not in df.columns:
            print(f"{var:<20} NOT FOUND")
            continue

        col = df[var].dropna()
        unique_vals = col.unique()

        # Detect binary variable
        if len(unique_vals) <= 2:
            binary_vars.append(var)
            continue

        print(f"{var:<20} "
              f"{col.min():>8.2f} "
              f"{col.max():>8.2f} "
              f"{col.mean():>8.2f} "
              f"{col.median():>8.2f} "
              f"{col.std():>8.2f}")

    # Statsitics on sex
    if binary_vars:
        for var in binary_vars:
            col = df[var]

            # Special handling for sex
            if var.lower() == 'sex':
                label_map = {1: 'Male', 2: 'Female'}
                col = col.map(label_map)

            counts = col.value_counts()
            percentages = col.value_counts(normalize=True) * 100

            print(f"")
            for category in counts.index:
                print(f"{str(category):<10} n = {counts[category]:<5} ({percentages[category]:.1f}%)")

def plot_demo_pyramid(z_data, age_col, sex_col, save_dir,
                         name='age_sex_pyramid',
                         sex_labels={1.0: 'Male', 2.0: 'Female'},
                         colors={'Male': 'tab:blue', 'Female': 'tab:red'},
                         age_label='Age', n_bins=30):
    "Back-to-back population pyramid of age by sex (males left, females right)"

    sub = z_data[[age_col, sex_col]].dropna()
    vals = sub[age_col].to_numpy()

    # integer-aligned bin edges spanning the full range
    lo, hi = int(np.floor(vals.min())), int(np.ceil(vals.max()))
    width = max(1, int(np.ceil((hi - lo) / n_bins)))
    edges = np.arange(lo, hi + width + 1, width)
    centers = (edges[:-1] + edges[1:]) / 2

    # counts per bin for each sex, on the SAME edges
    male_code  = [k for k, v in sex_labels.items() if v == 'Male'][0]
    female_code = [k for k, v in sex_labels.items() if v == 'Female'][0]
    male_counts,   _ = np.histogram(sub.loc[sub[sex_col] == male_code,   age_col], bins=edges)
    female_counts, _ = np.histogram(sub.loc[sub[sex_col] == female_code, age_col], bins=edges)

    fig, ax = plt.subplots(figsize=(6, 7))

    # males to the left (negative), females to the right
    ax.barh(centers, -male_counts,   height=width * 0.9,
            color=colors['Male'],   edgecolor='black', linewidth=0.5,
            label='Male', rasterized=True)
    ax.barh(centers,  female_counts, height=width * 0.9,
            color=colors['Female'], edgecolor='black', linewidth=0.5,
            label='Female', rasterized=True)

    ax.axvline(0, color='black', linewidth=0.8)

    # relabel x ticks as absolute counts (hide the negative sign on the male side)
    max_count = max(male_counts.max(), female_counts.max())
    step = max(1, int(np.ceil(max_count / 5 / 10) * 10))  # ~5 ticks, rounded
    ticks = np.arange(0, max_count + step, step)
    ax.set_xticks(np.concatenate([-ticks[::-1], ticks[1:]]))
    ax.set_xticklabels([str(int(t)) for t in np.concatenate([ticks[::-1], ticks[1:]])])

    sns.despine(ax=ax, left=True)
    ax.set_xlabel('Count', size=14)
    ax.set_ylabel(age_label, size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(
        title='Sex',
        fontsize=12,
        title_fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    for coll in ax.collections:
        coll.set_rasterized(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name + '.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

def plot_histograms(z_data, columns, labels, save_dir, name='histograms'):
    "Plot adversity and mental health histograms"

    fig, axes = plt.subplots(1, len(labels), figsize=(15, 4))
    if len(labels) == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        ax = axes[i]
        data = z_data[columns[i]].dropna()
        vals = data.to_numpy()

        is_integer = np.allclose(vals, np.round(vals))
        n_unique = np.unique(vals).size

        # --- choose bins + KDE smoothing per column ---
        if is_integer and n_unique <= 25:
            # discrete scores: one bar per integer, and smooth the KDE
            # so it doesn't wiggle between values
            bin_kwargs = {'discrete': True}
            bw_adjust = 2.5
        elif is_integer:
            # wide integer range (e.g. Mental Health): integer-aligned edges
            lo, hi = int(vals.min()), int(vals.max())
            width = max(1, int(np.ceil((hi - lo) / 30)))  # ~30 bins, width >= 1
            bin_kwargs = {'bins': np.arange(lo, hi + width + 1, width)}
            bw_adjust = 1.0
        else:
            # truly continuous
            bin_kwargs = {'bins': 'auto'}
            bw_adjust = 1.0

        sns.histplot(
            data,
            kde=True,
            alpha=1,
            kde_kws={'bw_adjust': bw_adjust},
            line_kws={'linewidth': 2},
            color=colors[label],
            ax=ax,
            rasterized=True,
            **bin_kwargs
        )

        for coll in ax.collections:
            coll.set_rasterized(True)

        sns.despine(ax=ax)
        ax.set_xlabel(label, size=14)
        ax.set_ylabel('Count', size=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name + '.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

def plot_pgs_violins(pgs, pgs_labels, categories, save_path=None, name='pgs_violins'):
    """Violin plot of all polygenic scores, grouped and labelled by category."""

    FONT_SIZE_LABEL = 17
    FONT_SIZE_TICK  = 17

    # Build ordered lists following category grouping
    ordered_labels = []
    category_boundaries = []

    pos = 0
    for cat_name, members in categories.items():
        for m in members:
            if m in pgs_labels:
                ordered_labels.append(m)
                pos += 1
        category_boundaries.append(pos - 0.5)

    # Collect data in category order
    data = [pgs[label].values for label in ordered_labels]
    x_positions = np.arange(len(ordered_labels))

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(ordered_labels) * 0.4), 4))

    parts = ax.violinplot(data, positions=x_positions, showmeans=False,
                          showmedians=True, showextrema=False)

    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor(colors["Genetics"])
        pc.set_edgecolor("#583D17")
        pc.set_alpha(0.8)
        pc.set_linewidth(0.5)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.5)

    # Add vertical dashed lines between categories
    for bx in category_boundaries[:-1]:
        ax.axvline(bx, color='gray', linewidth=3, linestyle='--')

    # Abbreviation labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_labels, rotation=45, ha='right',
                       fontsize=FONT_SIZE_TICK, rotation_mode='anchor')
    ax.set_ylabel('Polygenic Score', fontsize=FONT_SIZE_LABEL, labelpad=8)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
    ax.set_xlim(-0.8, len(ordered_labels) - 0.2)

    for coll in ax.collections:
        coll.set_rasterized(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for label in ax.get_xticklabels():
        offset = mtransforms.ScaledTranslation(5/72, 0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, name + '.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

def plot_corr(z_data, columns, labels, x_label, y_label, save_dir,
                     name='adver_mh_correlation'):
    "Plot adversity and mental health correlation"

    fig, ax = plt.subplots(figsize=(7, 5))
 
    x = z_data[columns[labels.index(x_label)]]
    y = z_data[columns[labels.index(y_label)]]
 
    r, p = stats.pearsonr(x, y)
    line_color = "red" if p < 0.05 else "black"
 
    sns.regplot(
        x=x, y=y,
        scatter_kws={'alpha': 0.2, 's': 20, 'color': "grey", 'rasterized': True},
        line_kws={'color': line_color, 'linewidth': 3, 'rasterized': True},
        ci=True,
        ax=ax
    )

    # Rastorize
    for coll in ax.collections:
        coll.set_rasterized(True)

    # Remove top and right spines
    sns.despine(ax=ax)
 
    # Add correlation text with italic r
    ax.text(
        0.7, 0.9,
        f"$r$ = {r:.2f}\n$p$ = {p:.3g}",
        transform=ax.transAxes,
        fontsize=12)
 
    ax.set_xlabel(x_label, size=14)
    ax.set_ylabel(y_label, size=14)
 
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
 
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name + '.pdf'), dpi=600,bbox_inches='tight')
    plt.show()

def plot_quadrants(z_data, columns, labels, x_label, y_label, save_dir,
                            pct=0.30, name='adver_mh_quadrants'):
    """Split sample into 4 extreme groups by top/bottom `pct` of each measure.
    Everyone not extreme on BOTH axes is plotted in grey."""

    fig, ax = plt.subplots(figsize=(7, 5))

    x = z_data[columns[labels.index(x_label)]]
    y = z_data[columns[labels.index(y_label)]]
    n_total = len(x)

    # per-axis percentile cutoffs: low = bottom `pct`, high = top `pct`
    x_lo_thr, x_hi_thr = x.quantile(pct), x.quantile(1 - pct)
    y_lo_thr, y_hi_thr = y.quantile(pct), y.quantile(1 - pct)

    hi_x, lo_x = x >= x_hi_thr, x <= x_lo_thr
    hi_y, lo_y = y >= y_hi_thr, y <= y_lo_thr

    masks = {
        'High / High': hi_x & hi_y,
        'High / Low':  hi_x & lo_y,
        'Low / High':  lo_x & hi_y,
        'Low / Low':   lo_x & lo_y,
    }

    # --- purple shade scale ---
    overlap_purple = '#7E4B9E'
    purple_scale = LinearSegmentedColormap.from_list(
        'overlap_purples', ['#2A0E33', overlap_purple, '#E6D4EE'])
    shade_list = [purple_scale(t) for t in np.linspace(0.10, 0.95, 4)]
    colors_q = {
        'High / High': shade_list[0],   # darkest
        'High / Low':  shade_list[1],
        'Low / High':  shade_list[2],
        'Low / Low':   shade_list[3],   # lightest
    }

    # everyone not in one of the four extreme groups -> grey
    in_group = np.zeros(n_total, dtype=bool)
    for m in masks.values():
        in_group |= m.to_numpy()
    other = ~in_group

    # plot grey first so colored points sit on top
    ax.scatter(x[other], y[other], s=20, alpha=0.15, color='grey',
               rasterized=True)
    for grp, m in masks.items():
        ax.scatter(x[m], y[m], s=20, alpha=0.45, color=colors_q[grp],
                   rasterized=True)

    # dividing lines at the per-axis percentile cutoffs
    for v in (x_hi_thr, x_lo_thr):
        ax.axvline(v, color='black', lw=1, ls='--')
    for v in (y_hi_thr, y_lo_thr):
        ax.axhline(v, color='black', lw=1, ls='--')
    sns.despine(ax=ax)
    for coll in ax.collections:
        coll.set_rasterized(True)
    ax.set_xlabel(x_label, size=14)
    ax.set_ylabel(y_label, size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_rasterization_zorder(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name + '.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

    n_grey = int(other.sum())
    summary = {grp: (int(m.sum()), np.round(100 * int(m.sum()) / n_total, 1))
               for grp, m in masks.items()}
    summary['Middle (grey)'] = (n_grey, np.round(100 * n_grey / n_total, 1))

    return summary

def plot_pgs_corrs(pgs, pgs_labels, data, labels, categories,
                           save_path=None, cmap='coolwarm'):
    " Plot correlations between PGS and outcomes "
 
    FONT_SIZE_TICK = 22
    cmap = plt.get_cmap(cmap)
 
    # Order PGS by category
    ordered_labels = []
    category_boundaries = []   # x-position of the divider line after each category

    pos = 0
    for cat_name, members in categories.items():
        for m in members:
            if m in pgs_labels:
                ordered_labels.append(m)
                pos += 1
        category_boundaries.append(pos - 0.5)
 
    # Rebuild (start_idx, end_idx, category_name) spans from the boundaries,
    # for shading and category titles. Skips any category with no members
    # present in pgs_labels.
    category_spans = []
    start = 0
    for (cat_name, _members), boundary in zip(categories.items(), category_boundaries):
        end = int(boundary - 0.5)
        if end >= start:
            category_spans.append((start, end, cat_name))
            start = end + 1

    # Initialize 
    corr_matrix = np.zeros((len(data.columns), len(ordered_labels)))
    pval_matrix = np.zeros_like(corr_matrix)
    ci_lo = np.zeros_like(corr_matrix)
    ci_hi = np.zeros_like(corr_matrix)
    z_crit = norm.ppf(0.975) # Set CI threshold
    nsub = len(data) # Sample size

    # Loop through each column
    results = []   # one row per outcome × PGS comparison

    # Loop through each column
    for i, col in enumerate(data.columns):
        # Loop through each PGS
        for j, pg in enumerate(ordered_labels):

            # Align PGS and this outcome by subject
            pair = pd.concat([pgs[pg], data[col]], axis=1, join='inner').dropna()
            n = len(pair)

            # Run correlation
            r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

            # Fisher-z 95% CI (use this pair's n, not the overall sample size)
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            ci_lo[i, j] = np.tanh(z - z_crit * se)
            ci_hi[i, j] = np.tanh(z + z_crit * se)

            results.append({
                'outcome': labels[i],
                'pgs': pg,
                'r': r,
                'ci_low': ci_lo[i, j],
                'ci_high': ci_hi[i, j],
                'p': p,
                'n': n,
                'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '',
            })

    # Set up variables for plotting
    sig_mask = pval_matrix < 0.05
    x = np.arange(len(ordered_labels))
    # symmetric color scale so r = 0 always maps to the neutral (middle) color
    vmax = np.abs(corr_matrix).max()
    color_norm = Normalize(vmin=-vmax, vmax=vmax)
 
    # gap between the top of the CI whisker and the significance star.
    # Small relative to the total y-range so the star sits right on top of
    # the whisker instead of floating above it.
    y_pad = 0.0001 * (ci_hi.max() - ci_lo.min())
 
    # ----- figure: one row per feature, all PGS along x -----
    n_rows = len(labels)
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True, sharey=True,
        figsize=(max(10, len(ordered_labels) * 0.65), 3.8 * n_rows),
        gridspec_kw={'hspace': 0.08},
    )
    axes = np.atleast_1d(axes)
 
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        r = corr_matrix[i, :]
        lo, hi = ci_lo[i, :], ci_hi[i, :]
        sig = sig_mask[i, :]
        face = cmap(color_norm(r))
 
        # zero reference line
        ax.axhline(0, color='black', linewidth=1, zorder=1)
 
        # alternating category shading + dashed dividers
        for k, (start, end, name) in enumerate(category_spans):
            if k % 2 == 1:
                ax.axvspan(start - 0.5, end + 0.5, color='grey', alpha=0.07, zorder=0)
        for bx in category_boundaries[:-1]:
            ax.axvline(bx, color='grey', linestyle='--', linewidth=1, zorder=0)
 
        # error bars, colored like their dot
        ax.vlines(x, lo, hi, colors=face, linewidth=2.5, zorder=2)
 
        # filled = significant, hollow = not
        ax.scatter(x[sig], r[sig], s=110, c=r[sig], cmap=cmap, norm=color_norm,
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.scatter(x[~sig], r[~sig], s=90, facecolor='white',
                   edgecolors=face[~sig], linewidth=2, zorder=3)
 
        # significance stars, placed just above each CI whisker
        for j in np.where(sig)[0]:
            p = pval_matrix[i, j]
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
            ax.text(x[j], hi[j] + y_pad, star, ha='center', va='bottom',
                    fontsize=FONT_SIZE_TICK - 3, color=face[j])
 
        ax.set_ylabel(f'{lab}\n', fontsize=FONT_SIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax.set_xlim(-0.7, len(ordered_labels) - 0.3)
        ax.set_ylim(ci_lo.min() - 0.05, ci_hi.max() + 0.05)
 
    import textwrap

    # category names above the top row (wrapped to two lines to avoid overlap)
    for start, end, name in category_spans:
        wrapped_name = textwrap.fill(name, width=14, break_long_words=False)
        axes[0].text((start + end) / 2, 1.05, wrapped_name, ha='center', va='bottom',
                    fontsize=FONT_SIZE_TICK - 4,
                    transform=axes[0].get_xaxis_transform())
 
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(ordered_labels, fontsize=FONT_SIZE_TICK,
                             rotation=45, ha='right', rotation_mode='anchor')
 
    cbar = fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap),
                        ax=list(axes), pad=0.01, fraction=0.025)
    cbar.set_label('Pearson r', fontsize=FONT_SIZE_TICK - 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 4)
 
    # rasterize the scatter/line layers so the saved PDF stays a reasonable size
    for ax in axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
 
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'pgs_correlation_dotplot.pdf'),
                    dpi=600, bbox_inches='tight')
    plt.show()

############################################### FUNCTIONS FOR C_connectome_descriptives ##########################################

def plot_connectome_descriptives(connectomes, euclidean_distance, save_dir=None,
                                 filename='connectome_descriptives.pdf',
                                 bins=50, bar_color='grey', edge_color='black',
                                 font_size_label=16, font_size_tick=15,
                                 figsize=(4.5, 9.5)):
    " Plot histograms of edge length, density and degree across subjects."
    nnodes, _, nsub = connectomes.shape

    # Density and degree per subject
    density = np.array([bct.density_und(connectomes[:, :, s])[0] for s in range(nsub)])
    degree = np.array([bct.degrees_und(connectomes[:, :, s]) for s in range(nsub)])

    # Distances of existing edges (upper triangle, no diagonal)
    iu = np.triu_indices(nnodes, k=1)
    total_distances = euclidean_distance[iu]
    connected_distances_all = np.concatenate([
        total_distances[connectomes[:, :, s][iu] > 0] for s in range(nsub)
    ])

    plt.rcParams.update({
        'font.family'      : 'sans-serif',
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size' : 5,
        'ytick.major.size' : 5,
    })

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig)

    panels = [
        (connected_distances_all, 'Euclidean distance (mm)', 'Count'),
        (density * 100,           'Density (%)',             'Frequency'),
        (degree.flatten(),        'Degree',                  'Frequency'),
    ]

    for i, (data, xlabel, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.hist(data, bins=bins, color=bar_color, edgecolor=edge_color,
                linewidth=0.4, alpha=0.9, rasterized=True)
        ax.set_xlabel(xlabel, fontsize=font_size_label, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=font_size_label, labelpad=8)
        ax.tick_params(labelsize=font_size_tick)

    def _summary(x):
        return {'max': np.max(x), 'min': np.min(x), 'sd': np.std(x)}

    stats = {
        'degree': _summary(degree.flatten()),
        'edge_length': _summary(connected_distances_all),
        'density': _summary(density * 100),
    }

    for name, s in stats.items():
        print(f"{name}: Max - {np.round(s['max'])}, Min - {np.round(s['min'])}, SD - {np.round(s['sd'])}")

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, filename), dpi=600,
                    bbox_inches='tight', facecolor='white')
    plt.show()

def plot_axial_node_plot(network, atlas_coords, lh_vertices, rh_vertices,
                         keep_fraction=0.8, edge_color="#8AD899", edge_alpha=0.3,
                         lw_min=1, lw_max=6, node_size=100,
                         save_dir=None, filename='node_plot_example.pdf',
                         figsize=(10, 8)):
    " Axial view node plot on top of cortical surface "
    nnodes = network.shape[0]

    # Non-zero edges in the upper triangle
    iu = np.triu_indices(nnodes, k=1)
    tri_weights = network[iu]
    nonzero_weights = tri_weights[tri_weights > 0]
    n_available = nonzero_weights.size
    n_keep = int(np.floor(keep_fraction * n_available))
    print(f"Total edges: {n_available} -> keeping top {n_keep} ({keep_fraction:.0%})")

    # Keep edges at or above the (1 - keep_fraction) quantile
    threshold = np.quantile(nonzero_weights, 1 - keep_fraction)
    edge_weights_thresh = np.where((network >= threshold) & (network > 0), network, 0)

    # Edges to draw and their line widths
    i_idx, j_idx = np.nonzero(np.triu(edge_weights_thresh, k=1))
    w = edge_weights_thresh[i_idx, j_idx]
    w_min, w_max = w.min(), w.max()
    if w_max > w_min:
        linewidths = lw_min + (w - w_min) / (w_max - w_min) * (lw_max - lw_min)
    else:
        linewidths = np.full_like(w, lw_min, dtype=float)

    segments = np.stack([atlas_coords[i_idx, :2], atlas_coords[j_idx, :2]], axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')
    ax.set_rasterization_zorder(0)

    ax.scatter(lh_vertices[:, 0], lh_vertices[:, 1], c='white', s=1, alpha=0.02, zorder=-1)
    ax.scatter(rh_vertices[:, 0], rh_vertices[:, 1], c='white', s=1, alpha=0.02, zorder=-1)
    ax.scatter(atlas_coords[:, 0], atlas_coords[:, 1], s=node_size, c='white',
               edgecolors='none', zorder=-1)

    ax.add_collection(LineCollection(segments, colors=edge_color, alpha=edge_alpha,
                                     linewidths=linewidths, zorder=-1))

    ax.axis('off')
    ax.set_aspect('equal')

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, filename), dpi=600,
                    bbox_inches='tight', facecolor='black')
    plt.show()

def plot_average_connectome(connectomes, lobes, cmap='Greens', vmin=0, vmax=0.01,
                            title="Average Connectome Matrix", label_fontsize=15,
                            line_color='gray', save_dir=None,
                            filename="average_matrix.pdf", figsize=(8, 6), show=True):
    " Plot the averaged connectivity matrix. "

    mean_connectome = np.mean(connectomes, axis=2)
    nnodes = mean_connectome.shape[0]
    lobes = np.asarray(lobes)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_connectome, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=False, ax=ax, rasterized=True)

    # Lobe tick positions and boundaries
    ticks, labels, boundaries = [], [], set()
    for lobe in np.unique(lobes):
        idx = np.where(lobes == lobe)[0]
        first, last = idx[0], idx[-1]
        ticks.append((first + last + 1) / 2)  # centre of the block
        labels.append(lobe)
        boundaries.update([first, last + 1])

    boundaries = sorted(boundaries)
    ax.hlines(boundaries, xmin=0, xmax=nnodes, colors=line_color, linewidth=1)
    ax.vlines(boundaries, ymin=0, ymax=nnodes, colors=line_color, linewidth=1)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=0, fontsize=label_fontsize)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=label_fontsize)
    ax.tick_params(axis='both', pad=5)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight')

    plt.show()

############################################### FUNCTIONS FOR D_run_PLSes #############################################

def run_pls_svd(X, Y, n_components=3):
    "Fit PLS-SVD model"

    # Ensure data is set up correctly
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # Run model
    svd_model = PLSSVD(n_components=n_components)
    svd_model.fit(X_scaled, Y_scaled)
    # Get scores (projections of original data onto latent variables)
    x_scores, y_scores = svd_model.transform(X_scaled, Y_scaled)

    return svd_model, x_scores, y_scores

def pls_permutations(X, Y, n_perms=1000, random_state=42):
    "Permutation test on singular values of XᵀY to assess significance of PLS LVs"

    # Set seed
    rng = np.random.default_rng(random_state)

    # Ensure data is set up correctly
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # Calculate cross-covariance matrix between X and Y
    obs_covariance = X_scaled.T @ Y_scaled

    # Run SVD
    obs_singular_vals = svd(obs_covariance, compute_uv=False)

    # Initialize permutation values
    perms = np.zeros((n_perms, len(obs_singular_vals)))
    # Loop through permutations
    for i in trange(n_perms, desc="Permutations"):
        # Permute Y
        idx = rng.permutation(Y.shape[0])
        # Calculate cross-covariance
        perm_covariance = X_scaled.T @ Y_scaled[idx]
        # Run SVD and store
        perms[i] = svd(perm_covariance, compute_uv=False)

    # Calculate signficance based on how many times permutated values are >= observed
    pvals = np.array([
        (1 + np.sum(perms[:, j] >= obs_singular_vals[j])) / (1 + n_perms)
        for j in range(len(obs_singular_vals))
    ])

    return obs_singular_vals, perms, pvals

def pls_bootstraps(X, Y, n_components,which="x",n_boot=1000, random_state=42, n_jobs=-1, desc=None):
    "Bootstrap estimation of brain salience reliability (BSRs) for PLS weights"

    # Ensure 'which' was defined properly 
    if which.lower() not in ("x", "y"):
        raise ValueError("`which` must be either 'x' or 'y'")

    # Ensure data is set up correctly
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Number of participants
    n_subjects = X.shape[0]

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # Run PLS on original data
    obs_model, _, _ = run_pls_svd(X_scaled, Y_scaled, n_components)

    # Pull weights from SVD model
    obs_x_weights = obs_model.x_weights_
    obs_y_weights = obs_model.y_weights_

    # Select weights based on if we are bootstrapping X or Y
    obs_weights = obs_x_weights if which.lower() == "x" else obs_y_weights
    obs_weights = np.atleast_2d(obs_weights)

    # Ensure shape is (features, latent_vars)
    if obs_weights.shape[0] == 1:
        obs_weights = obs_weights.T  

    # Pre-generate seeds for parallel boostrapping
    master_rng = np.random.default_rng(random_state)
    seeds = master_rng.integers(0, 1_000_000_000, size=n_boot)

    def single_boot(seed):

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_subjects, size=n_subjects)

        # Standardize data
        Xb = StandardScaler().fit_transform(X[idx])
        Yb = StandardScaler().fit_transform(Y[idx])

        model, _, _ = run_pls_svd(Xb, Yb, n_components)

        boot_weights = (
            model.x_weights_
            if which.lower() == "x"
            else model.y_weights_
        )

        boot_weights = np.atleast_2d(boot_weights)
        if boot_weights.shape[0] == 1:
            boot_weights = boot_weights.T

        n_features, n_comps = boot_weights.shape

        # Sign indeterminacy in PLS bootstrapping is discussed in McIntosh & Lobaugh
        # (2004, NeuroImage). Procrustes alignment is used when the weights matrix is
        # sufficiently overdetermined (n_features / n_comps > 5). For near-square
        # matrices, simple sign-flipping per component is used as a conservative
        # fallback, as Procrustes solutions become sensitive to small perturbations
        # in low-dimensional spaces.
        use_procrustes = (n_features / n_comps) > 5

        if use_procrustes:
            R, _ = orthogonal_procrustes(boot_weights, obs_weights)
            boot_weights = boot_weights @ R
        else:
            # For low-ratio cases (e.g. Y with 3 variables and 2 components),
            # fall back to simple column-wise sign flipping. Each bootstrap component
            # is flipped if its dot product with the observed component is negative,
            # meaning they point in opposite directions and need to be aligned.
            for col in range(n_comps):
                if np.dot(boot_weights[:, col], obs_weights[:, col]) < 0:
                    boot_weights[:, col] *= -1

        return boot_weights

    # Parallel execution of boostraps
    with tqdm_joblib(tqdm(total=n_boot, desc=desc)):
        weights_boot = Parallel(
            n_jobs=n_jobs
        )(
            delayed(single_boot)(seed)
            for seed in seeds
        )

    weights_boot = np.stack(weights_boot)

    # Compute standard error across bootstraps
    weights_se = weights_boot.std(axis=0, ddof=1)
    weights_se[weights_se == 0] = np.inf             # Prevent division by zero

    # Compute bootstrap ratio
    weights_bsr = obs_weights / weights_se

    return obs_weights, weights_bsr, weights_boot, weights_se

def plot_permutation_results(S_obs, perms, pvals, n_plot='all', name='variable',save_path=None):
    "Plot null distrubtion and observed observed singular values."

    n_LVs = len(S_obs) if n_plot == 'all' else min(n_plot, len(S_obs))

    n_cols = math.ceil(math.sqrt(n_LVs))
    n_rows = math.ceil(n_LVs / n_cols)

    # Plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for lv in range(n_LVs):
        ax = axes[lv]

        ax.hist(perms[:, lv], bins=30,
                color='lightgray', edgecolor='white', linewidth=0.4, alpha=0.9)

        # Colour the observed line red if significant, dark grey otherwise
        line_color = '#b01030' if pvals[lv] < 0.05 else '#555555'
        ax.axvline(S_obs[lv], color=line_color, linewidth=7)

        #  remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # no axis labels and no tick marks/labels on either axis
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused axes
    for i in range(n_LVs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    for coll in ax.collections:
        coll.set_rasterized(True)

    if save_path is not None:
            label = 'all' if n_plot == 'all' else str(n_plot)
            plt.savefig(os.path.join(save_path, f'{name}_perm_LV_{label}.pdf'),
                        dpi=600, bbox_inches='tight', facecolor='white')

    plt.show()

def plot_X_bsr(bsr, lv=1, threshold=2.0, feature_names=None,name='bootstrap', save_path=None):
    "Plots bootstrap ratios for each variable"

    x = np.arange(len(bsr))
    stable = np.abs(bsr) > threshold

    print(f"Number of stable variables: {np.sum(stable)} out of {len(bsr)}")

    # Set parameters
    FONT_SIZE_LABEL = 18
    FONT_SIZE_TICK  = 15

    fig, ax = plt.subplots(figsize=(max(8, len(bsr) * 0.6) if len(bsr) <= 50 else 16, 5))

    if len(bsr) > 50:
        # Use vlines for large datasets 
        ax.vlines(x[stable],  0, bsr[stable],  color="#4B8056",   linewidth=0.2)
    else:
        # Use bars for small datasets 
        colors = np.where(stable, '#b01030', 'lightgray')
        ax.bar(x, bsr, color=colors, edgecolor='white', linewidth=0.4, alpha=0.9)

    ax.axhline( threshold, color='black', linewidth=2, linestyle='--')
    ax.axhline(-threshold, color='black', linewidth=2, linestyle='--')

    if feature_names is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=90)

    ax.set_ylabel("Bootstrap Ratio", fontsize=FONT_SIZE_LABEL, labelpad=8)

    #  remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_yticklabels(['-4', '-2', '0', '2', '4'],fontsize=FONT_SIZE_TICK+1)
    # x ticks at 0, 5k, 10k, 15k, 20k with comma formatting
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xticklabels(['0', '5,000', '10,000', '15,000', '20,000'],fontsize=FONT_SIZE_TICK+1)

    # no axis labels
    ax.set_xlabel("")

    # clamp y axis
    ax.set_ylim(-5, 5)

    plt.tight_layout()

    for coll in ax.collections:
        coll.set_rasterized(True)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{name}_LV{lv}.pdf'),
                    dpi=600, bbox_inches='tight', facecolor='white')

    plt.show()

def plot_Y_bsr(bsr, lv=1, threshold=2.0, feature_names=None,name='bootstrap', save_path=None):
    "Plots bootstrap ratios for each variable"

    # Define colors for domains
    colors = {
    'Adversity':     '#8B0000',   
    'Household/Community Instability':     '#8B0000',   
    'Physical/Sexual Abuse':     '#8B0000',
    'Parental Neglect':     '#8B0000',   
    'Financial Difficulties':     '#8B0000',  
    'Mental Health': '#1F6FB2',   
    'Genetics':      '#D9A300',  
    'Overlap':       '#7B3294',   
    'Cognition': "#17BDAD"
    }

    x = np.arange(len(bsr))
    stable = np.abs(bsr) > threshold

    print(f"Number of stable variables: {np.sum(stable)} out of {len(bsr)}")

    # Set parameters
    FONT_SIZE_LABEL = 18
    FONT_SIZE_TICK  = 15

    fig, ax = plt.subplots(figsize=(max(8, len(bsr) * 0.2) if len(bsr) <= 50 else 16, 5))

    if 'A' in name:
        c = colors['Adversity']
    else:
        c = colors['Genetics']

    if len(bsr) > 50:
        # Use vlines for large datasets 
        #ax.vlines(x[~stable], 0, bsr[~stable], color='lightgray', linewidth=0.1, alpha=0.7)
        ax.vlines(x[stable],  0, bsr[stable],  color=c,   linewidth=0.2)
    else:
        # Use bars for small datasets 
        colors = np.where(stable, c, 'lightgray')
        ax.bar(x, bsr, color=colors, edgecolor='white', linewidth=0.4, alpha=0.9)

    ax.axhline( threshold, color='black', linewidth=2, linestyle='--')
    ax.axhline(-threshold, color='black', linewidth=2, linestyle='--')

    if feature_names is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=90)

    ax.set_ylabel("Bootstrap Ratio", fontsize=FONT_SIZE_LABEL, labelpad=8)
    #  remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([-25, -15, 0, 15, 25])
    ax.set_yticklabels(['-25', '-15', '0', '15', '25'],fontsize=FONT_SIZE_TICK+1)
    # no axis labels
    ax.set_xlabel("")
    ax.set_xticklabels(feature_names, rotation=45, ha='right', rotation_mode='anchor')

    # clamp y axis
    ax.set_ylim(-26, 26)

    plt.tight_layout()

    for coll in ax.collections:
        coll.set_rasterized(True)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{name}_LV{lv}.pdf'),
                    dpi=600, bbox_inches='tight', facecolor='white')

    plt.show()

############################################### FUNCTIONS FOR E_overlap_analysis #############################################
  
def run_cross_domain_glm(pls_x_weights, subsample_pls_x_weights,
                         associated_data, subsample_associated_data,
                         covariates, subsample_index,
                         zero_mask, genetics_zero_mask, n_edges,
                         threshold=2):
    """ Run GLMs looking at relationtions across PLS scores, specifically,
    connectivity signature scores (X) and domain scores (Y)
    """
    ### Standardised LV scores ###
    x_scores = {
        'Adversity':     StandardScaler().fit_transform(associated_data['X_adversity_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Mental Health': StandardScaler().fit_transform(associated_data['X_MH_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Genetics LV1':  StandardScaler().fit_transform(subsample_associated_data['X_gene_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Genetics LV2':  StandardScaler().fit_transform(subsample_associated_data['X_gene_LV2_score'].values.reshape(-1, 1)).flatten(),
    }
    y_scores = {
        'Adversity':     StandardScaler().fit_transform(associated_data['Y_adversity_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Mental Health': StandardScaler().fit_transform(associated_data['Y_MH_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Genetics LV1':  StandardScaler().fit_transform(subsample_associated_data['Y_gene_LV1_score'].values.reshape(-1, 1)).flatten(),
        'Genetics LV2':  StandardScaler().fit_transform(subsample_associated_data['Y_gene_LV2_score'].values.reshape(-1, 1)).flatten(),
    }

    x_names = list(x_scores.keys())
    y_names = list(y_scores.keys())
    n_x, n_y = len(x_names), len(y_names)
    beta_matrix = np.zeros((n_x, n_y))
    pval_matrix = np.zeros((n_x, n_y))

    formula = ('y ~ x + age + cognition + C(sex) + dmri_meanmotion + C(mrif_score) + '
               'C(site) + C(manufacturer_code) + C(softwareversion_code)')

    ### Run GLMs across LVs of all 4 PLSes ###
    print(f"\n{'':25} ", end='')
    for y_name in y_names:
        print(f"{y_name:>14}", end='')
    print()
    print('-' * (26 + 14 * n_y))

    for i, (x_name, x_data) in enumerate(x_scores.items()):
        print(f"{x_name:25}", end=' ')

        for j, (y_name, y_data) in enumerate(y_scores.items()):
            is_pgs_x = 'Genetics' in x_name
            is_pgs_y = 'Genetics' in y_name

            if is_pgs_x and is_pgs_y:
                # Both already in subsample space
                x, y = x_data, y_data
                covs = covariates.loc[subsample_index]
            elif is_pgs_x:
                # X is subsampled, Y is full -> subset Y and covs
                x, y = x_data, y_data[subsample_index]
                covs = covariates.loc[subsample_index]
            elif is_pgs_y:
                # Y is subsampled, X is full -> subset X and covs
                x, y = x_data[subsample_index], y_data
                covs = covariates.loc[subsample_index]
            else:
                # Neither involves PGS -> full sample
                x, y = x_data, y_data
                covs = covariates

            glm_df = pd.DataFrame({
                'y': y,
                'x': x,
                'age': covs['age'].values.astype(float),
                'cognition': covs['cognitive_score'].values.astype(float),
                'sex': covs['sex'].values.astype(str),
                'dmri_meanmotion': covs['dmri_meanmotion'].values.astype(float),
                'mrif_score': covs['mrif_score'].values.astype(float),
                'site': covs['site'].values.astype(str),
                'manufacturer_code': covs['manufacturer_code'].values.astype(str),
                'softwareversion_code': covs['softwareversion_code'].values.astype(str),
            })

            model = glm(formula=formula, data=glm_df, family=sm.families.Gaussian()).fit()

            b = model.params['x']
            p = model.pvalues['x']
            beta_matrix[i, j] = b
            pval_matrix[i, j] = p

            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  β={b:>6.3f} {sig:>3}", end='')
        print()

    ### Exact p-value table ###
    print("\nExact p-values:")
    print(f"{'':25} ", end='')
    for y_name in y_names:
        print(f"{y_name:>14}", end='')
    print()
    print('-' * (26 + 14 * n_y))

    for i, x_name in enumerate(x_names):
        print(f"{x_name:25}", end=' ')
        for j in range(n_y):
            print(f"{pval_matrix[i, j]:>14.4g}", end='')
        print()

    return {
        'beta_matrix': pd.DataFrame(beta_matrix, index=x_names, columns=y_names),
        'pval_matrix': pd.DataFrame(pval_matrix, index=x_names, columns=y_names)
        }

def extract_masks(*matrices, names=None, flip=None):
    " Compute pairwise overlaps between 2 or 3 PLS edge-weight matrices."

    # Validate number of matrices
    n_masks = len(matrices)
    if not 2 <= n_masks <= 3:
        raise ValueError(
            f"extract_masks requires between 2 and 3 matrices, got {n_masks}."
        )
 
    # Default names if not provided
    if names is None:
        names = [f"Mask{i + 1}" for i in range(n_masks)]
    elif len(names) != n_masks:
        raise ValueError(
            f"Number of names ({len(names)}) must match number of matrices ({n_masks})."
        )
 
    flip = [] if flip is None else list(flip)
    unknown = [f for f in flip if f not in names]
    if unknown:
        raise ValueError(f"Names in `flip` not found in `names`: {unknown}")
 
    # Presence masks (edge is non-zero) and sign-adjusted direction (-1/0/+1)
    masks = [np.asarray(m) != 0 for m in matrices]
    signs = [np.sign(np.asarray(m)) * (-1 if names[i] in flip else 1)
             for i, m in enumerate(matrices)]
 
    # Initialise output dictionaries
    overlap_masks         = {}
    aligned_overlap_masks = {}
    overlap_percentages   = {}
 
    # Helper: count edges in a symmetric matrix (upper triangle only)
    def n_edges(m):
        return np.count_nonzero(m) // 2
 
    # Helper: overlap and sign-aligned overlap across the given indices
    def overlaps(indices):
        present = np.logical_and.reduce([masks[k] for k in indices])
        first = signs[indices[0]]
        same_sign = np.logical_and.reduce([signs[k] == first for k in indices])
        aligned = np.where(present & same_sign, first, 0).astype(int)
        return present, aligned
 
    # Helper: store results and print a summary line for one overlap
    def record(indices, key, pct_key):
        present, aligned = overlaps(indices)
        overlap_masks[key] = present.astype(int)
        aligned_overlap_masks[key] = aligned
 
        shared    = n_edges(present)
        n_aligned = n_edges(aligned)
        n_risk    = n_edges(aligned > 0)
        n_health  = n_edges(aligned < 0)
        aligned_pct = (n_aligned / shared * 100) if shared > 0 else 0
 
        pcts = {k: (shared / n_edges(masks[k]) * 100) if n_edges(masks[k]) > 0 else 0
                for k in indices}
 
        overlap_percentages[pct_key] = {
            f"% overlap of {names[k]}": pcts[k] for k in indices
        } | {
            "shared_edges": shared,
            "aligned_edges": n_aligned,
            "risk_aligned_edges": n_risk,
            "health_aligned_edges": n_health,
            "% of shared edges aligned": aligned_pct,
        }
 
        header = " & ".join(f"{names[k]} ({n_edges(masks[k]):,} edges)" for k in indices)
        print(f"{header}:")
        print(f"    {shared:,} overlapping edges, {n_aligned:,} shared edges grouped into: "
              f"{n_risk:,} Risk-aligned, {n_health:,} Health-aligned",
              "\n")
 
    ############################### Pairwise overlaps ##################################
 
    for (i, j) in itertools.combinations(range(n_masks), 2):
        record([i, j], f"{names[i]} & {names[j]}", f"{names[i]} vs {names[j]}")
 
    return overlap_masks, aligned_overlap_masks, overlap_percentages

def bsr_permutations_overlap(target, others, nnodes, n_edges,
                             n_perms=1000, random_state=42, threshold=2):
    """
    Permute bootstrap ratios and assess if Jaccard index is signfiicantly higher
    than expected at chance.
    """
    if not (1 <= len(others) <= 3):
        raise ValueError(f"`others` must have 1 to 3 entries, got {len(others)}.")

    entries = [target] + list(others)
    rng = np.random.default_rng(random_state)

    def n_edges_in(m):
        "Edge count of a symmetric matrix (upper triangle only)."
        return np.count_nonzero(m) // 2

    def bsr_values(bsr):
        return np.asarray(bsr.values if hasattr(bsr, 'values') else bsr)

    def jaccard(masks):
        """Jaccard between target (first) and the intersection of the rest.
        Returns (jaccard, n_comparison, n_shared)."""
        target_m = masks[0]
        comparison = np.logical_and.reduce(masks[1:])
        n_shared = n_edges_in(target_m & comparison)
        n_union = n_edges_in(target_m | comparison)
        return (n_shared / n_union if n_union > 0 else 0.0), n_edges_in(comparison), n_shared

    def extract_boot_mask_binary(bsr, nnodes, n_edges, zero_mask, threshold=2):
        """Extract binary mask of stable edges based on BSR threshold."""
        
        bsr_mask = (np.abs(bsr) > threshold).astype(float)
        matrix = convert_to_matrix(bsr_mask, nnodes, n_edges, zero_mask)
        mask = (matrix != 0).astype(int)

        return mask
    
    # Observed 
    obs_masks = [np.asarray(e['matrix']) != 0 for e in entries]
    obs_jaccard, n_comparison, obs_shared = jaccard(obs_masks)

    # Null distribution: permute every BSR vector independently
    bsrs = [bsr_values(e['bsr']) for e in entries]
    perms = np.zeros(n_perms)
    for i in trange(n_perms, desc="Permutations"):
        perm_masks = [
            extract_boot_mask_binary(b[rng.permutation(len(b))],
                                     nnodes, n_edges, e['zero_mask'], threshold).astype(bool)
            for b, e in zip(bsrs, entries)
        ]
        perms[i] = jaccard(perm_masks)[0]

    # Two-tailed p-value 
    p_upper = (1 + np.sum(perms >= obs_jaccard)) / (1 + n_perms)
    p_lower = (1 + np.sum(perms <= obs_jaccard)) / (1 + n_perms)
    pval = min(2 * min(p_upper, p_lower), 1.0)

    # Print out results
    null_mean = np.mean(perms)
    other_names = " & ".join(e['name'] for e in others)
 
    print(" & ".join(f"{e['name']} ({n_edges_in(m):,} edges)"
                     for e, m in zip(entries, obs_masks)))
    if len(others) > 1:
        print(f"    {other_names} intersection: {n_comparison:,} edges")
    print(f"    {obs_shared:,} overlapping edges")
    print(f"    Jaccard: observed = {obs_jaccard:.4f}, null mean = {null_mean:.4f}")
    if pval < 0.05:
        direction = "higher" if obs_jaccard > null_mean else "lower"
        print(f"    p = {pval:.4f} → overlap is significantly {direction} than chance")
        print("\n")
    else:
        print(f"    p = {pval:.4f} → overlap is not significant")
        print("\n")

def plot_aligned_edge_example(aligned_mask, matrices, matrix_names, plot_args,
                              risk_edge=None, health_edge=None,
                              save_name='aligned_edge_example.pdf', flip=None):
    "Plot two specific edges as examples of risk-aligned and health-aligned edges. "

    pos_color = "#184d18"   # Risk-aligned
    neg_color = "#a73685"   # Health-aligned

    region_labels = plot_args['region_labels']
    atlas_coords = plot_args['atlas_coords']
    lh_vertices = plot_args['lh_vertices']
    rh_vertices = plot_args['rh_vertices']
    save_path = plot_args['save_path']

    if risk_edge is None and health_edge is None:
        raise ValueError("Give at least one of risk_edge or health_edge.")
    flip = [] if flip is None else list(flip)

    aligned_mask = np.asarray(aligned_mask)
    iu = np.triu_indices(aligned_mask.shape[0], k=1)
    alignment = aligned_mask[iu]

    # Sign-adjusted combined weight (flipped matrices reversed so all point the same way)
    combined = sum(np.asarray(m) * (-1 if name in flip else 1)
                   for name, m in zip(matrix_names, matrices))
    combined_label = " ".join(
        (('-' if name in flip else '+') if k else ('-' if name in flip else '')) + name
        for k, name in enumerate(matrix_names)).replace('+', '+ ').replace(' -', ' - ')

    # Region name -> node (region_labels is 0-indexed in matrix order; spaces ignored)
    node_of = {str(name).replace(' ', ''): i for i, name in enumerate(region_labels)}

    def find_node(name):
        key = name.replace(' ', '')
        if key not in node_of:
            raise KeyError(f"Region '{name}' not found in region_labels.")
        return node_of[key]

    # Look up and describe each chosen edge
    edges = []
    for wanted, label, color in [(1, risk_edge, pos_color), (-1, health_edge, neg_color)]:
        if label is None:
            continue
        kind = 'Risk-aligned' if wanted > 0 else 'Health-aligned'
        i, j = find_node(label[0]), find_node(label[1])
        actual = aligned_mask[i, j]

        print(f"\n{kind} edge (chosen): node {i} <-> node {j}")
        print(f"Regions: {region_labels[i]}  <->  {region_labels[j]}")

        if actual == 0:
            print("WARNING: this edge is not an aligned edge in this mask.")
        else:
            actual_kind = 'Risk-aligned' if actual > 0 else 'Health-aligned'
            if actual != wanted:
                print(f"WARNING: this edge is {actual_kind}, not {kind}.")
            # Rank / percentile among aligned edges of the edge's actual direction
            same = np.abs(combined[iu][alignment == actual])
            w = abs(combined[i, j])
            rank = int((same > w).sum()) + 1
            pct = (same <= w).mean() * 100
            print(f"Rank {rank} of {same.size} {actual_kind} edges ({pct:.1f}th percentile)")

        print(f"Combined weight ({combined_label}): {combined[i, j]:.4f}")
        for name, m in zip(matrix_names, matrices):
            print(f"  {name}: {m[i, j]:.4f}")

        edges.append((i, j, color))

    # Plot the chosen edges and their endpoint nodes
    node_size = 300
    edge_width = 9

    plt.figure(figsize=(10, 8))
    plt.scatter(lh_vertices[:, 1], lh_vertices[:, 2], c='grey', s=1, alpha=0.08, rasterized=True)
    plt.scatter(rh_vertices[:, 1], rh_vertices[:, 2], c='grey', s=1, alpha=0.08, rasterized=True)

    for i, j, color in edges:
        plt.plot([atlas_coords[i, 1], atlas_coords[j, 1]],
                 [atlas_coords[i, 2], atlas_coords[j, 2]],
                 color=color, alpha=0.9, linewidth=edge_width,
                 solid_capstyle='round', zorder=4, rasterized=True)
        plt.scatter([atlas_coords[i, 1], atlas_coords[j, 1]],
                    [atlas_coords[i, 2], atlas_coords[j, 2]],
                    s=node_size, c=color, alpha=1, edgecolors='black',
                    linewidths=1, zorder=5, rasterized=True)

    plt.axis('off')
    plt.gca().set_aspect('equal')

    # Legend: only the edge types plotted
    legend = []
    if risk_edge is not None:
        legend.append(Line2D([0], [0], color=pos_color, lw=edge_width, marker='o',
                             markersize=12, markeredgecolor='black', label='Risk-aligned'))
    if health_edge is not None:
        legend.append(Line2D([0], [0], color=neg_color, lw=edge_width, marker='o',
                             markersize=12, markeredgecolor='black', label='Health-aligned'))
    plt.legend(handles=legend, loc='upper right', frameon=False)

    plt.savefig(os.path.join(save_path, save_name), dpi=600, bbox_inches='tight')
    plt.show()

def edge_group_node_plot(aligned_mask, weight_matrix, plot_args, save_name='node_plot.pdf',
                         keep_fraction=0.8, group='', type='Both',connectomes=None):
    "Node plot of sign-aligned overlapping edges."
 
    # Set up plotting variables
    pos_label = 'Risk-aligned'
    neg_label = 'Health-aligned'
    pos_color = "#184d18"   # Risk-aligned
    neg_color = "#a73685"   # Health-aligned
    atlas_coords = plot_args['atlas_coords']
    lh_vertices = plot_args['lh_vertices']
    rh_vertices = plot_args['rh_vertices']
    save_path = plot_args['save_path']
 
    plot_type = str(type).capitalize()
    if plot_type not in ('Both', 'Risk', 'Health'):
        raise ValueError("type must be 'Both', 'Risk' or 'Health'")
    show_pos = plot_type in ('Both', 'Risk')
    show_neg = plot_type in ('Both', 'Health')
 
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
 
    aligned_mask = np.asarray(aligned_mask)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if aligned_mask.shape != weight_matrix.shape:
        raise ValueError(f"aligned_mask {aligned_mask.shape} and weight_matrix "
                         f"{weight_matrix.shape} must have the same shape.")
 
    nnodes = aligned_mask.shape[0]
    iu = np.triu_indices(nnodes, k=1)
 
    # Direction from the aligned mask, magnitude from the weight matrix
    direction = aligned_mask[iu]
    magnitude = np.abs(weight_matrix[iu])
 
    # Keep only the requested direction(s)
    selected = ((direction > 0) & show_pos) | ((direction < 0) & show_neg)
    nz = np.flatnonzero(selected)
    n_total = nz.size
    if n_total == 0:
        raise ValueError(f"No edges for type='{plot_type}' — nothing to plot.")
    if np.any(magnitude[nz] == 0):
        print(f"Warning: {np.sum(magnitude[nz] == 0)} aligned edges have zero weight "
              f"in weight_matrix and will be drawn at minimum width.")
 
    # ---- keep exactly the top n_keep edges by magnitude ----
    n_keep = max(1, int(round(keep_fraction * n_total)))
    order = np.argsort(-magnitude[nz], kind='stable')
    keep_idx = nz[order[:n_keep]]
 
    rows, cols = iu[0][keep_idx], iu[1][keep_idx]
    signs = direction[keep_idx]
    mags = magnitude[keep_idx]
    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
 
    prefix = f"{group} " if group else ""
    kind = {'Both': 'Aligned', 'Risk': pos_label, 'Health': neg_label}[plot_type]
    counts = []
    if show_pos:
        counts.append(f"{n_pos:,} {pos_label}")
    if show_neg:
        counts.append(f"{n_neg:,} {neg_label}")
    print(f"{prefix}{kind} edges: {n_total:,} -> keeping top {n_keep:,} "
          f"({keep_fraction:.0%}): {', '.join(counts)}")

    # Set line width (default is PLS weight, but if connectomes are provided then it will be average edge weight)
    if connectomes is None:
        # From |weight_matrix|, linear 1-9, one scale shared by both signs
        widths_from = mags
        line_alpha = 0.7
        w_min, w_max = widths_from.min(), widths_from.max()
        lws = (1.0 + (widths_from - w_min) / (w_max - w_min) * 8.0
               if w_max > w_min else np.full(widths_from.shape, 1.0))
    else:
        # From mean |connectivity| across subjects (per-sign scaling)
        connectomes = np.asarray(connectomes)
        if connectomes.shape[:2] != (nnodes, nnodes):
            raise ValueError(f"connectomes must be (nnodes, nnodes, nsubj); "
                             f"got {connectomes.shape}")
        mean_abs = np.mean(np.abs(connectomes), axis=2)
        mean_abs = np.maximum(mean_abs, mean_abs.T)   
        widths_from = mean_abs[rows, cols]
        line_alpha = 1.0
        LW_MIN, LW_MAX = 0.3, 8.0
        lws = np.zeros(widths_from.shape)
        for sign_mask in (signs > 0, signs < 0):
            if not sign_mask.any():
                continue
            w = widths_from[sign_mask]
            w_min, w_max = w.min(), w.max()
            t = (w - w_min) / (w_max - w_min) if w_max > w_min else np.ones(w.shape)
            lws[sign_mask] = LW_MIN + t * (LW_MAX - LW_MIN)
        n_zero_conn = int((widths_from == 0).sum())
        if n_zero_conn:
            print(f"    {n_zero_conn} edges have zero mean connectivity; drawn at minimum width")
 
 
    w_min, w_max = mags.min(), mags.max()
 
    plt.figure(figsize=(10, 8))
    plt.scatter(lh_vertices[:, 1], lh_vertices[:, 2], c='grey', s=1, alpha=0.08, rasterized=True)
    plt.scatter(rh_vertices[:, 1], rh_vertices[:, 2], c='grey', s=1, alpha=0.08, rasterized=True)
    plt.scatter(atlas_coords[:, 1], atlas_coords[:, 2], s=50, c='darkgray',
                edgecolors='none', rasterized=True)
    
    # draw weakest first so the strongest edges sit on top
    for k in np.argsort(widths_from, kind='stable'):
        i, j, s = rows[k], cols[k], signs[k]
        plt.plot([atlas_coords[i, 1], atlas_coords[j, 1]],
                 [atlas_coords[i, 2], atlas_coords[j, 2]],
                 color=pos_color if s > 0 else neg_color,
                 alpha=line_alpha, linewidth=lws[k], rasterized=True)
 
    plt.axis('off')
    plt.gca().set_aspect('equal')
 
    handles = []
    if show_pos:
        handles.append(Line2D([0], [0], color=pos_color, lw=3, label=pos_label))
    if show_neg:
        handles.append(Line2D([0], [0], color=neg_color, lw=3, label=neg_label))
    plt.legend(handles=handles, loc='upper right', frameon=False)
 
    plt.savefig(os.path.join(save_path, save_name), dpi=600, bbox_inches='tight')
    plt.show()
 
def get_edge_group_scores(connectomes, aligned_overlap_masks, target_sets=None):
    """
    Create participant scores per edge group.

    Edge group metrics:
    1) n_edges   : number of edges in the set (Risk-aligned + Health-aligned)
    2) n_pos     : number of Risk-aligned (+1) edges in the set
    3) n_neg     : number of Health-aligned (-1) edges in the set

    Participant metrics:
    1) score              : mean connectivity across the group = sum(risk-aligned edge weights) - sum(health-aligned edge weights) / n_edges
    2) mean_pos (or neg)  : average connection weight within risk-aligned (pos) or health-aligned (neg) groups
    3) count_pos (or neg) : total number of risk-aligned (pos) or health-aligned (neg) edges 

    """
    connectomes = np.asarray(connectomes)
    nnodes = connectomes.shape[0]
    if connectomes.shape[:2] != (nnodes, nnodes):
        raise ValueError(f"connectomes must be (nnodes, nnodes, n_subjects); got {connectomes.shape}")

    # Upper-triangle edges only (each edge once); subjects x edges
    iu = np.triu_indices(nnodes, k=1)
    edge_data = connectomes[iu[0], iu[1], :].T

    scores = {}
    details = {}

    for set_label, mask in aligned_overlap_masks.items():

        # Skip any edge set not in the requested list
        if target_sets is not None and set_label not in target_sets:
            continue

        mask = np.asarray(mask)
        if mask.shape != (nnodes, nnodes):
            raise ValueError(f"'{set_label}' mask is {mask.shape}; expected ({nnodes}, {nnodes}).")

        # Direction of each edge straight from the aligned mask: +1 / -1 / 0
        signs = np.sign(mask[iu])
        pos_edges = signs > 0
        neg_edges = signs < 0

        # ---- Per-set edge counts ----
        n_pos = int(pos_edges.sum())
        n_neg = int(neg_edges.sum())
        n_edges = n_pos + n_neg
        if n_edges == 0:
            print(f"{set_label}: no aligned edges — skipped.\n")
            continue

        # ---- Main score: mean signed connectivity (same denominator for everyone) ----
        score = edge_data @ signs / n_edges

        # ---- Per-sign means over non-zero connections only ----
        pos_block = edge_data[:, pos_edges]
        neg_block = edge_data[:, neg_edges]
        count_pos = (pos_block != 0).sum(axis=1)
        count_neg = (neg_block != 0).sum(axis=1)

        # Zeros add nothing to the sum, so sum / non-zero count = mean over non-zero entries
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_pos = np.where(count_pos > 0, pos_block.sum(axis=1) / count_pos, np.nan)
            mean_neg = np.where(count_neg > 0, neg_block.sum(axis=1) / count_neg, np.nan)

        scores[set_label] = score
        details[set_label] = {
            "n_edges":   n_edges,
            "n_risk":     n_pos,
            "n_health":     n_neg,
            "score":     score,
            "weight_risk":  mean_pos,
            "weight_health":  mean_neg,
            "count_risk": count_pos,
            "count_health": count_neg,
        }

    scores = pd.DataFrame(scores)

    return scores, details

def edge_group_plots(connectomes, aligned_overlap_masks, edge_group_scores, save_path,
                             group_labels=None, participant=6281):
    """
    Summary plots for edge groups:
    1) Plots a heatmap for one participant (`participant`, default 6281) with one row per edge group.
    2) Plots same participant's connection-weight histogram for each edge group.
    3) Plots the distribution of the summary score across all participants for each
       edge group.
    """
 
    # ---- Settings ----
    POS_COLOR = "#184d18"   # Risk-aligned
    NEG_COLOR = "#a73685"   # Health-aligned
    VLIM      = 0.02        # |weight| at/above this is fully opaque in the heatmap
    ZERO_EPS  = 1e-8        # |weight| <= this counts as "absent"
    SCORE_COLORS = {'M&G Edges': '#3D2A30', 'M&A Edges': '#BEA4AD', 'A&G Edges': '#9A6B79'}
    GROUP_ORDER = ['M&A Edges', 'A&G Edges', 'M&G Edges']   # top -> bottom
    if group_labels is None:
        group_labels = {
            'Mental Health & Genetics':  'M&G Edges',
            'Adversity & Mental Health': 'M&A Edges',
            'Adversity & Genetics':      'A&G Edges',
        }
 
    # Keep only groups that exist in the masks
    groups = {k: v for k, v in group_labels.items() if k in aligned_overlap_masks}
    if not groups:
        raise ValueError("None of the group_labels keys are in aligned_overlap_masks.")
 
    green = np.array(mcolors.to_rgb(POS_COLOR))
    pink = np.array(mcolors.to_rgb(NEG_COLOR))
 
    def colorize(values):
        "Weights -> RGBA: green if > 0, pink if < 0, opacity = |value| / VLIM, white if ~0."
        values = np.asarray(values).ravel()
        rgba = np.ones((values.size, 4))
        alpha = np.clip(np.abs(values) / VLIM, 0, 1)
        pos = values > ZERO_EPS
        neg = values < -ZERO_EPS
        rgba[pos, :3] = green
        rgba[pos, 3] = alpha[pos]
        rgba[neg, :3] = pink
        rgba[neg, 3] = alpha[neg]
        return rgba
 
    connectomes = np.asarray(connectomes)
    nnodes, _, n_subjects = connectomes.shape
    iu = np.triu_indices(nnodes, k=1)
 
    # Risk- and Health-aligned edge indices for each edge group
    group_info = {}
    for key, label in groups.items():
        signs = np.sign(np.asarray(aligned_overlap_masks[key])[iu])
        pos_ids = np.flatnonzero(signs > 0)   # Risk-aligned edges
        neg_ids = np.flatnonzero(signs < 0)   # Health-aligned edges
        group_info[key] = dict(label=label, pos_ids=pos_ids, neg_ids=neg_ids)
 
    # Plotting order: M&A, A&G, M&G 
    ordered_keys = sorted(group_info, key=lambda k: GROUP_ORDER.index(group_info[k]['label'])
                          if group_info[k]['label'] in GROUP_ORDER else len(GROUP_ORDER))
    
    # 1) One participant's heatmap
    pid = int(participant)
    if not 0 <= pid < n_subjects:
        raise ValueError(f"participant {pid} is out of range (0-{n_subjects - 1}).")
    # Vector of selected participant's edges
    subj_vec = connectomes[iu[0], iu[1], pid]       
 
    rows = []
    for key in ordered_keys:
        info = group_info[key]
        order = np.concatenate([info['pos_ids'], info['neg_ids']])
        order_signs = np.concatenate([np.ones(info['pos_ids'].size),
                                      -np.ones(info['neg_ids'].size)])
        rows.append(dict(label=info['label'], n_pos=info['pos_ids'].size,
                         n_edges=order.size,
                         data=subj_vec[order] * order_signs))   # sign-aligned weights
 
    rows = [r for r in rows if r['n_edges'] > 0]
    if rows:
        n_rows = len(rows)
        max_edges = max(r['n_edges'] for r in rows)
 
        LEFT, MAX_W = 0.22, 0.62        # left margin for labels; width of widest row
        TOP, BOTTOM = 0.86, 0.12
        GAP = 0.03                      # vertical gap between rows
        row_h = (TOP - BOTTOM - GAP * (n_rows - 1)) / n_rows
 
        fig = plt.figure(figsize=(14, 10))
        for i, r in enumerate(rows):
            w = MAX_W * r['n_edges'] / max_edges     # row width proportional to edge count
            y = TOP - row_h - i * (row_h + GAP)
            ax = fig.add_axes([LEFT, y, w, row_h])
            ax.imshow(colorize(r['data'])[None, :, :], aspect='auto', interpolation='nearest')
            ax.axvline(r['n_pos'] - 0.5, color='black', lw=0.9, alpha=0.7)  # Risk | Health split
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(-0.015, 0.5, r['label'], transform=ax.transAxes,
                    ha='right', va='center', fontsize=15)
 
        # Legend strip: pink <- 0 -> green, opacity following |weight|
        leg_ax = fig.add_axes([LEFT, 0.05, MAX_W, 0.025])
        leg_ax.imshow(colorize(np.linspace(-VLIM, VLIM, 256))[None, :, :],
                      aspect='auto', extent=[-VLIM, VLIM, 0, 1])
        leg_ax.set_yticks([])
        leg_ax.set_xticks([-VLIM, 0, VLIM])
        leg_ax.tick_params(labelsize=15)
        leg_ax.set_xlabel('Edge Weight', fontsize=20)
        plt.savefig(os.path.join(save_path, 'participant_aligned_heatmaps.pdf'),
                    dpi=600, bbox_inches='tight')
        plt.show()
 
    # 2) Participant's connection-weight histograms per edge group
    for key in ordered_keys:
        info = group_info[key]
        label = info['label']
        short = label.replace('&', '')
        w_pos = subj_vec[info['pos_ids']]
        w_neg = subj_vec[info['neg_ids']]
        w_pos = w_pos[np.abs(w_pos) > ZERO_EPS]      # present connections only
        w_neg = w_neg[np.abs(w_neg) > ZERO_EPS]
 
        both = np.concatenate([w_pos, w_neg])
        if both.size == 0:
            continue
        bins = np.linspace(both.min(), both.max(), 45)
 
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(w_pos, bins=bins, color=POS_COLOR, alpha=0.7, rasterized=True,
                label=f'Risk-aligned ({w_pos.size:,} out of {info["pos_ids"].size:,})')
        ax.hist(w_neg, bins=bins, color=NEG_COLOR, alpha=0.5, rasterized=True,
                label=f'Health-aligned ({w_neg.size:,} out of {info["neg_ids"].size:,})')
        ax.set_xlabel('Edge Weight', fontsize=25)
        ax.set_ylabel('Frequency', fontsize=25)
        ax.tick_params(axis='both', labelsize=20)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=15)
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'{short}_participant_hist.pdf'),
                    dpi=600, bbox_inches='tight')
        plt.show()
 
    # 3) Summary score histograms across participants
    for key in ordered_keys:
        info = group_info[key]
        info = group_info[key]
        label = info['label']
        short = label.replace('&', '')
        if key not in edge_group_scores:
            continue
        score = np.asarray(edge_group_scores[key])
 
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(score, bins=50, color=SCORE_COLORS.get(label, 'grey'), edgecolor='white',
                linewidth=0.5, alpha=0.85, rasterized=True)
        ax.axvline(0, color='red', lw=4, ls='--', label='Zero')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel(f'{label} Score', fontsize=25)
        ax.set_ylabel('Frequency', fontsize=25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=20, loc='upper right')
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'{short}_aligned_score.pdf'),
                    dpi=600, bbox_inches='tight')
        plt.show()

############################################### FUNCTIONS FOR F_mediation_and_sliding_scale #############################################
  
def run_edge_group_mediation(edge_group_scores, associated_data, covariates,
                             group_labels=None, n_boots=1000, seed=42):
    """
    Run mediation analysis using A&G, A&G, and M&G scores as mediator.

    a = adversity -> edge-group score
    b = edge-group score -> mental health (holding adversity constant)
    c = total effect of adversity on mental health
    c' = direct effect of adversity on mental health (holding edge-group score constant)
    
    95% confidence intervals calculated for the indirect effect (a*b) via bootstrapping.
    FDR-correction (Benjamini-Hochberg) applied to p-values across edge groups.

    """

    # Set up parameters
    CI = (2.5, 97.5)   # 95% confidence interval
    CONTINUOUS = ['age', 'dmri_meanmotion', 'cognitive_score']
    CATEGORICAL = ['sex', 'site', 'mrif_score', 'manufacturer_code', 'softwareversion_code']

    if group_labels is None:
        group_labels = {
            'Adversity & Mental Health': 'M&A',
            'Mental Health & Genetics':  'M&G',
            'Adversity & Genetics':      'A&G',
        }

    # Define X (predictor) and Y (outcome)
    X_raw = associated_data['Y_adversity_LV1_score'].values  
    Y_raw = associated_data['Y_MH_LV1_score'].values        

    # Prepare covariates
    missing = [col for col in CONTINUOUS + CATEGORICAL if col not in covariates.columns]
    extra = [col for col in covariates.columns if col not in CONTINUOUS + CATEGORICAL]
    if missing or extra:
        raise ValueError(f"covariates must have exactly {CONTINUOUS + CATEGORICAL}; "
                         f"missing {missing}, unexpected {extra}")

    # Continuous (use as is)
    C_cont = covariates[CONTINUOUS].astype(float).values
    # Categorical (identify variables as a category, not continuous)
    C_cat = pd.get_dummies(covariates[CATEGORICAL].astype(str), drop_first=True,
                           dtype=float).values
    C = np.hstack([C_cont, C_cat])

    n = len(Y_raw)
    if not (len(X_raw) == len(covariates) == len(edge_group_scores) == n):
        raise ValueError(f"Row counts differ: associated_data {n}, covariates {len(covariates)}, "
                         f"edge_group_scores {len(edge_group_scores)}. "
                         f"All must be the same participants in the same order.")


    # ---- Helper: z-score a variable (or every column of a matrix) ----
    def z(v):
        v = np.asarray(v, dtype=float)
        return StandardScaler().fit_transform(v.reshape(len(v), -1))

    # ---- Helper: fit the mediation regressions once and return the path coefficients ----
    def fit_paths(X, M, Y, C):
        XC = np.hstack([X, C])                                    # predictors for paths a and c
        XMC = np.hstack([X, M, C])                                # predictors for paths b and c'
        a = LinearRegression().fit(XC, M.ravel()).coef_[0]        # a : M ~ X + C
        c = LinearRegression().fit(XC, Y.ravel()).coef_[0]        # c : Y ~ X + C (total effect)
        cp, b = LinearRegression().fit(XMC, Y.ravel()).coef_[:2]  # c', b : Y ~ X + M + C
        return a, b, c, cp

    # Standardise X, Y and continuous covariates once (shared by every edge group)
    X, Y = z(X_raw), z(Y_raw)
    C = np.hstack([z(C_cont), C_cat]) if C_cont.shape[1] else C_cat

    # Same bootstrap resamples for every edge group, so groups are directly comparable
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boots)]

    # Run the mediation for each edge group
    rows = []
    for key, label in group_labels.items():
        if key not in edge_group_scores:
            raise KeyError(f"'{key}' not found in edge_group_scores columns.")

        # Mediator: edge-group score
        M = z(edge_group_scores[key].values)        

        # Point estimates of every path
        a, b, c, cp = fit_paths(X, M, Y, C)
        indirect = a * b

        # Bootstrap the indirect effect: refit on resampled participants each time
        boots = np.array([np.prod(fit_paths(X[i], M[i], Y[i], C[i])[:2]) for i in boot_idx])

        lo, hi = np.percentile(boots, CI)                         # 95% percentile CI
        p = 2 * min((boots >= 0).mean(), (boots <= 0).mean())     # Two-sided bootstrap p-value
        p = max(p, 1 / n_boots)                                   # Calcualte p value, but never 0 (for FDR correction)

        rows.append({'edge_group': key, 'label': label,
                     'a': a, 'b': b, 'c': c, 'c_prime': cp, 'indirect': indirect,
                     'ci_lower': lo, 'ci_upper': hi, 'p_value': p})

    # FDR-correct
    results_df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')
    results_df['p_fdr'] = p_fdr
    results_df['significant_fdr'] = reject

    # Print summary
    print(f"Total effect (c, Adversity -> Mental Health) = {results_df['c'].iloc[0]:.4f}")
    for _, r in results_df.iterrows():
        sig = ' *' if r['significant_fdr'] else ''
        print(f"\n{r['label']}:")
        print(f"    a (Adversity -> {r['label']} score)     = {r['a']:.4f}")
        print(f"    b ({r['label']} score -> Mental Health) = {r['b']:.4f}")
        print(f"    c' (direct, Adversity -> Mental Health) = {r['c_prime']:.4f}")
        print(f"    indirect (a*b) = {r['indirect']:.4f} "
              f"[95% CI {r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
              f"p = {r['p_value']:.4f}, FDR p = {r['p_fdr']:.4f}{sig}")

    return results_df

def plot_mediation(mediation_results, save_path, save_name='mediation_barplot.pdf'):
    " Create barplot of indirect effects of mediation for each edge group score. "

    # ---- Settings ----
    PANEL_COLORS = {'M&A': '#BEA4AD', 'A&G': '#9A6B79', 'M&G': '#3D2A30'}
    ORDER = ['M&A', 'A&G', 'M&G']     # top -> bottom

    # Keep the groups present in the results, in the set order (others after)
    labels = [l for l in ORDER if l in set(mediation_results['label'])]
    labels += [l for l in mediation_results['label'] if l not in labels]
    df = mediation_results.set_index('label').loc[labels]

    # barh puts y = 0 at the bottom, so reverse to put the first label at the top
    df = df.iloc[::-1]
    labels = df.index.tolist()

    # Each bar = indirect effect as % of total effect
    ratio = (df['indirect'] / df['c'] * 100).values
    p_fdr = df['p_fdr'].values
    significant = df['significant_fdr'].values
    colors = [PANEL_COLORS.get(l, 'grey') for l in labels]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(5, 6))
    y = np.arange(len(labels))
    ax.barh(y, ratio, color=colors, edgecolor='#6E5560', linewidth=0.8)

    # Annotation: right of the bar end for positive bars; right of zero for negative
    # bars (so it doesn't run into the y-axis labels).
    # Significant (FDR) -> size % and p-value; not significant -> p-value only.
    for yi, val, p, sig in zip(y, ratio, p_fdr, significant):
        x_text = max(val, 0) + 0.1
        p_text = r'$p_{adj}$ < 0.001' if p < 0.001 else rf'$p_{{adj}}$ = {p:.3f}'
        if sig:
            ax.text(x_text, yi, f'{abs(val):.0f}%', va='bottom', ha='left',
                    fontsize=20, color='#333333')
            ax.text(x_text, yi, p_text, va='top', ha='left',
                    fontsize=15, color='#333333')
        else:
            ax.text(x_text, yi, p_text, va='center', ha='left',
                    fontsize=15, color='#333333')

    # x-axis: -2 to 8 by default, widened if a bar (plus its label) falls outside
    x_lo = min(-2, 2 * np.floor((ratio.min() - 1) / 2))
    x_hi = max(8, 2 * np.ceil((ratio.max() + 1.5) / 2))
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis='x', labelsize=15)

    ax.axvline(0, color='#444444', lw=2, ls='--', alpha=0.7)   # zero reference
    ax.set_yticks(y)
    ax.set_yticklabels([f'{l} Score' for l in labels], size=15)
    ax.set_xlabel('Indirect / total effect (%)', fontsize=20)
    ax.set_title('Mediation Size', fontsize=25)
    ax.spines[['top', 'right']].set_visible(False)

    for coll in ax.collections:
        coll.set_rasterized(True)

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, save_name), dpi=600, bbox_inches='tight')
    plt.show()

def run_sliding_window_analysis(edge_group_scores, associated_data, save_path,
                                group_labels=None, n_windows=100, nsub_per_group=800,
                                tail_threshold=1.0, min_nsub=30, alpha=0.05,
                                example_group='Mental Health & Genetics', example_pct=70):
    """
    Run sliding-window analysis along the adversity axis.

    First, participants are ordered based on their adversity scores. Then, starting with the lowest
    adversity, a subgroup of 'nsub_per_group' participants is selected. Within the subgroup, people
    are split into high or low mental health tails (based on > 1SD of the subgroup). Then, edge scores
    are compared between high and low mental health groups using Welch's t-test and Cohen's d.

    This is repeated for 'n_windows' number of subgroups.
    """

    # Set plot colors
    PANEL_COLORS = {'M&A': '#BEA4AD', 'A&G': '#9A6B79', 'M&G': '#3D2A30'}
    COL_HIGH = '#15527F'   
    COL_LOW = '#7FB0D9'    
    FIT_COLOR = '#BE2121'

    if group_labels is None:
        group_labels = {
            'Adversity & Mental Health': 'M&A',
            'Mental Health & Genetics':  'M&G',
            'Adversity & Genetics':      'A&G',
        }

    # Convert adversity into a percentile (0-100)
    adversity_raw = associated_data['Y_adversity_LV1_score'].to_numpy(dtype=float)
    adversity_pct = rankdata(adversity_raw) / len(adversity_raw) * 100
    mental_health = associated_data['Y_MH_LV1_score'].to_numpy(dtype=float)

    # Number of subjects
    n_subjects = len(adversity_pct)

    ### Define window positions ###

    # Sort participants from lowest to highest adversity
    order = np.argsort(adversity_pct)
    half_n = nsub_per_group // 2
    # Define window centres as evenly spaced (half below the centre, half above)
    centre_ranks = np.linspace(half_n, n_subjects - half_n, n_windows).astype(int)

    # Loop through all edge groups
    results = {}
    for key, label in group_labels.items():
        if key not in edge_group_scores:
            raise KeyError(f"'{key}' not found in edge_group_scores columns.")

        # Get score
        score = edge_group_scores[key].to_numpy(dtype=float)

        # Loop through windows
        rows = []
        prev_window, prev_high, prev_low = set(), set(), set()
        for centre_rank in centre_ranks:

            # The nsub_per_group participants around this rank
            window_idx = order[centre_rank - half_n: centre_rank + half_n]
            window_pct = adversity_pct[window_idx]
            window_mh = mental_health[window_idx]

            # High / low MH tails, relative to this window's own MH mean and SD
            mh_mean, mh_sd = window_mh.mean(), window_mh.std(ddof=1)
            high_idx = window_idx[window_mh > mh_mean + tail_threshold * mh_sd]
            low_idx = window_idx[window_mh < mh_mean - tail_threshold * mh_sd]

            # Overlap with the previous window (how much the sample changes per step)
            window_set, high_set, low_set = set(window_idx), set(high_idx), set(low_idx)

            # Create a row of results for this window
            row = {
                'centre': np.median(window_pct),
                'pct_min': window_pct.min(),
                'pct_max': window_pct.max(),
                'n_window': len(window_idx),
                'n_high': len(high_idx),
                'n_low': len(low_idx),
                'p': np.nan,
                'd': np.nan,
                'overlap_window_pct': len(window_set & prev_window) / len(window_set) * 100,
                'overlap_high_pct': (len(high_set & prev_high) / len(high_set) * 100
                                     if high_set else np.nan),
                'overlap_low_pct': (len(low_set & prev_low) / len(low_set) * 100
                                    if low_set else np.nan),
            }

            # Compare edge-group scores between the tails, if both are big enough
            if len(high_idx) >= min_nsub and len(low_idx) >= min_nsub:
                high_scores, low_scores = score[high_idx], score[low_idx]
                row['p'] = ttest_ind(high_scores, low_scores, equal_var=False).pvalue
                row['d'] = pg.compute_effsize(high_scores, low_scores, eftype='cohen') 

            rows.append(row)
            prev_window, prev_high, prev_low = window_set, high_set, low_set

        df = pd.DataFrame(rows)

        # FDR correction across all windows that were tested 
        valid = df['p'].notna()
        df['q'] = np.nan
        if valid.any():
            df.loc[valid, 'q'] = multipletests(df.loc[valid, 'p'], alpha=alpha,
                                               method='fdr_bh')[1]
        df['sig_fdr'] = df['q'] < alpha

        ### Determine if Cohen's d changed along the adversity axis ###

        # Straight-line fit of d against window centre, over all tested windows
        fit = linregress(df.loc[valid, 'centre'], df.loc[valid, 'd']) if valid.sum() > 2 else None

        results[key] = {'label': label, 'df': df, 'fit': fit,
                        'colour': PANEL_COLORS.get(label, 'grey'),
                        'adversity': adversity_pct, 'mental_health': mental_health,
                        'score': score}

        # Print results
        skipped = (~valid).sum()
        fit_text = (f"r = {fit.rvalue:+.3f}, p = {fit.pvalue:.2e}" if fit is not None
                    else "too few tested windows to fit a trend")
        print(f"{label}: {fit_text}, {df['sig_fdr'].sum()}/{valid.sum()} windows "
              f"significant after FDR"
              + (f" ({skipped} windows skipped: a tail had < {min_nsub} participants)"
                 if skipped else ""))

    # Plot: Sliding scale for the edge group with the largest |r| 
    fitted = {k: v for k, v in results.items() if v['fit'] is not None}
    if fitted:
        key = max(fitted, key=lambda k: abs(fitted[k]['fit'].rvalue))
        label, df, fit = results[key]['label'], results[key]['df'], results[key]['fit']

        valid = df['p'].notna()            # windows that were tested
        sig = valid & df['sig_fdr']        # FDR-significant windows
        ns = valid & ~df['sig_fdr']        # tested but not significant

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df.loc[ns, 'centre'], df.loc[ns, 'd'], s=150, color='grey',
                   alpha=0.30, edgecolor='none', label='Not significant', rasterized=True)
        ax.scatter(df.loc[sig, 'centre'], df.loc[sig, 'd'], s=150,
                   color=PANEL_COLORS.get(label, 'grey'), alpha=0.95, edgecolor='white',
                   linewidth=0.5, label=rf'$p_{{adj}}$ < {alpha}', rasterized=True)

        # Fitted line across the tested windows
        x_line = np.array([df.loc[valid, 'centre'].min(), df.loc[valid, 'centre'].max()])
        ax.plot(x_line, fit.intercept + fit.slope * x_line, color=FIT_COLOR, lw=6,
                label=f'Fit:  r = {fit.rvalue:+.2f},  p = {fit.pvalue:.2e}')

        ax.set_xlabel('Adversity percentile (%)', fontsize=20)
        ax.set_ylabel("Effect size (Cohen's d)", fontsize=20)
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, fontsize=15, loc='best')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=15)
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'sliding_scatter_plot.pdf'),
                    dpi=600, bbox_inches='tight')
        plt.show()

    # Plot: example of one window's calculation
    if example_group is not None:
        if example_group not in results:
            raise KeyError(f"example_group '{example_group}' is not one of the edge groups run.")
        label = results[example_group]['label']
        score = results[example_group]['score']

        # The nsub_per_group participants nearest example_pct (same window rule as above)
        centre_pos = np.searchsorted(adversity_pct[order], example_pct)
        centre_pos = np.clip(centre_pos, half_n, n_subjects - half_n)   # keep window in range
        window_idx = order[centre_pos - half_n: centre_pos + half_n]
        window_mh, window_score = mental_health[window_idx], score[window_idx]

        # High / low MH tails, relative to this window's own mean and SD
        mh_mean, mh_sd = window_mh.mean(), window_mh.std(ddof=1)
        hi_cut = mh_mean + tail_threshold * mh_sd
        lo_cut = mh_mean - tail_threshold * mh_sd
        high_scores = window_score[window_mh > hi_cut]
        low_scores = window_score[window_mh < lo_cut]

        fig, (axH, axV) = plt.subplots(2, 1, figsize=(6, 8))

        # Top: the window's MH distribution, tails coloured
        _, bins, patches = axH.hist(window_mh, bins=40, color='#cccccc',
                                    edgecolor='white', rasterized=True)
        for patch, left in zip(patches, bins[:-1]):
            if left >= hi_cut:
                patch.set_facecolor(COL_HIGH)
            elif left < lo_cut:
                patch.set_facecolor(COL_LOW)
        axH.axvline(hi_cut, color=COL_HIGH, ls='--', lw=2)
        axH.axvline(lo_cut, color=COL_LOW, ls='--', lw=2)
        axH.set_xlabel('Mental health LV', fontsize=20)
        axH.set_ylabel('Number of subjects', fontsize=20)
        axH.tick_params(axis='both', labelsize=15)
        axH.legend(handles=[Patch(facecolor=COL_HIGH, label='Poor MH'),
                            Patch(facecolor=COL_LOW, label='Good MH')],
                   frameon=False, fontsize=15)
        axH.spines[['top', 'right']].set_visible(False)

        # Bottom: edge-group scores in each tail
        parts = axV.violinplot([low_scores, high_scores], positions=[0, 0.5],
                               showmeans=True, showextrema=False, widths=0.4)
        for body, col in zip(parts['bodies'], [COL_LOW, COL_HIGH]):
            body.set_facecolor(col)
            body.set_alpha(0.85)
            body.set_edgecolor('white')
            body.set_rasterized(True)
        parts['cmeans'].set_color('#000000')
        axV.set_xticks([0, 0.5])
        axV.set_xlim(-0.4, 0.9)
        axV.set_xticklabels([f'Good MH\n(n = {len(low_scores)})',
                             f'Poor MH\n(n = {len(high_scores)})'], fontsize=15)
        axV.tick_params(axis='y', labelsize=12)
        axV.set_ylabel(f'{label} Score', fontsize=20)
        axV.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'sliding_sample_example.pdf'),
                    dpi=600, bbox_inches='tight')
        plt.show()

############################################### FUNCTIONS FOR G_heterogenous_profile_matched #############################################
  
def match_adversity_mh_groups(associated_data, save_path, quantile_threshold=0.30,
                              nsub_target=200, caliper=0.15):
    """
    Build four matched groups crossing adversity (high / low) with mental health (poor / good):

        HP : high adversity, poor mental health (high MH score)
        HG : high adversity, good mental health (low MH score)
        LP : low adversity,  poor mental health
        LG : low adversity,  good mental health

    How the matching works:
    1) Candidates: participants in the top / bottom `quantile_threshold` of adversity and of
       MH are sorted into the four groups above (people in the middle are not used).
    2) Anchor: the smallest group is the anchor, as it limits how many sets can be built.
    3) For each anchor participant (in order), one partner is chosen from each other group:
         - same-adversity group (opposite MH): adversity within `caliper` SD of the anchor
         - same-MH group (opposite adversity): MH within `caliper` SD of the anchor
         - opposite-on-both group: adversity within `caliper` of the same-MH partner and
           MH within `caliper` of the same-adversity partner
       Among partners that pass, the one closest to the anchor in age and sex is taken.
       If any partner can't be found, that anchor is skipped. Each participant is used once. 
       Stops once `nsub_target` sets of four are built (or candidates run out).
    """

    # Group names
    GROUPS = ['HP', 'HG', 'LP', 'LG']

    # Extract data
    participant_data = pd.DataFrame({
        'mh_score':  associated_data['Y_MH_LV1_score'].to_numpy(dtype=float),
        'adv_score': associated_data['Y_adversity_LV1_score'].to_numpy(dtype=float),
        'age':       associated_data['age'].to_numpy(dtype=float),
        'sex':       associated_data['sex'].to_numpy(),
    })
    # Sex as numbers (needed for distances and t-tests): if it's text, code it 0/1/...
    if not pd.api.types.is_numeric_dtype(participant_data['sex']):
        participant_data['sex'] = participant_data['sex'].astype('category').cat.codes
    participant_data['sex'] = participant_data['sex'].astype(float)

    # z-scored copies, so calipers and distances are in SD units
    z = {col: zscore(participant_data[col].to_numpy()) for col in participant_data.columns}
    demographics = np.column_stack([z['age'], z['sex']])   # used to find the closest partner

    # 1) Define candidates from the quantiles
    adv, mh = participant_data['adv_score'], participant_data['mh_score']
    high_adv = adv >= adv.quantile(1 - quantile_threshold)
    low_adv = adv <= adv.quantile(quantile_threshold)
    poor_mh = mh >= mh.quantile(1 - quantile_threshold)  
    good_mh = mh <= mh.quantile(quantile_threshold)

    candidates = {
        'HP': np.flatnonzero(high_adv & poor_mh),
        'HG': np.flatnonzero(high_adv & good_mh),
        'LP': np.flatnonzero(low_adv & poor_mh),
        'LG': np.flatnonzero(low_adv & good_mh),
    }

    # Adversity / MH level of each group, used to work out how groups relate
    levels = {'HP': ('high', 'poor'), 'HG': ('high', 'good'),
              'LP': ('low', 'poor'), 'LG': ('low', 'good')}

    print(f"Possible participants (top/bottom {quantile_threshold:.0%} of adversity and MH):")
    for g in GROUPS:
        print(f"    {g}: {len(candidates[g]):,}")

    # 2) Define the anchor group (i.e., smallest group)
    anchor = min(GROUPS, key=lambda g: len(candidates[g]))
    a_adv, a_mh = levels[anchor]
    flip = {'high': 'low', 'low': 'high', 'poor': 'good', 'good': 'poor'}
    group_with = {lv: g for g, lv in levels.items()}              # (adv, mh) -> group
    same_adv = group_with[(a_adv, flip[a_mh])]                    # same adversity, opposite MH
    same_mh = group_with[(flip[a_adv], a_mh)]                     # same MH, opposite adversity
    opposite = group_with[(flip[a_adv], flip[a_mh])]              # opposite on both

    print(f"\nAnchor (smallest) group: {anchor} -> at most {len(candidates[anchor]):,} "
          f"matched sets possible")

    # 3) Create matches for each group

    # Participants already taken, per group
    used = {g: set() for g in GROUPS}       

    def best_match(group, anchor_person, constraints):
        """Closest (age/sex) unused candidate in `group` passing every caliper.
        constraints: list of (z-scored column, reference value) pairs."""
        best, best_dist = None, np.inf
        for person in candidates[group]:
            if person in used[group]:
                continue
            # Must be within `caliper` SD of every reference value
            if any(abs(col[person] - ref) > caliper for col, ref in constraints):
                continue
            dist = np.linalg.norm(demographics[person] - demographics[anchor_person])
            if dist < best_dist:
                best, best_dist = person, dist
        return best

    matched_sets = []
    for person in candidates[anchor]:
        if len(matched_sets) >= nsub_target:
            break

        # Same adversity group: match on adversity to the anchor
        p_adv = best_match(same_adv, person, [(z['adv_score'], z['adv_score'][person])])
        if p_adv is None:
            continue
        # Same MH group: match on MH to the anchor
        p_mh = best_match(same_mh, person, [(z['mh_score'], z['mh_score'][person])])
        if p_mh is None:
            continue
        # Opposite group: adversity like the same-MH partner, MH like the same-adversity partner
        p_opp = best_match(opposite, person, [(z['adv_score'], z['adv_score'][p_mh]),
                                              (z['mh_score'], z['mh_score'][p_adv])])
        if p_opp is None:
            continue

        # All three found: lock them in so they can't be reused
        used[same_adv].add(p_adv)
        used[same_mh].add(p_mh)
        used[opposite].add(p_opp)
        matched_sets.append({anchor: person, same_adv: p_adv, same_mh: p_mh, opposite: p_opp})

    print(f"Matched sets found: {len(matched_sets):,}")
    if len(matched_sets) < nsub_target:
        print(f"    WARNING: fewer than the {nsub_target:,} wanted. "
              f"Loosen the caliper / quantile_threshold or lower nsub_target.")
    if not matched_sets:
        raise ValueError("No matched sets found — nothing to test or plot.")

    # Participants (row positions) in each group, and their data
    group_idx = {g: np.array([s[g] for s in matched_sets]) for g in GROUPS}
    group_frames = {g: participant_data.loc[group_idx[g]] for g in GROUPS}

    # True/False per row of associated_data, and each matched row's group label
    matched_index = np.zeros(len(participant_data), dtype=bool)
    labels = np.empty(len(participant_data), dtype=object)
    for g in GROUPS:
        matched_index[group_idx[g]] = True
        labels[group_idx[g]] = g
    group_membership = labels[matched_index]

    ### Checks: t-tests between groups ###
    names = {'adv_score': 'adversity', 'mh_score': 'MH', 'age': 'age', 'sex': 'sex'}

    def ttest_text(g1, g2, col):
        t, p = ttest_ind(group_frames[g1][col], group_frames[g2][col])
        p_text = f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"   # small p in scientific notation
        return f"{names[col]} t = {t:.2f}, p = {p_text}"

    def check(g1, g2, differ_col):
        match_cols = [c for c in names if c != differ_col]
        print(f"    {g1} vs {g2}: {ttest_text(g1, g2, differ_col)} | "
              + " | ".join(ttest_text(g1, g2, c) for c in match_cols))

    print("\nAdversity contrast (should DIFFER on adversity, MATCH on MH / age / sex):")
    check('HP', 'LP', 'adv_score')
    check('HG', 'LG', 'adv_score')
    print("\nMH contrast (should DIFFER on MH, MATCH on adversity / age / sex):")
    check('HP', 'HG', 'mh_score')
    check('LP', 'LG', 'mh_score')

    # Plot: adversity and MH per group
    purples = LinearSegmentedColormap.from_list('purples', ['#2A0E33', '#7E4B9E', '#E6D4EE'])
    colours = dict(zip(GROUPS, [purples(t) for t in np.linspace(0.10, 0.95, 4)]))
    xpos = {g: i + 1 for i, g in enumerate(GROUPS)}

    def stars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

    panels = [('adv_score', 'Adversity', [('HP', 'LP'), ('HG', 'LG')]),
              ('mh_score', 'Poor Mental Health', [('HP', 'HG'), ('LP', 'LG')])]

    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    for ax, (col, ylabel, pairs) in zip(axes, panels):
        data = [group_frames[g][col].to_numpy() for g in GROUPS]

        violin = ax.violinplot(data, showmeans=True, showextrema=True)
        for body, g in zip(violin['bodies'], GROUPS):
            body.set_facecolor(colours[g])
            body.set_edgecolor('black')
            body.set_alpha(0.8)
        for piece in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            violin[piece].set_edgecolor('0.1')
            violin[piece].set_linewidth(1)

        # Significance brackets above the violins, one level per pair (they overlap in x)
        top, span = max(d.max() for d in data), np.ptp(np.concatenate(data))
        y0, step, tick = top + span * 0.06, span * 0.14, span * 0.02
        for level, (g1, g2) in enumerate(pairs):
            p = ttest_ind(group_frames[g1][col], group_frames[g2][col]).pvalue
            y = y0 + level * step
            x1, x2 = xpos[g1], xpos[g2]
            ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], lw=1.2, c='0.1', clip_on=False)
            ax.text((x1 + x2) / 2, y + tick, stars(p), ha='center', va='bottom',
                    fontsize=18, clip_on=False)
        ax.set_ylim(top=y0 + len(pairs) * step)

        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(GROUPS)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', labelsize=15)
        ax.spines[['top', 'right']].set_visible(False)
        for coll in ax.collections:
            coll.set_rasterized(True)

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'heterogenous_profiles_violins.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()

    return {'matched_index': matched_index,
            'group_membership': group_membership,
            'group_frames': group_frames}

def compare_matched_groups(matched_df, save_path):
    """
    Compare the four matched groups (HP, HG, LP, LG) on each edge-group metric.

    For each metric (count_risk, count_health, weight_risk, weight_health):
      1) Levene's test to test which anova is appropriate (equal or unequal variances).
      2) Either one-way ANOVA if variances are equal, Welch ANOVA if not.
      3) Post-hoc pairwise tests (Tukey HSD if variances are equal, Games-Howell if not)

    """

    # Define group and metric names
    GROUPS = ['HP', 'HG', 'LP', 'LG']
    METRICS = ['count_risk', 'count_health', 'weight_risk', 'weight_health']
    # Metrics to plot: (column, y-axis label, file name)
    PLOTS = [('count_risk', 'Connection Count', 'matched_risk_count'),
             ('weight_health', 'Mean Weight', 'matched_health_weight')]

    def stars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

    def p_text(p):
        return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"

    # Run statstical tests for each metric
    results = {}
    for col in METRICS:
        data = matched_df[['group', col]].dropna()

        # 1) Levene's test for equal variances across groups
        levene = pg.homoscedasticity(data, dv=col, group='group', method='levene')
        equal_var = bool(levene['equal_var'].iloc[0])

        # 2) ANOVA + 3) post-hoc, chosen by whether variances are equal
        if equal_var:
            full_anova = pg.anova(data, dv=col, between='group')
            posthoc = pg.pairwise_tukey(data, dv=col, between='group')
            omni_name, ph_name, p_col = 'One-way ANOVA', 'Tukey HSD', 'p_tukey'
        else:
            full_anova = pg.welch_anova(data, dv=col, between='group')
            posthoc = pg.pairwise_gameshowell(data, dv=col, between='group')
            omni_name, ph_name, p_col = 'Welch ANOVA', 'Games-Howell', 'pval'

        # Standadrize scoring so pingouin returns the same column names regardless of version
        full_anova.columns = full_anova.columns.str.replace('-', '_')
        posthoc.columns = posthoc.columns.str.replace('-', '_')
        posthoc = posthoc.rename(columns={p_col: 'p'})   # same p column name either way

        # Store results
        results[col] = {'levene': levene, 'full_anova': full_anova, 'posthoc': posthoc}

        # Print
        o = full_anova.iloc[0]
        print(f"\n {col} ")
        print(f"  Levene: W = {levene['W'].iloc[0]:.2f}, p = {p_text(levene['pval'].iloc[0])} "
              f"({'equal' if equal_var else 'unequal'} variances -> {omni_name} + {ph_name})")
        print(f"  {omni_name}: F({o['ddof1']:g}, {o['ddof2']:.1f}) = {o['F']:.2f}, "
              f"p = {p_text(o['p_unc'])}, ηp² = {o['np2']:.3f} {stars(o['p_unc'])}")
        print(f"  Post-hoc ({ph_name}):")
        for _, row in posthoc.iterrows():
            flag = '  <-- HG' if 'HG' in (row['A'], row['B']) else ''
            print(f"    {row['A']} vs {row['B']}: T = {row['T']:.2f}, g = {row['hedges']:.2f}, "
                  f"p = {p_text(row['p'])} {stars(row['p'])}{flag}")

    # Plot HG significant pairs for each metric
    purples = LinearSegmentedColormap.from_list('purples', ['#2A0E33', '#7E4B9E', '#E6D4EE'])
    colours = dict(zip(GROUPS, [purples(t) for t in np.linspace(0.10, 0.95, 4)]))
    xpos = {g: i + 1 for i, g in enumerate(GROUPS)}

    for col, ylabel, fname in PLOTS:
        posthoc = results[col]['posthoc']
        omni_p = results[col]['full_anova']['p_unc'].iloc[0]
        values = [matched_df.loc[matched_df['group'] == g, col].dropna().to_numpy()
                  for g in GROUPS]

        fig, ax = plt.subplots(figsize=(5, 4.5))
        violin = ax.violinplot(values, showmeans=True, showextrema=True)
        for body, g in zip(violin['bodies'], GROUPS):
            body.set_facecolor(colours[g])
            body.set_edgecolor('black')
            body.set_alpha(0.8)
        for piece in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            violin[piece].set_edgecolor('0.1')
            violin[piece].set_linewidth(1)

        # Significant pairs involving HG (none if the full_anova test isn't significant)
        hg_sig = posthoc[(posthoc['A'].eq('HG') | posthoc['B'].eq('HG')) & (posthoc['p'] < 0.05)]
        if omni_p >= 0.05:
            hg_sig = hg_sig.iloc[0:0]

        # Brackets: shortest at the bottom, wider ones stacked above
        top = max(v.max() for v in values)
        span = top - min(v.min() for v in values)
        y0, step, tick = top + span * 0.08, span * 0.11, span * 0.02
        pairs = sorted(hg_sig.itertuples(), key=lambda r: abs(xpos[r.A] - xpos[r.B]))
        for level, r in enumerate(pairs):
            x1, x2, y = xpos[r.A], xpos[r.B], y0 + level * step
            ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], lw=1.3, c='0.1', clip_on=False)
            ax.text((x1 + x2) / 2, y + tick, stars(r.p), ha='center', va='bottom',
                    fontsize=18, fontweight='bold', clip_on=False)
        ax.set_ylim(top=y0 + (len(pairs) * step if pairs else span * 0.05))

        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(GROUPS)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', labelsize=15)
        ax.spines[['top', 'right']].set_visible(False)
        for coll in ax.collections:
            coll.set_rasterized(True)

        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'{fname}.pdf'), dpi=600, bbox_inches='tight')
        plt.show()

    return results

def hemisphere_location(aligned_mask, region_labels, group=''):
    "Identify the hemisphere location of edges."

    # Define mask and number of edges
    aligned_mask = np.asarray(aligned_mask)
    nnodes = aligned_mask.shape[0]

    # Hemisphere of every node, read from its region name (e.g. 'amyg_L_...' -> 'L')
    hemi = np.array([(region_labels[i].split('_') + ['?', '?'])[1] for i in range(nnodes)])
    
    # Grab upper triangle (to count each edge only once)
    rows, cols = np.triu_indices(nnodes, k=1)
    direction = aligned_mask[rows, cols]

    # Edge type: both ends left, both ends right, or one end in each (anything else = inter)
    both_left = (hemi[rows] == 'L') & (hemi[cols] == 'L')
    both_right = (hemi[rows] == 'R') & (hemi[cols] == 'R')

    # Print out percentages
    print(f"Hemisphere location{f' ({group})' if group else ''}")
    counts = {}
    for name, in_set in [('Risk-aligned', direction > 0), ('Health-aligned', direction < 0)]:
        c = {'left': int((in_set & both_left).sum()),
             'right': int((in_set & both_right).sum())}
        c['inter'] = int(in_set.sum()) - c['left'] - c['right']
        counts[name] = c
        # Print: each category as % of all edges of this sign, then the raw counts
        total = sum(c.values())
        if total == 0:
            print(f"    {name}: no edges")
            continue
        pct = '  '.join(f"{k} = {v / total * 100:5.1f}%" for k, v in c.items())
        print(f"    {name:<15}{pct}")

def connection_type(aligned_mask, rich_club, group=''):
    """Categorize edges as rich (between two rich club nodes), 
    feeder (between one rich club node and one non-rich club node), 
    or local (between two non-rich club nodes)."""

    # Define mask and number of nodes
    aligned_mask = np.asarray(aligned_mask)
    nnodes = aligned_mask.shape[0]

    # True for every rich-club node
    is_rich = np.zeros(nnodes, dtype=bool)
    is_rich[list(rich_club)] = True

    # Each edge once (upper triangle): its two end nodes and its direction
    rows, cols = np.triu_indices(nnodes, k=1)
    direction = aligned_mask[rows, cols]

    # Number of rich-club ends per edge: 2 = rich, 1 = feeder, 0 = local
    n_rich_ends = is_rich[rows].astype(int) + is_rich[cols].astype(int)

    # Print out percentages
    print(f"Connection type{f' ({group})' if group else ''}")
    counts = {}
    for name, in_set in [('Risk-aligned', direction > 0), ('Health-aligned', direction < 0)]:
        c = {'rich': int((in_set & (n_rich_ends == 2)).sum()),
             'feeder': int((in_set & (n_rich_ends == 1)).sum()),
             'local': int((in_set & (n_rich_ends == 0)).sum())}
        counts[name] = c
        # Print: each category as % of all edges of this sign, then the raw counts
        total = sum(c.values())
        if total == 0:
            print(f"    {name}: no edges")
            continue
        pct = '  '.join(f"{k} = {v / total * 100:5.1f}%" for k, v in c.items())
        print(f"    {name:<15}{pct}")

def region_involvement(aligned_mask, general_region, lobe_labels, save_path,
                       nperm=10000, seed=42, group=''):
    """
    Assess if risk- and health-aligned edges are significnatly more involved in different brain regions.

    Method:
    1) Involvement: for each region, count how many edge endpoints fall in it,
       separately for Risk- and Health-aligned edges (each edge has 2 endpoints, so 2 counts per edge).
    2) Share: divide each count by each group's total endpoints (2 x number of edges), so the two
       groups are comparable even if one has more edges.
    3) Test: Permutate edges (e.g., randomize the endpoints of all edges) and then determine with a
        two-sided p-value if the observed difference in share between the two groups is bigger than 
        expected by chance.
    4) Plot: Risk vs Health share per region, grouped by lobe.

    """

    # Plot details
    RISK_COLOR, HEALTH_COLOR = '#2E8B57', '#D6608E'
    LOBE_ORDER = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Insular', 'Limbic', 'Subcortical']

    def stars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

    aligned_mask = np.asarray(aligned_mask)
    general_region = np.asarray(general_region)
    lobe_labels = np.asarray(lobe_labels)
    nnodes = aligned_mask.shape[0]
    #tag = f" ({group})" if group else ""

    # 1) Involvement: endpoint counts per general region
    regions = np.unique(general_region)
    n_regions = len(regions)
    node_region = np.searchsorted(regions, general_region)   # each node -> region number

    # Each edge once (upper triangle): its two end nodes and direction
    rows, cols = np.triu_indices(nnodes, k=1)
    direction = aligned_mask[rows, cols]

    def endpoint_counts(edge_sel):
        "Number of endpoints of the selected edges that land in each region."
        ends = np.concatenate([node_region[rows[edge_sel]], node_region[cols[edge_sel]]])
        return np.bincount(ends, minlength=n_regions)

    is_risk, is_health = direction > 0, direction < 0
    n_risk, n_health = int(is_risk.sum()), int(is_health.sum())

    # 2) Share: % of each group's endpoints in each region
    risk_share = endpoint_counts(is_risk) / (2 * n_risk) * 100
    health_share = endpoint_counts(is_health) / (2 * n_health) * 100
    obs_diff = risk_share - health_share                     # > 0 = leans Risk-aligned
    
    # 3) Permutation test: random scattering of the same numbers of edges
    rng = np.random.default_rng(seed)

    def random_shares(n_edges):
        "Scatter n_edges at random (two different regions per edge); return % shares."
        a = rng.integers(0, n_regions, size=n_edges)
        b = rng.integers(0, n_regions, size=n_edges)
        same = a == b
        while same.any():                                    # redraw until ends differ
            b[same] = rng.integers(0, n_regions, size=same.sum())
            same = a == b
        counts = np.bincount(a, minlength=n_regions) + np.bincount(b, minlength=n_regions)
        return counts / (2 * n_edges) * 100

    null_diff = np.array([random_shares(n_risk) - random_shares(n_health)
                          for _ in range(nperm)])

    # Two-sided p: how often a random difference is at least as extreme, in either tail
    p_high = (np.sum(null_diff >= obs_diff, axis=0) + 1) / (nperm + 1)
    p_low = (np.sum(null_diff <= obs_diff, axis=0) + 1) / (nperm + 1)
    p_values = np.minimum(2 * np.minimum(p_high, p_low), 1.0)

    sig_regions = [i for i in np.argsort(p_values) if p_values[i] < 0.05]   # significant only
    if not sig_regions:
        print("    No regions significant at p < 0.05")
    else:
        print(f"    {'region':14}{'risk%':>7}{'health%':>9}{'diff%':>8}{'p':>9}")
    for i in sig_regions:
        lean = 'Risk' if obs_diff[i] > 0 else 'Health'
        print(f"    {regions[i]:14}{risk_share[i]:7.2f}{health_share[i]:9.2f}"
              f"{obs_diff[i]:+8.2f}{p_values[i]:9.4f}  {lean}-leaning {stars(p_values[i])}")

    # 4) Plot: shares per region, grouped by lobe
    # Each general region's lobe = the most common lobe among its nodes
    region_lobe = {}
    for r in regions:
        node_lobes = lobe_labels[general_region == r]
        region_lobe[r] = Counter(node_lobes).most_common(1)[0][0]
        if len(set(node_lobes)) > 1:
            print(f"note: {r} spans lobes {set(node_lobes)} -> assigned {region_lobe[r]}")

    # Order: by lobe (LOBE_ORDER first, any others after), then by total share within lobe
    lobes = list(dict.fromkeys(LOBE_ORDER + sorted(set(region_lobe.values()))))
    total_share = dict(zip(regions, risk_share + health_share))
    ordered = [r for lobe in lobes
               for r in sorted((r for r in regions if region_lobe[r] == lobe),
                               key=lambda r: total_share[r])]
    ordered = ordered[::-1]                                  # barh draws bottom-up
    idx = [np.flatnonzero(regions == r)[0] for r in ordered]
    risk_sorted, health_sorted, p_sorted = risk_share[idx], health_share[idx], p_values[idx]

    y = np.arange(len(ordered))
    bar_h = 0.4
    fig, ax = plt.subplots(figsize=(7.5, max(6, len(ordered) * 0.28)))
    ax.barh(y + bar_h / 2, risk_sorted, height=bar_h, color=RISK_COLOR,
            edgecolor='black', linewidth=0.5, label='Risk-aligned')
    ax.barh(y - bar_h / 2, health_sorted, height=bar_h, color=HEALTH_COLOR,
            edgecolor='black', linewidth=0.5, label='Health-aligned')

    # Stars at the end of the longer bar for p < 0.05
    bar_end = np.maximum(risk_sorted, health_sorted)
    xmax = bar_end.max()
    for yi, p in enumerate(p_sorted):
        if p < 0.05:
            ax.text(bar_end[yi] + xmax * 0.015, yi, stars(p), va='center', ha='left',
                    fontsize=15, fontweight='bold')

    # Lobe blocks: dashed line between lobes, lobe name to the right of each block
    region_lobes = [region_lobe[r] for r in ordered]
    starts = [i for i in range(len(ordered)) if i == 0 or region_lobes[i] != region_lobes[i - 1]]
    for k, start in enumerate(starts):
        end = (starts[k + 1] if k + 1 < len(starts) else len(ordered)) - 1
        if start > 0:
            ax.axhline(start - 0.5, color='0.6', lw=0.8, ls='--')
        ax.text(xmax * 1.12, (start + end) / 2, region_lobes[start], va='center',
                ha='center', fontsize=18, fontweight='bold', color='0.25')

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=15)
    ax.set_xlabel('Edge Involvement (%)', fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlim(right=xmax * 1.18)                          

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'region_edge_counts_by_lobe.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()

    return {r: {'risk_share': risk_share[i], 'health_share': health_share[i],
                'diff': obs_diff[i], 'p': p_values[i]} for i, r in enumerate(regions)}

def region_circos(aligned_mask, general_region, region_results, save_path, group=''):
    "Creat circos plot of all risk- and health-aligned edges that touch significant regions."

    # Set up plot parameters
    RISK_COLOR = "#184d18"     
    HEALTH_COLOR = "#a73685"   
    GREY_LABEL = "0.6"         # label colour for non-significant regions
    PLACEHOLDER = 1.0          # arc size for regions with no drawn links

    aligned_mask = np.asarray(aligned_mask)
    general_region = np.asarray(general_region)

    # Identify significant regions
    sig_regions = {r for r, res in region_results.items() if res['p'] < 0.05}

    # Count Risk / Health edges between each pair of regions 
    # Each edge once (upper triangle); region pair stored in alphabetical order so
    # A-B and B-A are the same key
    rows, cols = np.triu_indices(aligned_mask.shape[0], k=1)
    direction = aligned_mask[rows, cols]
    risk_count, health_count = defaultdict(int), defaultdict(int)
    for r, c, d in zip(rows, cols, direction):
        if d == 0:
            continue
        key = tuple(sorted((general_region[r], general_region[c])))
        (risk_count if d > 0 else health_count)[key] += 1

    # Keep pairs touching a significant region; one link per colour 
    links = []
    for key in sorted(set(risk_count) | set(health_count)):
        if key[0] not in sig_regions and key[1] not in sig_regions:
            continue
        a, b = key
        if health_count[key]:          # pink first so green sits on top
            links.append({'a': a, 'b': b, 'w': health_count[key], 'color': HEALTH_COLOR})
        if risk_count[key]:
            links.append({'a': a, 'b': b, 'w': risk_count[key], 'color': RISK_COLOR})

    # Arc size per region = total width of the links that end there
    region_names = sorted(set(general_region))
    arc = defaultdict(float)
    for lk in links:
        arc[lk['a']] += lk['w']
        arc[lk['b']] += lk['w']
    sectors = {r: max(arc[r], PLACEHOLDER) for r in region_names}

    # Give each link its own slot on both of its regions' arcs
    cursor = {r: 0.0 for r in region_names}
    for lk in links:
        for end in ('a', 'b'):
            region, start = lk[end], cursor[lk[end]]
            lk[f'{end}_range'] = (start, start + lk['w'])
            cursor[region] = start + lk['w']

    # Create plot
    circos = Circos(sectors, space=1, start=-270, end=90)
    for sector in circos.sectors:
        sector.text(sector.name, r=108, size=15,
                    color='black' if sector.name in sig_regions else GREY_LABEL,
                    orientation='vertical', adjust_rotation=True)

    # Widest links first so thinner ones stay visible on top
    for lk in sorted(links, key=lambda d: -d['w']):
        circos.link((lk['a'], *lk['a_range']), (lk['b'], *lk['b_range']),
                    color=lk['color'], alpha=0.55, lw=0, height_ratio=0.8)

    fig = circos.plotfig(figsize=(11, 11))
    fig.axes[0].set_rasterization_zorder(2)

    prefix = f"{group.replace('&', '')}_" if group else ''
    plt.savefig(os.path.join(save_path, f'{prefix}region_circos_sig.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()