# figures/plot_utils.py
import matplotlib
import matplotlib.pyplot as plt
import os

def setup_publication_style():
    """IEEE/Medical Image Analysis ready. 1000 DPI, Arial, colorblind-safe."""
    matplotlib.rcParams.update({
        'figure.dpi':       1000,
        'savefig.dpi':      1000,
        'font.family':      'Arial',
        'font.size':        8,
        'axes.titlesize':   9,
        'axes.labelsize':   8,
        'xtick.labelsize':  7,
        'ytick.labelsize':  7,
        'legend.fontsize':  7,
        'lines.linewidth':  1.0,
        'axes.linewidth':   0.8,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.linewidth':   0.5,
        'pdf.fonttype':     42,   # TrueType — required by IEEE
        'ps.fonttype':      42,
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })
    # Wong (2011) colorblind-safe 8-color palette
    return {
        'black':      '#000000',
        'orange':     '#E69F00',
        'sky_blue':   '#56B4E9',
        'green':      '#009E73',
        'yellow':     '#F0E442',
        'blue':       '#0072B2',
        'vermillion': '#D55E00',
        'pink':       '#CC79A7',
    }

def save_fig(fig, name, output_dir="./figures/output"):
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(f"{output_dir}/{name}.{fmt}",
                    dpi=1000, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"Saved {name} to {output_dir}/")
