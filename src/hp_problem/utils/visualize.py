"""Minimal plotting helpers for HP lattice folds."""
from __future__ import annotations

import io
import re
import logging
from typing import List, Optional, Tuple
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Utility: plot reward & energy curves
# --------------------------------------------------------------------------- #

def _plot_metrics(csv_path: pathlib.Path, out_dir: pathlib.Path, ma_window: int = 500):
    """
    Plot episode reward and physical energy together with their moving-average
    curves and save <reward_curve.png> & <energy_curve.png> into *out_dir*.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    # moving averages
    df["Reward_MA"] = df["Reward"].rolling(ma_window, min_periods=1).mean()
    df["Energy_MA"] = df["Energy"].rolling(ma_window, min_periods=1).mean()

    # ---------- Reward ----------
    fig, ax = plt.subplots()
    ax.plot(df["episode"], df["Reward"], alpha=0.30, label="Reward")
    ax.plot(df["episode"], df["Reward_MA"],            label=f"Reward MA({ma_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Episode reward vs. moving average")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=300)
    plt.close(fig)

    # ---------- Energy ----------
    # fig, ax = plt.subplots()
    # ax.plot(df["episode"], df["Energy"], alpha=0.30, label="Energy")
    # ax.plot(df["episode"], df["Energy_MA"],            label=f"Energy MA({ma_window})")
    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Physical folding energy")
    # ax.set_title("Physical energy vs. moving average")
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(out_dir / "energy_curve.png", dpi=300)
    # plt.close(fig)

    print(f"Learning-curves saved to {out_dir}")


def expand_sequence(expr: str) -> List[str]:
    """
    Expand strings like "(hp)2ph(hp)2(ph)2hp(ph)2" → ['h','p','h','p',…].
    """
    expr = expr.lower()
    pattern = re.compile(r'\(([hp]{2})\)(\d+)')
    while True:
        m = pattern.search(expr)
        if not m:
            break
        unit, count = m.group(1), int(m.group(2))
        expr = expr[:m.start()] + unit * count + expr[m.end():]
    return [ch for ch in expr if ch in ('h','p')]


def find_hh_contacts(seq: List[str],
                     coords: np.ndarray) -> List[Tuple[int,int]]:
    """
    All non‐adjacent H–H Manhattan contacts.
    """
    L = len(seq)
    contacts = []
    for i in range(L):
        if seq[i] != 'H':
            continue
        for j in range(i+2, L):
            if seq[j] != 'H':
                continue
            if abs(coords[i,0] - coords[j,0]) + abs(coords[i,1] - coords[j,1]) == 1:
                contacts.append((i, j))
    logger.debug(f"Found {len(contacts)} H-H contacts in sequence {''.join(seq)}")
    return contacts


def find_hh_contacts_3d(seq: List[str],
                        coords: np.ndarray) -> List[Tuple[int,int]]:
    """
    All non‐adjacent H–H Manhattan contacts in 3D.
    Assumes coords is an array of shape (L, 3).
    """
    L = len(seq)
    contacts = []
    if L == 0:
        return contacts
        
    for i in range(L):
        if seq[i] != 'h':
            continue
        for j in range(i + 2, L):  # Non-adjacent residues
            if seq[j] != 'h':
                continue
            # Check 3D Manhattan distance
            manhattan_dist = abs(coords[i,0] - coords[j,0]) + \
                             abs(coords[i,1] - coords[j,1]) + \
                             abs(coords[i,2] - coords[j,2])
            if manhattan_dist == 1:
                contacts.append((i, j))
    return contacts


def contrasting_text_color(bg_color: str) -> str:
    rgb = mcolors.to_rgb(bg_color)
    lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return 'black' if lum > 0.5 else 'white'


def _plot_hp_chain(
    seq: List[str],
    coords: np.ndarray,
    ax,
    start_idx:       int    = 0,
    end_idx:         int    = None,
    # styling ↓
    title:            str   = "HP Folded Chain", # New: Title for the plot
    backbone_color:   str   = 'dimgray',
    backbone_lw:      float = 2.0,
    hh_contact_color: str   = 'crimson',
    hh_contact_ls:    str   = '--',
    hh_contact_lw:    float = 2.0,
    marker_size:      float = 300,   # applies to BOTH H and P scatter points
    h_color:          str   = '#222222', # 'black',
    p_color:          str   = '#ffffff', # 'white',
    p_edgecolor:      str   = 'black',
    p_edgecolor_lw:   float = 1.5,   # New: Control P marker edge width
    background_color: str   = 'whitesmoke',
    annotate_pad:     float = 0.3,
    annotate_fontsize: float = 12,
    grid_color:       str   = 'gray',
):
    """
    Plot HP‐chain on a 2D grid with:
      - single marker_size for all residues
      - distinct h_color / p_color
      - automatic contrasting label color for S/E
    """
    if end_idx is None:
        end_idx = len(seq) - 1

    def contrasting_text_color(bg_color: str) -> str:
        rgb = mcolors.to_rgb(bg_color)
        lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        return 'black' if lum > 0.5 else 'white'

    # fig, ax = plt.subplots(figsize=(6,6))
    ax.set_facecolor(background_color)
    if title:
        ax.set_title(title)

    # 1) backbone
    # Plot entire backbone at once for potential minor efficiency and cleaner code
    ax.plot(coords[:,0], coords[:,1],
            '-', color=backbone_color,
            linewidth=backbone_lw,
            zorder=1) # zorder: backbone at the bottom

    # 2) nodes (H vs P)
    for i, kind in enumerate(seq):
        x, y = coords[i]
        if kind.upper() == 'H':
            ax.scatter(x, y,
                       s=marker_size,
                       c=h_color,
                       edgecolors=h_color, # H markers are solid
                       zorder=3) # zorder: markers above backbone and contacts
        else: # kind == 'p'
            ax.scatter(x, y,
                       s=marker_size,
                       facecolors=p_color,
                       edgecolors=p_edgecolor,
                       linewidths=p_edgecolor_lw, # Use new parameter
                       zorder=3) # zorder: markers above backbone and contacts

    # 3) H–H contacts (non‐adjacent)
    # for i,j in find_hh_contacts(seq, coords):
    hh_contacts = find_hh_contacts(seq, coords)
    for i,j in hh_contacts:
        x0, y0 = coords[i]
        x1, y1 = coords[j]
        ax.plot([x0, x1], [y0, y1],
                hh_contact_ls,
                color=hh_contact_color,
                linewidth=hh_contact_lw,
                zorder=2) # zorder: contacts above backbone, below markers

    # 4) annotate start (S) & end (E) with contrasting text
    for idx, label in ((start_idx, 'S'), (end_idx, 'E')):
        # Determine the base color of the marker S/E is annotating
        marker_actual_color = h_color if seq[idx]=='H' else p_color
        txt_col  = contrasting_text_color(marker_actual_color)

        ax.text(*coords[idx],
                label,
                fontsize=annotate_fontsize, #marker_size * 0.12, # Font size relative to marker_size
                color=txt_col,
                ha='center', va='center',
                bbox=dict(facecolor=marker_actual_color, #annotate_boxcolor, # Color of the circle around S/E
                          edgecolor=marker_actual_color, # Make bbox edge same as face
                          boxstyle=f'circle,pad={annotate_pad}'),
                zorder=4) # zorder: annotations on top of everything

    # 5) grid & limits
    xmin, xmax = coords[:,0].min()-1, coords[:,0].max()+1
    ymin, ymax = coords[:,1].min()-1, coords[:,1].max()+1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks(range(int(np.floor(xmin)), int(np.ceil(xmax))+1))
    ax.set_yticks(range(int(np.floor(ymin)), int(np.ceil(ymax))+1))
    ax.grid(True, linestyle=':', color=grid_color, linewidth=0.5)

    # add text box top-right corner about hh contacts number
    num_contacts = len(hh_contacts)
    ax.text(xmax - 0.5, ymax - 0.5, f"#H-H: {num_contacts}", fontsize=12, ha='right', va='top', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))   

    # plt.tight_layout()
    # plt.show()


def _plot_hp_chain_3d(
    seq: List[str],
    coords: np.ndarray,  # Shape (L, 3)
    ax, 
    start_idx:       int    = 0,
    end_idx:         int    = None,
    # styling ↓
    title:            str   = "3D HP Folded Chain",
    backbone_color:   str   = 'dimgray',
    backbone_lw:      float = 2.0,
    hh_contact_color: str   = 'crimson',
    hh_contact_ls:    str   = '--',
    hh_contact_lw:    float = 2.0,
    marker_size:      float = 300,  # Adjusted default for 3D
    h_color:          str   = '#222222',
    p_color:          str   = '#ffffff',
    hp_edgecolor:      str   = 'black',
    hp_edgecolor_lw:   float = 1.5,
    background_color: str   = 'whitesmoke',
    annotate_pad:     float = 0.1,
    annotate_fontsize:float = 12,
    grid_color:       str   = 'gray',
    annotate_s_e:     bool  = True, # Toggle for S/E annotations
):
    """
    Plot HP‐chain on a 3D grid.
    """
    if end_idx is None:
        end_idx = len(seq) - 1
    if coords.shape[1] != 3:
        raise ValueError("Coordinates must be 3D (shape L,3)")

    # fig = plt.figure(figsize=(8, 8)) # May need to adjust for optimal view
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(background_color)
    if title:
        ax.set_title(title)

    # 1) Backbone
    ax.plot(coords[:,0], coords[:,1], coords[:,2], '-',
            color=backbone_color, linewidth=backbone_lw, zorder=1, label='Backbone')

    # 2) Nodes (H vs P)
    h_indices = [i for i, res_type in enumerate(seq) if res_type.lower() == 'h']
    p_indices = [i for i, res_type in enumerate(seq) if res_type.lower() == 'p']

    if h_indices:
        h_coords = coords[h_indices, :]
        ax.scatter(h_coords[:,0], h_coords[:,1], h_coords[:,2],
                   s=marker_size, 
                   c=h_color, 
                   edgecolors=h_color,
                   label='H (hydrophobic)', 
                   zorder=3, 
                   depthshade=False
                   )
    if p_indices:
        p_coords = coords[p_indices, :]
        ax.scatter(p_coords[:,0], p_coords[:,1], p_coords[:,2],
                   s=marker_size, 
                   facecolors=p_color,
                   edgecolors=hp_edgecolor,
                   linewidths=hp_edgecolor_lw, 
                   label='P (polar)', 
                   zorder=3, 
                   depthshade=False
                   )

    # 3) H–H contacts (non‐adjacent)
    contacts_3d = find_hh_contacts_3d(seq, coords)
    for i,j in contacts_3d:
        ax.plot([coords[i,0], coords[j,0]],
                [coords[i,1], coords[j,1]],
                [coords[i,2], coords[j,2]],
                linestyle=hh_contact_ls, color=hh_contact_color,
                linewidth=hh_contact_lw, zorder=2,
                label='H–H contact' if i == contacts_3d[0][0] else "" # Only label once
                )

    # 4) Annotate Start (S) & End (E)
    if annotate_s_e:
        for idx, label_char in ((start_idx, 'S'), (end_idx, 'E')):
            marker_actual_color = h_color if seq[idx].lower() == 'h' else p_color
            txt_col = contrasting_text_color(marker_actual_color)
            if 0 <= idx < len(coords): # Check index bounds
                ax.text(coords[idx,0], coords[idx,1], coords[idx,2]- 0.1, # Slight offset for visibility
                        label_char,
                        color=txt_col,
                        fontsize=annotate_fontsize,
                        ha='center', va='bottom', 
                        bbox=dict(facecolor=marker_actual_color, #annotate_boxcolor,
                                  edgecolor=marker_actual_color,
                                  boxstyle=f'circle,pad={annotate_pad}'
                                ),
                        zorder=4)

    # 5) Grid, Labels & Limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set aspect ratio to be equal, creating a cubic bounding box for unit lattice
    # Calculate ranges for each axis
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    
    ax.set_xlim(x_min -1, x_max + 1)
    ax.set_ylim(y_min -1, y_max + 1)
    ax.set_zlim(z_min -1, z_max + 1)
    ax.set_xticks(np.arange(int(x_min), int(x_max)+1))
    ax.set_yticks(np.arange(int(y_min), int(y_max)+1))
    ax.set_zticks(np.arange(int(z_min), int(z_max)+1))
    ax.grid(True, linestyle=':', color=grid_color, linewidth=0.5)
    # ax.view_init(elev=20, azim=30)  # Adjust view angle for better visibility
    # Set equal aspect ratio for all axes   
    # ax.set_aspect('auto')  # 'auto' for 3D, or use 'equal' for 2D

    # For a visually 'equal' aspect ratio in 3D based on data ranges
    ax.set_box_aspect((np.ptp(coords[:,0]) or 1, 
                       np.ptp(coords[:,1]) or 1, 
                       np.ptp(coords[:,2]) or 1))


    # Add a legend if H or P residues were plotted
    if h_indices or p_indices:
        ax.legend(
            markerscale=220/300,    # scale factor for legend markers
            scatterpoints=1,       # how many points per scatter legend entry
            fontsize=annotate_fontsize
        )
        

def render_hp_chain_ascii(
    seq: List[str],
    coords: np.ndarray,
    fill_char: str = '•',
    pad: int = 1,
    legend: Optional[str] = None,
    
) -> str:
    """
    Generate a text-based visualization of a 2D HP chain on its lattice,
    complete with backbone connectors, start/end markers, padding, and a legend.

    Args:
        seq: List of 'h' or 'p' representing the sequence.
        coords: numpy array of shape (L, 2) of integer coordinates for each residue.
        fill_char: Character to use for empty cells (default: '•').
        pad: Number of layers of padding around the structure (default: 1).

    Returns:
        A multiline string representing the grid with backbone, start/end,
        and a legend indicating types.
    """
    # at the top of your file
    SUPERSCTIPS = {
        'S': '\u02E2',  # ˢ
        'E': '\u1D49',  # ᵉ
        'H': '\u02B0',  # ʰ
        'P': '\u1D18',  # ᴘ
    }
    RED    = '\x1b[31m'
    BLUE   = '\x1b[34m'
    RESET  = '\x1b[0m'

    if not isinstance(coords, np.ndarray):
        coords = np.array(coords, dtype=int)
    if not isinstance(seq, list):
        seq = list(seq)

    L = coords.shape[0]  # L = len(seq)

    xs, ys = coords[:, 0], coords[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    # Initialize cell grid with fill_char
    grid = [[fill_char for _ in range(width)] for _ in range(height)]
    pos_map = {(x, y): i for i, (x, y) in enumerate(coords)}

    # Place residues and start/end markers
    for (x, y), idx in pos_map.items():
        r = ymax - y
        c = x - xmin
        if idx == 0:
            # grid[r][c] = 'S'
            _type = seq[idx].upper()
            mark = SUPERSCTIPS['S']
            color = RED if _type == 'H' else BLUE
            grid[r][c] = f"{color}{_type}{mark}{RESET}"
        elif idx == L - 1:
            # grid[r][c] = 'E'
            _type = seq[idx].upper()
            mark = SUPERSCTIPS['E']
            color = RED if _type == 'H' else BLUE
            grid[r][c] = f"{color}{_type}{mark}{RESET}"
        else:
            grid[r][c] = seq[idx]

    # Expand grid for connectors
    exp_h = height * 2 - 1
    exp_w = width * 2 - 1
    exp_grid = [[fill_char for _ in range(exp_w)] for _ in range(exp_h)]

    # Copy cells
    for r in range(height):
        for c in range(width):
            exp_grid[r * 2][c * 2] = grid[r][c]

    # Draw backbone connectors
    for i in range(L - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        r1, c1 = (ymax - y1) * 2, (x1 - xmin) * 2
        r2, c2 = (ymax - y2) * 2, (x2 - xmin) * 2
        dr, dc = r2 - r1, c2 - c1
        if dr == 0:
            exp_grid[r1][c1 + (dc // 2)] = '-'
        else:
            exp_grid[r1 + (dr // 2)][c1] = '|'

    # Add padding
    total_h = exp_h + 2 * pad
    total_w = exp_w + 2 * pad
    padded = [[fill_char] * total_w for _ in range(total_h)]
    for r in range(exp_h):
        for c in range(exp_w):
            padded[r + pad][c + pad] = exp_grid[r][c]

    # Compose lines
    lines = [''.join(row) for row in padded]

    if isinstance(legend, str):
        return '\n'.join(lines) + '\t▶ ' + legend + '\n'
    # if legend is not None:
    #     # Legend for start/end types
    #     start_type = seq[0].upper()
    #     end_type = seq[-1].upper()
    #     legend = (
    #         f"Legend: S = start ({start_type}),"
    #         f" E = end ({end_type}),"
    #         " '-' and '|' are backbone connectors"
    #     )
    #     return '\n'.join(lines) + '\t▶ ' + legend + '\n'
    elif legend == True:
        # compute energy
        contacts = find_hh_contacts(seq, coords)
        return '\n'.join(lines) + '\t▶ ' + str(len(contacts)) + ' H-H contacts\n'
    else:
        return '\n'.join(lines) + '\n'


def plot_and_export(seq, coords,
                    three_d=False, 
                    mode='human',
                    savepath=None,
                    display=False, 
                    **style_kwargs
                    ):
    """
    seq, coords: 
    three_d: whether to do 3D
    mode: 'human' (show), or 'rgb_array' (return numpy array)
    savepath: if given, save a PNG there.
    returns: None or np.ndarray (H×W×3) if mode='rgb_array'
    """
    if not isinstance(seq, list):
        seq = list(seq)
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must be a 2D array with shape (L,2) or (L,3)")
    if len(seq) != coords.shape[0]:
        raise ValueError("Length of seq and coords must match")
    
    if three_d:
        fig = plt.figure(figsize=style_kwargs.pop('figsize', (8,8)))
        ax = fig.add_subplot(111, projection='3d')
        _plot_hp_chain_3d(seq, coords, ax, **style_kwargs)
    else:
        fig, ax = plt.subplots(figsize=style_kwargs.pop('figsize', (6,6)))
        _plot_hp_chain(seq, coords, ax, **style_kwargs)

    fig.tight_layout()

    if mode == 'human':
        plt.show()
        if savepath:
            fig.savefig(savepath)
        plt.close(fig)
        return None

    elif mode == 'rgb_array':
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        arr = np.array(img)
        if savepath:
            img.save(savepath)
        # if display:
        #     return fig

        buf.close()
        plt.close(fig)
        return arr

    else:
        raise ValueError(f"Unknown render mode: {mode}")
   


# Example usage
def _example_text():
    # seq = list('hphpphhphpphphhpphph')
    seq = "HPHPPHHPHPPHPHHPPHPH"
    coords = np.array([
        [ 0,  0], [ 1,  0], [ 1, -1], [ 1, -2],
        [ 0, -2], [ 0, -1], [-1, -1], [-1, -2],
        [-2, -2], [-3, -2], [-3, -1], [-2, -1],
        [-2,  0], [-1,  0], [-1,  1], [-2,  1],
        [-2,  2], [-1,  2], [ 0,  2], [ 0,  1],
    ])
    print(render_hp_chain_ascii(seq, coords, fill_char=".", pad=1))
 

if __name__ == "__main__":
    # --- 1) Expand sequence
    expr = "(hp)2ph(hp)2(ph)2hp(ph)2"
    original_seq  = expand_sequence(expr) # Use a different name to avoid confusion
    L    = len(original_seq)
    print(f"Original sequence length = {L}, seq = {''.join(original_seq)}")

    # --- 2) Define moves and generate coordinates
    # (0,0) is residue 0
    # moves[i] is the vector from residue i to residue i+1
    moves = [
        (1, 0),  # 0→1
        (0, -1), # 1→2
        (0, -1), # 2→3
        (-1, 0), # 3→4
        (0, 1),  # 4→5
        (-1, 0), # 5→6
        (0, -1), # 6→7
        (-1, 0), # 7→8
        (-1, 0), # 8→9
        (0, 1),  # 9→10
        (1, 0),  # 10→11
        (0, 1),  # 11→12
        (1, 0),  # 12→13
        (0, 1),  # 13→14
        (-1, 0), # 14→15
        (0, 1),  # 15→16
        (1, 0),  # 16→17
        (1, 0),  # 17→18
        (0, -1), # 18→19
    ]

    # Walk the chain and record coordinates:
    # The first residue is at (0,0)
    current_coords = [(0, 0)] # List to store coordinates, starting with the first residue
    _seq = [original_seq[0]] # Start with the first residue
    for i, (dx, dy) in enumerate(moves):
        x, y = current_coords[-1] # Get the last added coordinates
        
        print(f"s: ({_seq[-1]}, {current_coords[-1]}), action: {dx, dy} -> ", end="")
        current_coords.append((x + dx, y + dy))
        _seq.append(original_seq[i + 1])

        print(f"s`: ({_seq[-1]}, {current_coords[-1]})")
        _line = render_hp_chain_ascii(_seq, np.array(current_coords), fill_char=".", pad=1)

        print(_line)
   
    coords_np = np.array(current_coords)
    print(f"Coordinates after walking:\n{coords_np}")

    # The sequence for plotting is the original_seq
    # The previous logic for _seq was just rebuilding original_seq
    # print(f"Sequence after walking: {''.join(original_seq)} ({len(original_seq)})")
    # for i, (x, y) in enumerate(coords_np):
    #     print(f"Residue {i:2d} ({original_seq[i]}): ({x:2d}, {y:2d})")

    assert coords_np.shape == (len(original_seq), 2), "coords must be (L,2)"
    assert len(moves) == len(original_seq) - 1, "Number of moves must be L-1"


    # --- 3) Plot with full control
    # _plot_hp_chain(
    #     original_seq, coords_np, # Pass the original sequence and calculated coordinates
    # )
    plot_and_export(
        original_seq, coords_np,
        three_d=False,
        mode='human',
        savepath=None,
        display=True,
    )
    
    print("\n--- 3D Plotting Demo ---")
    
    # Example 1: Simple L-shape in 3D planes
    seq_3d_1 = expand_sequence("HHPPHH") # L=6
    coords_3d_1 = np.array([
        [0,0,0], # H0
        [1,0,0], # H1
        [1,1,0], # P2
        [1,1,1], # P3
        [2,1,1], # H4 (Potential contact H1-H4 if H4 was e.g. [1,0,1])
        [2,2,1]  # H5
    ])
    print(f"Sequence 1: {''.join(seq_3d_1)} {seq_3d_1 = }")
    print(f"Coordinates 1:\n{coords_3d_1}")
    # _plot_hp_chain_3d(seq_3d_1, coords_3d_1, title="3D HP Fold - Example 1 (L-shape)")
    plot_and_export(
        seq_3d_1, coords_3d_1,
        three_d=True,
        mode='human',
        savepath=None,
        display=True,
    )


    # Example 3: A slightly longer chain
    seq_3d_3 = expand_sequence("HPHPPHHPHP") # L=10
    # Manually define some non-overlapping coordinates for this 10-mer
    coords_3d_3 = np.array([
        [0,0,0], # H
        [1,0,0], # P
        [1,1,0], # H
        [1,1,1], # P
        [0,1,1], # P
        [0,2,1], # H - Contact with H at [1,1,0] if moves allow (e.g. if H was at [0,1,0])
                 # Current H2 is [1,1,0]. H5 is [0,2,1]. Dist: |0-1|+|2-1|+|1-0| = 1+1+1=3. No contact.
                 # Let's make a contact for H5: H2(1,1,0), P3(1,0,0), P4(0,0,0), H5(0,1,0) - Contact H2-H5
        [0,2,2], # H
        [1,2,2], # P
        [1,3,2], # H - Contact with H at [0,2,2]
        [1,3,1]  # P
    ])
    
 # Example 3: A slightly longer chain
    seq_3d_3 = expand_sequence("HPHPPHHPHP") # L=10
    # Manually define some non-overlapping coordinates for this 10-mer
    coords_3d_3 = np.array([
        [0,0,0], # H
        [1,0,0], # P
        [1,1,0], # H
        [1,1,1], # P
        [0,1,1], # P
        [0,2,1], # H - Contact with H at [1,1,0] if moves allow (e.g. if H was at [0,1,0])
                 # Current H2 is [1,1,0]. H5 is [0,2,1]. Dist: |0-1|+|2-1|+|1-0| = 1+1+1=3. No contact.
                 # Let's make a contact for H5: H2(1,1,0), P3(1,0,0), P4(0,0,0), H5(0,1,0) - Contact H2-H5
        [0,2,2], # H
        [1,2,2], # P
        [1,3,2], # H - Contact with H at [0,2,2]
        [1,3,1]  # P
    ])
    # Corrected coordinates for seq_3d_3 to show contacts:
    # H0(0,0,0) P1(1,0,0) H2(1,1,0) P3(2,1,0) P4(2,0,0) H5(2,0,1) H6(1,0,1) P7(0,0,1) H8(0,1,1) P9(0,1,0)
    # H0(0,0,0), H8(0,1,1) -> d=2 no
    # H2(1,1,0), H5(2,0,1) -> d=3 no
    # H2(1,1,0), H6(1,0,1) -> d=2 no
    # H2(1,1,0), H8(0,1,1) -> d=2 no
    # H5(2,0,1), H8(0,1,1) -> d=3 no
    # H6(1,0,1), H8(0,1,1) -> d=1 YES! Contact (idx 6, idx 8)
    
    coords_3d_3_contact = np.array([
        [0,0,0], #0 H
        [1,0,0], #1 P
        [1,1,0], #2 H
        [2,1,0], #3 P
        [2,0,0], #4 P
        [2,0,1], #5 H
        [1,0,1], #6 H 
        [0,0,1], #7 P
        [0,1,1], #8 H - Contact with H6: |0-1|+|1-0|+|1-1|=1+1+0=2. No. This should be 1.
                 # Let H8 be at [1,1,1]. Then |1-1|+|1-0|+|1-1|=1. YES.
        [0,1,0]  #9 P
    ])
    coords_3d_3_contact[8] = [1,1,1] # H8 to make contact with H6 [1,0,1]
    
    print(f"\nSequence 3: {''.join(seq_3d_3)}")
    print(f"Coordinates 3 (with intended H6-H8 contact):\n{coords_3d_3_contact}")
    # _plot_hp_chain_3d(seq_3d_3, coords_3d_3_contact, title="3D HP Fold - Example 3 (10-mer)")
    plot_and_export(
        seq_3d_3, coords_3d_3_contact,
        three_d=True,
        mode='human',
        savepath=None,
        display=True,
        title="3D HP Fold - Example 3 (10-mer)",
        annotate_s_e=True, # Show S/E annotations
    )