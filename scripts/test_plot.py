#!/usr/bin/env python3
"""
plot_tk_dmeff.py

Scan a directory for CLASS transfer-function files named like:
  pk_m<VAL>_sig<VAL>_..._tk.dat
Parse m_dmeff and sigma from filenames, read the data (k and T_i(k)), and plot
selected species vs k.

Usage:
  python plot_tk_dmeff.py --indir ./outputs --species d_b d_g --outdir ./plots

Defaults:
  --indir .
  --species d_b d_cdm d_g  (plots only those found in the file)
  --outdir ./plots
"""
import argparse, os, re, glob, sys
import pandas as pd
import matplotlib.pyplot as plt


def parse_names_from_header(path):
    """
    Parse column names from CLASS header lines like:
    "#    1:k (h/Mpc)    2:d_g   3:d_b   ..."
    Returns list of names (index 0 corresponds to the first numeric column).
    If parsing fails, returns None.
    """
    name_map = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.lstrip().startswith("#"):
                    # Stop after header section
                    break
                # Find tokens like "2:d_g" or "1:k (h/Mpc)"
                for m in re.finditer(r'(\d+)\s*:\s*([A-Za-z0-9_]+)', line):
                    idx = int(m.group(1))
                    name = m.group(2)
                    name_map[idx] = name
    except Exception:
        return None
    if not name_map:
        return None
    # Convert to ordered list by index, 1-based to 0-based
    max_idx = max(name_map.keys())
    names = [None]*max_idx
    for i, nm in name_map.items():
        names[i-1] = nm
    # Some files include fewer/extra numeric columns; we may pad or trim after reading
    return names

def parse_params_from_filename(fname):
    """
    Extract m_dmeff and sigma_dmeff from filenames like:
      pk_m1.0e-05_sig1.0e-25_np2_tk.dat
    Returns dictionary with keys 'm' and 'sig' as strings (original tokens), and floats if parseable.
    """
    base = os.path.basename(fname)
    m_match = re.search(r'_m([0-9.+\-eE]+)', base)
    s_match = re.search(r'_sig([0-9.+\-eE]+)', base)
    info = {"m_token": None, "sig_token": None, "m_val": None, "sig_val": None}
    if m_match:
        info["m_token"] = m_match.group(1)
        try:
            info["m_val"] = float(info["m_token"])
        except Exception:
            pass
    if s_match:
        info["sig_token"] = s_match.group(1)
        try:
            info["sig_val"] = float(info["sig_token"])
        except Exception:
            pass
    return info

def read_data(path, names_hint=None):
    """
    Read whitespace-delimited data, skipping comment lines starting with '#'. 
    If names_hint provided, use it (trim/pad to actual width).
    """
    df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
    if names_hint:
        # pad/trim names to df width
        names = names_hint[:len(df.columns)]
        # fill any None with generic names
        names = [ (n if n is not None else f"col{i+1}") for i,n in enumerate(names) ]
        df.columns = names
    return df

def ensure_outdir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=".", help="Directory to scan for *_tk.dat files")
    ap.add_argument("--glob", default="pk_*_tk.dat", help="Glob pattern to match files")
    ap.add_argument("--species", nargs="*", default=["d_b","d_cdm","d_g"],
                    help="Species columns to plot if present (e.g., d_b d_g d_cdm)")
    ap.add_argument("--outdir", default="./plots", help="Directory to save PNGs")
    ap.add_argument("--loglog", action="store_true", help="Use log-log axes (default: semilogx on k)")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    files = sorted(glob.glob(os.path.join(args.indir, args.glob)))
    if not files:
        print(f"No files found under {args.indir!r} matching {args.glob!r}")
        sys.exit(1)

    for fp in files:
        names_hint = parse_names_from_header(fp)
        df = read_data(fp, names_hint=names_hint)

        # Determine available columns
        k_col = None
        for cand in ["k", "k_h/Mpc", "col1"]:
            if cand in df.columns:
                k_col = cand
                break
        if k_col is None:
            # fall back to first column
            k_col = df.columns[0]

        present = [s for s in args.species if s in df.columns]
        if not present:
            # if none of requested species present, try to guess common ones
            present = [c for c in ["d_b","d_cdm","d_idm","d_g"] if c in df.columns]
        if not present:
            print(f"Warning: no requested species found in {os.path.basename(fp)}; available columns: {list(df.columns)[:10]}...")
            continue

        meta = parse_params_from_filename(fp)
        title_bits = []
        if meta["m_token"] is not None: title_bits.append(f"m={meta['m_token']}")
        if meta["sig_token"] is not None: title_bits.append(f"sigma={meta['sig_token']}")
        title_suffix = ", ".join(title_bits) if title_bits else os.path.basename(fp)

        # Plot
        plt.figure()
        for sp in present:
            plt.plot(df[k_col], df[sp], label=sp)
        plt.xlabel(f"{k_col}")
        plt.ylabel("Transfer function T_i(k)")
        plt.title(f"T_i(k) @ {title_suffix}")
        plt.legend()
        if args.loglog:
            plt.xscale("log"); plt.yscale("log")
        else:
            plt.xscale("log")

        outname = os.path.splitext(os.path.basename(fp))[0] + ".png"
        outpath = os.path.join(args.outdir, outname)
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()
