"""
04_check_active_site.py

For each hypothetical protein that has a lyase/heparinase FoldSeek hit,
structurally align it to the reference PDB and check whether the active-site
residue composition supports beta-elimination (Tyr/His as base + Arg/Lys for
carboxylate neutralisation) vs hydrolysis (Asp/Glu catalytic pair).

Usage:
    python PULpredSVM/04_check_active_site.py --task gag
    python PULpredSVM/04_check_active_site.py --task alginate

IMPORTANT — candidate selection policy:
    Active-site verdict (beta_elim_likely / lyase_possible) is a necessary but
    NOT sufficient condition for inclusion in the final candidate list.  Proteins
    must also come from a CGC whose GAG-SVM score >= 0.5 (all_cgc_gag_scores.csv).
    Proteins from sub-threshold CGCs are excluded regardless of active-site result,
    because the CGC context (transporter type, co-enzyme composition) is required
    to confirm GAG substrate specificity.  The SVM cutoff is the primary gate;
    active-site chemistry is the secondary confirmation.
"""

import argparse
import subprocess
import urllib.request
import re
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

TMALIGN     = "/work3/zhayu/envs/pulpred/bin/TMalign"
STRUCT_BASE = Path("PULpredSVM/results")

# Keywords that flag a hit as a lyase candidate
LYASE_KW = {"lyase", "heparinase", "heparanase", "chondroitinase",
             "sulfatase", "sulfohydrolase"}

# Residues associated with beta-elimination (PL family) active sites
BETA_ELIM = {"Y", "H"}          # catalytic base
NEUTRALISE = {"R", "K"}         # carboxylate neutralisation
HYDROLASE  = {"D", "E"}         # catalytic acid/base in GHs

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
       'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
       'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


# ── PDB helpers ──────────────────────────────────────────────────────────────

def parse_hit_id(hit_str: str):
    """
    Parse FoldSeek hit string like '8r70-assembly1_A' or 'AF-Q4QKA6-F1-model_v6'.
    Returns (pdb_id, chain) or (None, None) for AlphaFold entries.
    """
    # AlphaFold hits
    if hit_str.startswith("AF-"):
        return None, None
    # PDB hits: XXXX-assemblyN_CHAIN
    m = re.match(r'^([0-9a-zA-Z]{4})-assembly\d+_([A-Za-z0-9])$', hit_str)
    if m:
        return m.group(1).upper(), m.group(2)
    return None, None


def download_pdb_chain(pdb_id: str, chain: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode(errors="replace")
    except Exception as e:
        print(f"    Download failed for {pdb_id}: {e}")
        return False
    # Extract chain — include HETATM (ligands) but not water
    lines = []
    for l in data.splitlines():
        if l.startswith("ATOM") and l[21] == chain:
            lines.append(l)
        elif l.startswith("HETATM") and l[21] == chain:
            resname = l[17:20].strip()
            if resname not in ("HOH", "WAT", "DOD"):
                lines.append(l)
        elif l.startswith("END"):
            lines.append(l)
    if not lines:
        print(f"    No ATOM records for chain {chain} in {pdb_id}")
        return False
    out_path.write_text("\n".join(lines) + "\n")
    return True


def load_ca_residues(pdb_path: Path):
    """Return list of (resnum, aa_1letter) for CA atoms, in order."""
    res = []; seen = set()
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            rn = int(line[22:26])
            if rn not in seen:
                res.append((rn, AA3.get(line[17:20].strip(), "?")))
                seen.add(rn)
    return res


def add_chain_id(src: Path, dst: Path, chain="A"):
    """Rewrite PDB adding chain ID at col 22 (ESMFold output lacks it)."""
    if dst.exists():
        return
    lines = []
    for line in src.read_text().splitlines():
        if line.startswith("ATOM") or line.startswith("TER"):
            line = line[:21] + chain + line[22:]
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


# ── TMalign ──────────────────────────────────────────────────────────────────

def run_tmalign(query_pdb: Path, ref_pdb: Path):
    """Return (tm_score, rmsd, seq_id, pairs) where pairs = list of
    (q_resnum, q_aa, r_resnum, r_aa, match_char)."""
    result = subprocess.run(
        [TMALIGN, str(query_pdb), str(ref_pdb)],
        capture_output=True, text=True
    )
    out = result.stdout

    tm = rmsd = seqid = None
    for line in out.splitlines():
        if "TM-score=" in line and "normalized by length of Structure_2" in line:
            tm = float(re.search(r"TM-score=\s*([\d.]+)", line).group(1))
        if "RMSD=" in line and "Aligned length" in line:
            rmsd   = float(re.search(r"RMSD=\s*([\d.]+)", line).group(1))
            seqid  = float(re.search(r"Seq_ID=\S+=\s*([\d.]+)", line).group(1))

    lines = out.splitlines()
    aln_idx = next((i for i, l in enumerate(lines) if "denotes residue pairs" in l), None)
    if aln_idx is None:
        return tm, rmsd, seqid, []

    seq1  = lines[aln_idx + 1]
    match = lines[aln_idx + 2]
    seq2  = lines[aln_idx + 3]
    return tm, rmsd, seqid, (seq1, match, seq2)


def build_pairs(aln, qry_res, ref_res):
    seq1, match, seq2 = aln
    q_idx = r_idx = 0
    pairs = []
    for c1, cm, c2 in zip(seq1, match, seq2):
        if c1 != '-' and c2 != '-':
            if q_idx < len(qry_res) and r_idx < len(ref_res):
                pairs.append((qry_res[q_idx][0], qry_res[q_idx][1],
                               ref_res[r_idx][0],  ref_res[r_idx][1], cm))
            q_idx += 1; r_idx += 1
        elif c1 != '-': q_idx += 1
        elif c2 != '-': r_idx += 1
    return pairs


# ── Active-site residue identification ───────────────────────────────────────

def get_ligand_contact_residues(pdb_path: Path, dist=6.0):
    """Residues within `dist` Å of any HETATM ligand (excluding water)."""
    import math

    atom_coords = {}    # resnum -> list of (x,y,z)
    ligand_coords = []

    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and len(line) >= 54:
            rn = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atom_coords.setdefault(rn, []).append((x, y, z))
        elif line.startswith("HETATM") and len(line) >= 54:
            resname = line[17:20].strip()
            if resname not in ("HOH", "WAT", "DOD"):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ligand_coords.append((x, y, z))

    if not ligand_coords:
        return set()

    contacts = set()
    for rn, coords in atom_coords.items():
        for ax, ay, az in coords:
            for lx, ly, lz in ligand_coords:
                d = math.sqrt((ax-lx)**2 + (ay-ly)**2 + (az-lz)**2)
                if d <= dist:
                    contacts.add(rn)
                    break
            if rn in contacts:
                break
    return contacts


def active_site_residues(pdb_path: Path, ca_res: list):
    """
    Best-effort active site residues from reference PDB.
    Priority: ligand contacts → geometric centre proximity.
    Returns set of residue numbers.
    """
    contacts = get_ligand_contact_residues(pdb_path)
    if contacts:
        return contacts

    # Fallback: residues closest to geometric centre (top 30)
    import math
    coords = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and len(line) >= 54:
            rn = int(line[22:26])
            coords[rn] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if not coords:
        return set()
    cx = sum(v[0] for v in coords.values()) / len(coords)
    cy = sum(v[1] for v in coords.values()) / len(coords)
    cz = sum(v[2] for v in coords.values()) / len(coords)
    ranked = sorted(coords.keys(),
                    key=lambda r: math.sqrt((coords[r][0]-cx)**2 +
                                             (coords[r][1]-cy)**2 +
                                             (coords[r][2]-cz)**2))
    return set(ranked[:30])


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_active_site(pairs, ref_active_site):
    """
    Given aligned pairs and reference active-site residue numbers,
    count beta-elim indicators vs hydrolase indicators in the QUERY.
    Returns dict with counts and a verdict.
    """
    q_aas = [qa for qr, qa, rr, ra, m in pairs
             if rr in ref_active_site and m in (':', '.')]

    base_count    = sum(1 for a in q_aas if a in BETA_ELIM)
    neutral_count = sum(1 for a in q_aas if a in NEUTRALISE)
    hydro_count   = sum(1 for a in q_aas if a in HYDROLASE)
    total         = len(q_aas)

    if total == 0:
        verdict = "no_data"
    elif base_count >= 1 and neutral_count >= 1 and hydro_count <= base_count + neutral_count:
        # Y/H present (catalytic base) + R/K present (carboxylate neutralisation)
        # and D/E doesn't outnumber the lyase residues
        verdict = "beta_elim_likely"
    elif hydro_count >= 2 and base_count == 0 and neutral_count == 0:
        verdict = "hydrolase_likely"
    elif base_count >= 2 and neutral_count == 0:
        # Strong Y/H signal but no R/K — weaker lyase evidence
        verdict = "lyase_possible"
    else:
        verdict = "ambiguous"

    return {
        "active_site_residues_aligned": total,
        "beta_elim_base (Y/H)": base_count,
        "neutralisation (R/K)": neutral_count,
        "hydrolase (D/E)": hydro_count,
        "query_active_site_seq": "".join(q_aas),
        "verdict": verdict,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gag", "alginate"])
    ap.add_argument("--top_hits", type=int, default=5,
                    help="Check up to this many lyase hits per protein")
    args = ap.parse_args()

    res_dir    = STRUCT_BASE / f"{args.task}_minprot5"
    hits_tsv   = res_dir / "hypothetical_foldseek_hits.tsv"
    struct_dir = res_dir / "structures"
    pdb_cache  = res_dir / "pdb_cache"
    pdb_cache.mkdir(exist_ok=True)

    if not hits_tsv.exists():
        raise FileNotFoundError(f"Run 03_fold_structure.py first: {hits_tsv}")

    import pandas as pd
    hits = pd.read_csv(hits_tsv, sep="\t")

    # Filter to lyase/heparinase hits
    mask = hits["hit_description"].str.lower().apply(
        lambda d: any(kw in d for kw in LYASE_KW)
    )
    lyase_hits = hits[mask].copy()
    print(f"Lyase/heparinase hits: {len(lyase_hits)} rows across "
          f"{lyase_hits['protein_id'].nunique()} proteins\n")

    rows = []

    for pid, grp in lyase_hits.groupby("protein_id", sort=False):
        grp = grp.head(args.top_hits)
        print(f"── {pid} ({len(grp)} lyase hits)")

        # Find ESMFold structure
        pdb_stem    = pid.replace("|", "_").replace("/", "_")
        query_src   = struct_dir / f"{pdb_stem}.pdb"
        query_chain = res_dir / f"{pdb_stem}_chainA.pdb"
        if not query_src.exists():
            print(f"   No ESMFold structure — skipping")
            continue
        add_chain_id(query_src, query_chain)
        qry_res = load_ca_residues(query_chain)

        for _, row in grp.iterrows():
            target     = row["target"]
            desc       = row["hit_description"]
            prob       = row["prob"]
            pdb_id, chain = parse_hit_id(target)

            print(f"   hit: {target}  prob={prob}")
            print(f"        {desc[:80]}")

            if pdb_id is None:
                print("   → AlphaFold entry — skipping (no experimental active site)")
                continue

            # Download reference PDB chain
            ref_pdb = pdb_cache / f"{pdb_id}_{chain}.pdb"
            if not download_pdb_chain(pdb_id, chain, ref_pdb):
                continue
            ref_res = load_ca_residues(ref_pdb)
            if not ref_res:
                print("   → empty PDB — skipping")
                continue

            # TMalign
            tm, rmsd, seqid, aln = run_tmalign(query_chain, ref_pdb)
            if not aln:
                print(f"   → TMalign failed")
                continue
            pairs = build_pairs(aln, qry_res, ref_res)
            print(f"   TMalign: TM={tm:.3f}  RMSD={rmsd:.2f}Å  SeqID={seqid:.2f}"
                  f"  aligned={len(pairs)}")

            if tm < 0.4:
                print("   → TM-score too low — structures not similar")
                continue

            # Active site
            active_resnums = active_site_residues(ref_pdb, ref_res)
            src = "ligand_contacts" if get_ligand_contact_residues(ref_pdb) else "geometric_centre"
            print(f"   Active site defined by: {src} ({len(active_resnums)} residues)")

            # Score
            sc = score_active_site(pairs, active_resnums)
            verdict_icon = {"beta_elim_likely": "✓ LYASE",
                            "hydrolase_likely":  "✗ HYDROLASE",
                            "ambiguous":         "? AMBIGUOUS",
                            "no_data":           "— NO DATA"}.get(sc["verdict"], sc["verdict"])
            print(f"   Active site (query): {sc['query_active_site_seq']}")
            print(f"   Y/H={sc['beta_elim_base (Y/H)']}"
                  f"  R/K={sc['neutralisation (R/K)']}"
                  f"  D/E={sc['hydrolase (D/E)']}"
                  f"  → {verdict_icon}\n")

            rows.append({
                "protein_id":   pid,
                "ref_pdb":      target,
                "ref_desc":     desc[:100],
                "foldseek_prob": prob,
                "tm_score":     tm,
                "rmsd":         rmsd,
                "seq_id":       seqid,
                "active_site_source": src,
                **sc,
            })

    if rows:
        import pandas as pd
        out = res_dir / "active_site_lyase_check.tsv"
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(f"\nResults saved → {out}")
        print(pd.DataFrame(rows)[["protein_id","ref_pdb","tm_score","verdict"]].to_string())
    else:
        print("No results.")


if __name__ == "__main__":
    main()
