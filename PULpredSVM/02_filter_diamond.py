#!/usr/bin/env python3
"""
Annotate novel GAG PUL candidates from SVM prediction results.

For each candidate PUL:
  1. Extract all proteins from PULDB.xlsx with CAZy annotations
  2. Classify: CAZy-annotated / unannotated
  3. Extract unannotated protein sequences from pul_proteins.faa
  4. Optionally run DIAMOND against CAZy characterized sequences (fast, sensitive)
  5. Optionally run hmmscan against Pfam-A (finds non-CAZy functional domains)
  6. Optionally run NCBI BLAST on unannotated proteins (web, slow)
  7. Output per-PUL annotation table + overall candidate summary

Setup for DIAMOND:
    conda install -c bioconda diamond
    wget https://bcb.unl.edu/dbCAN2/download/CAZyDB.07312023.fa
    diamond makedb --in CAZyDB.07312023.fa -d CAZyDB

Usage:
    # Annotation from PULDB.xlsx only (fast, no external tools)
    python PULpredSVM/02_filter_diamond.py --topn 20

    # + DIAMOND against CAZy (recommended for unannotated proteins)
    python PULpredSVM/02_filter_diamond.py --topn 20 --diamond_db /path/to/CAZyDB.dmnd

    # + hmmscan against Pfam (non-CAZy functional domains)
    python PULpredSVM/02_filter_diamond.py --topn 20 --hmm_db /path/to/Pfam-A.hmm

    # + NCBI web BLAST (slow, no setup required)
    python PULpredSVM/02_filter_diamond.py --topn 20 --blast
"""

import argparse
import re
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from Bio import SeqIO, SearchIO
from Bio.Blast import NCBIWWW, NCBIXML

# ── Constants ─────────────────────────────────────────────────────────────────

# CAZy families associated with GAG degradation (for highlighting)
GAG_CAZY = {"PL8","PL12","PL21","PL29","PL35","GH88","GH20","GH35","GH79",
            "GH91","GH105","CBM70","CE12"}

# Families that indicate polysaccharide lyase activity generally
PL_FAMILIES = lambda f: f.startswith("PL")
GH_FAMILIES = lambda f: f.startswith("GH")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates",  default="PULpredSVM/results/novel_gag_candidates.csv")
    p.add_argument("--puldb_xlsx",  default="data/raw/PULDB.xlsx")
    p.add_argument("--proteins_faa",default="data/raw/pul_proteins.faa")
    p.add_argument("--topn",        type=int, default=20,
                   help="Annotate top-N candidates from the CSV")
    p.add_argument("--out_dir",     default="PULpredSVM/results/annotated")
    p.add_argument("--diamond_db",  default=None,
                   help="Path to DIAMOND database (.dmnd) built from CAZyDB. "
                        "Recommended for annotating unannotated proteins.")
    p.add_argument("--diamond_bin", default="diamond")
    p.add_argument("--diamond_evalue", type=float, default=1e-5)
    p.add_argument("--diamond_max_hits", type=int, default=5)
    p.add_argument("--hmm_db",      default=None,
                   help="Path to Pfam-A.hmm (pressed). Finds non-CAZy functional "
                        "domains in unannotated proteins. Download from InterPro.")
    p.add_argument("--hmmscan_bin", default="hmmscan")
    p.add_argument("--blast",       action="store_true",
                   help="Run NCBI web BLAST on unannotated proteins (slow, no setup)")
    p.add_argument("--blast_db",    default="nr",
                   help="NCBI BLAST database (default: nr)")
    p.add_argument("--blast_max_hits", type=int, default=5)
    # ── Localization prediction ───────────────────────────────────────────────
    p.add_argument("--signalp_bin", default=None,
                   help="Path to SignalP binary (4.x / 5.0 / 6.0). "
                        "Predicts signal peptides on all GH+PL proteins.")
    p.add_argument("--signalp_org", default="gram-",
                   help="SignalP organism type: gram- / gram+ / euk  (default: gram-)")
    p.add_argument("--tmhmm_bin",   default=None,
                   help="Path to TMHMM 2.0 binary. "
                        "Predicts TM helices on all GH+PL proteins.")
    return p.parse_args()


# ── Sequence index ─────────────────────────────────────────────────────────────

def build_seq_index(faa_path: str) -> dict:
    print(f"Indexing {faa_path} …")
    index = {}
    for rec in SeqIO.parse(faa_path, "fasta"):
        if rec.id not in index:
            index[rec.id] = rec
    print(f"  {len(index):,} unique sequences loaded")
    return index


# ── PULDB annotation ──────────────────────────────────────────────────────────

def load_puldb(xlsx_path: str) -> pd.DataFrame:
    print("Loading PULDB.xlsx …")
    df = pd.read_excel(xlsx_path, dtype=str,
                       usecols=["Name","protein_id","protein_name","hmm"])
    df["hmm"] = df["hmm"].fillna("").str.strip()
    return df


def annotate_pul(pul_name: str, puldb: pd.DataFrame) -> pd.DataFrame:
    rows = puldb[puldb["Name"] == pul_name].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["class"] = rows["hmm"].apply(lambda h: "CAZy" if h else "unannotated")
    rows["is_pl"] = rows["hmm"].apply(
        lambda h: any(f.startswith("PL") for f in h.split(";") if f))
    rows["is_gag_related"] = rows["hmm"].apply(
        lambda h: bool(set(f.split("_")[0] for f in h.split(";") if f) & GAG_CAZY))
    return rows[["protein_id","protein_name","hmm","class","is_pl","is_gag_related"]]


# ── Sequence extraction ───────────────────────────────────────────────────────

def extract_sequences(protein_ids: list, seq_index: dict, out_faa: Path) -> list:
    found, missing = [], []
    records = []
    for pid in protein_ids:
        if pid in seq_index:
            records.append(seq_index[pid])
            found.append(pid)
        else:
            missing.append(pid)

    if records:
        SeqIO.write(records, out_faa, "fasta")
    if missing:
        print(f"    [warn] {len(missing)} proteins not in FASTA: {missing[:3]}{'…' if len(missing)>3 else ''}")
    return found


# ── hmmscan ───────────────────────────────────────────────────────────────────

def run_hmmscan(faa_path: Path, hmm_db: str, hmmscan_bin: str,
                out_dir: Path) -> pd.DataFrame:
    tbl_out = out_dir / (faa_path.stem + "_hmmscan.tbl")
    cmd = [hmmscan_bin, "--noali", "--cpu", "4",
           "--domtblout", str(tbl_out),
           hmm_db, str(faa_path)]
    print(f"    Running hmmscan … ({faa_path.name})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [warn] hmmscan failed: {result.stderr[:200]}")
        return pd.DataFrame()

    rows = []
    if tbl_out.exists():
        for line in tbl_out.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            rows.append({
                "protein_id": parts[3],
                "hmm_name":   parts[0],
                "evalue":     float(parts[11]),
                "score":      float(parts[13]),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("evalue").drop_duplicates("protein_id")
    return df


# ── DIAMOND ──────────────────────────────────────────────────────────────────

def run_diamond(faa_path: Path, db: str, diamond_bin: str,
                evalue: float, max_hits: int, out_dir: Path) -> pd.DataFrame:
    tsv_out = out_dir / (faa_path.stem + "_diamond.tsv")
    cmd = [
        diamond_bin, "blastp",
        "-q", str(faa_path),
        "-d", db,
        "-o", str(tsv_out),
        "--evalue", str(evalue),
        "--max-target-seqs", str(max_hits),
        "--outfmt", "6",
        "qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "stitle",
        "--sensitive",
        "--threads", "4",
        "--quiet",
    ]
    print(f"    Running DIAMOND … ({faa_path.name})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [warn] DIAMOND failed: {result.stderr[:300]}")
        return pd.DataFrame()

    if not tsv_out.exists() or tsv_out.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(tsv_out, sep="\t", header=None,
                     names=["protein_id","hit_id","pident","length",
                            "evalue","bitscore","hit_title"])
    # Keep best hit per query protein
    df = df.sort_values("evalue").drop_duplicates("protein_id")
    # Official CAZyDB header: >ACCESSION|FAMILY_SUBFAMILY  (e.g. QDZ53590.1|GH31_13)
    # hit_id (sseqid) = "QDZ53590.1|GH31_13", family is after the pipe; strip subfamily suffix
    def _parse_family(hit_id: str) -> str:
        if not isinstance(hit_id, str) or "|" not in hit_id:
            return "—"
        raw = hit_id.split("|", 1)[1]          # e.g. "GH31_13"
        return re.split(r"_\d+$", raw)[0]      # e.g. "GH31"
    df["cazy_family"] = df["hit_id"].apply(_parse_family)
    return df[["protein_id","cazy_family","pident","evalue","bitscore","hit_id","hit_title"]]


# ── NCBI BLAST ────────────────────────────────────────────────────────────────

def run_ncbi_blast(faa_path: Path, db: str, max_hits: int) -> pd.DataFrame:
    records = list(SeqIO.parse(faa_path, "fasta"))
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        print(f"    BLAST: {rec.id} ({len(rec.seq)} aa) …", end=" ", flush=True)
        try:
            handle = NCBIWWW.qblast("blastp", db, str(rec.seq),
                                     hitlist_size=max_hits, expect=1e-5)
            blast_records = list(NCBIXML.parse(handle))
            hits = []
            for br in blast_records:
                for aln in br.alignments[:max_hits]:
                    hsp = aln.hsps[0]
                    hits.append({
                        "protein_id":  rec.id,
                        "hit_id":      aln.accession,
                        "hit_title":   aln.title[:80],
                        "identity":    round(hsp.identities / hsp.align_length * 100, 1),
                        "evalue":      hsp.expect,
                        "score":       hsp.score,
                    })
            if hits:
                rows.extend(hits)
                print(f"top hit: {hits[0]['hit_title'][:50]} ({hits[0]['identity']}%)")
            else:
                print("no hits")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(2)   # NCBI rate limit

    return pd.DataFrame(rows)


# ── SignalP ───────────────────────────────────────────────────────────────────

def _sp_detect_version(signalp_bin: str) -> int:
    """Return SignalP major version integer (4, 5, or 6)."""
    for flag in ("-V", "--version", "-version"):
        try:
            r = subprocess.run([signalp_bin, flag], capture_output=True, text=True, timeout=8)
            text = r.stdout + r.stderr
            m = re.search(r'SignalP[-\s]+(\d)', text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return 4  # default assumption


def _sp_org_flag(org: str, version: int) -> str:
    """Translate organism flag for the given SignalP version."""
    if version >= 5:
        mapping = {"gram-": "gram_neg", "gram+": "gram_pos", "euk": "euk"}
        return mapping.get(org, org)
    return org  # v4: gram- / gram+ / euk


def run_signalp(faa_path: Path, signalp_bin: str, org: str,
                out_dir: Path) -> pd.DataFrame:
    """Run SignalP (v4.x / 5.x / 6.x) and return SP predictions."""
    version = _sp_detect_version(signalp_bin)
    org_flag = _sp_org_flag(org, version)
    print(f"    Running SignalP-{version} ({org_flag}) … ({faa_path.name})")

    if version >= 6:
        sp_out_dir = out_dir / (faa_path.stem + "_signalp6")
        sp_out_dir.mkdir(exist_ok=True)
        cmd = [signalp_bin, "--fastafile", str(faa_path),
               "--organism", org_flag, "--output_dir", str(sp_out_dir), "--mode", "fast"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        pred_file = sp_out_dir / "prediction_results.txt"
        raw = pred_file.read_text() if pred_file.exists() else result.stdout
    elif version == 5:
        cmd = [signalp_bin, "-fasta", str(faa_path), "-org", org_flag, "-format", "short"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        raw = result.stdout
    else:
        # SignalP 4.x
        cmd = [signalp_bin, "-t", org_flag, "-f", "short", str(faa_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        raw = result.stdout

    if result.returncode != 0:
        print(f"    [warn] SignalP failed: {(result.stderr or raw)[:250]}")
        return pd.DataFrame()

    (out_dir / (faa_path.stem + "_signalp.txt")).write_text(raw)

    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        pid = parts[0]
        if version >= 5:
            # "ID  Prediction  SP_score  ..."
            if len(parts) < 3:
                continue
            pred = parts[1]
            has_sp = not pred.startswith("OTHER")
            sp_score = float(parts[2])
        else:
            # SignalP 4.x short: name Cmax pos Ymax pos Smax pos Smean D Y/N ...
            if len(parts) < 10:
                continue
            has_sp = parts[9] == "Y"
            pred   = "SP" if has_sp else "OTHER"
            sp_score = float(parts[8])
        rows.append({"protein_id": pid, "has_sp": has_sp,
                     "sp_type": pred, "sp_score": round(sp_score, 4)})

    return pd.DataFrame(rows) if rows else \
        pd.DataFrame(columns=["protein_id", "has_sp", "sp_type", "sp_score"])


# ── TMHMM ─────────────────────────────────────────────────────────────────────

def run_tmhmm(faa_path: Path, tmhmm_bin: str, out_dir: Path) -> pd.DataFrame:
    """Run TMHMM 2.0 (or tmhmm.py) and return TM helix count per protein.

    tmhmm_bin can be:
      - "tmhmm"               → official TMHMM 2.0 binary
      - "python -m tmhmm"     → tmhmm.py Python reimplementation (pip install tmhmm.py)
    """
    print(f"    Running TMHMM … ({faa_path.name})")
    # Support "python -m tmhmm" as a single string argument
    if tmhmm_bin.strip().startswith("python"):
        cmd = tmhmm_bin.split() + [str(faa_path)]
    else:
        cmd = [tmhmm_bin, str(faa_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [warn] TMHMM failed: {result.stderr[:250]}")
        return pd.DataFrame()

    raw = result.stdout
    (out_dir / (faa_path.stem + "_tmhmm.txt")).write_text(raw)

    rows: dict = {}
    for line in raw.splitlines():
        # Comment-style long output: "# ID Number of predicted TMHs:  N"
        m = re.match(r'^#\s+(\S+)\s+Number of predicted TMHs:\s+(\d+)', line)
        if m:
            pid, n = m.group(1), int(m.group(2))
            rows.setdefault(pid, {"protein_id": pid, "n_tm": n, "topology": ""})
            rows[pid]["n_tm"] = n
            continue
        # Short tabular output: "ID\tlen=X\t...\tPredHel=N\tTopology=T"
        if "\t" in line and "PredHel=" in line:
            parts = line.split("\t")
            pid = parts[0]
            n_tm, topo = 0, ""
            for p in parts:
                if p.startswith("PredHel="):
                    n_tm = int(p.split("=")[1])
                elif p.startswith("Topology="):
                    topo = p.split("=", 1)[1]
            rows[pid] = {"protein_id": pid, "n_tm": n_tm, "topology": topo}
            continue
        # Long per-segment lines: count TMhelix segments
        if "\tTMhelix\t" in line:
            pid = line.split("\t")[0]
            rows.setdefault(pid, {"protein_id": pid, "n_tm": 0, "topology": ""})
            rows[pid]["n_tm"] += 1

    return pd.DataFrame(list(rows.values())) if rows else \
        pd.DataFrame(columns=["protein_id", "n_tm", "topology"])


# ── Localization inference ─────────────────────────────────────────────────────

def infer_localization(has_sp: bool, n_tm: int) -> str:
    """Combine signal peptide and TM helix predictions into a localization label."""
    if has_sp and n_tm == 0:
        return "Periplasmic"
    if has_sp and n_tm > 0:
        return "Membrane-anchored(SP+TM)"
    if not has_sp and n_tm > 0:
        return "Inner-membrane"
    return "Cytoplasmic"


def _is_gh_or_pl(hmm_val) -> bool:
    """Return True if protein carries any GH or PL family annotation."""
    if not isinstance(hmm_val, str) or not hmm_val.strip():
        return False
    return bool(re.search(r'\b(?:GH|PL)\d', hmm_val.replace("DIAMOND:", "")))


# ── Summary helpers ───────────────────────────────────────────────────────────

def summarize_pul(pul_name: str, ann: pd.DataFrame, svm_score: float,
                  n_proteins: int) -> dict:
    if ann.empty:
        return {"pul_name": pul_name, "svm_score": svm_score,
                "n_proteins": n_proteins, "error": "not_in_puldb"}

    n_cazy    = (ann["class"] == "CAZy").sum()
    n_unann   = (ann["class"] == "unannotated").sum()
    n_pl      = ann["is_pl"].sum()
    n_gag     = ann["is_gag_related"].sum()
    cazy_list = ";".join(sorted(set(h for h in ann["hmm"] if h)))

    return {
        "pul_name":      pul_name,
        "svm_score":     svm_score,
        "n_proteins":    n_proteins,
        "n_cazy":        int(n_cazy),
        "n_pl":          int(n_pl),
        "n_gag_related": int(n_gag),
        "n_unannotated": int(n_unann),
        "all_cazy":      cazy_list,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    cands = pd.read_csv(args.candidates)
    cands = cands.head(args.topn)
    print(f"Annotating top {len(cands)} candidates from {args.candidates}\n")

    # Load PULDB
    puldb = load_puldb(args.puldb_xlsx)

    # Build sequence index (lazy, only if needed)
    seq_index = None

    summary_rows = []
    all_annotations: dict = {}   # pul_name -> (ann_df, ann_csv_path)

    for _, row in cands.iterrows():
        pul_name  = row["pul_name"]
        svm_score = row["svm_score"]
        n_prot    = row["n_proteins"]

        print(f"\n{'='*70}")
        print(f"  {pul_name}  (SVM score={svm_score}, n_proteins={n_prot})")
        print(f"{'='*70}")

        # ── 1. Annotate from PULDB ──────────────────────────────────────────
        ann = annotate_pul(pul_name, puldb)
        if ann.empty:
            print("  [warn] PUL not found in PULDB.xlsx — skipping")
            summary_rows.append(summarize_pul(pul_name, ann, svm_score, n_prot))
            continue

        print(ann.to_string(index=False))

        # Save per-PUL annotation
        safe_name = pul_name.replace("/", "_").replace(" ", "_")
        ann_csv = out_dir / f"{safe_name}_annotation.tsv"
        ann.to_csv(ann_csv, sep="\t", index=False)
        all_annotations[pul_name] = (ann, ann_csv)

        # ── 2. Extract unannotated protein sequences ────────────────────────
        unann_ids = ann.loc[ann["class"] == "unannotated", "protein_id"].tolist()
        print(f"\n  Unannotated proteins: {len(unann_ids)}/{len(ann)}")

        if unann_ids:
            if seq_index is None:
                seq_index = build_seq_index(args.proteins_faa)

            unann_faa = out_dir / f"{safe_name}_unannotated.faa"
            found = extract_sequences(unann_ids, seq_index, unann_faa)
            print(f"  Sequences extracted: {len(found)}/{len(unann_ids)} → {unann_faa.name}")

            # ── 3. DIAMOND vs CAZyDB ──────────────────────────────────────
            if args.diamond_db and unann_faa.exists():
                dmnd_hits = run_diamond(unann_faa, args.diamond_db,
                                        args.diamond_bin, args.diamond_evalue,
                                        args.diamond_max_hits, out_dir)
                if not dmnd_hits.empty:
                    print(f"\n  DIAMOND hits ({len(dmnd_hits)}):")
                    print(dmnd_hits.to_string(index=False))
                    dmnd_hits.to_csv(
                        out_dir / f"{safe_name}_diamond.tsv",
                        sep="\t", index=False)
                    # Fold DIAMOND hits back into annotation table
                    for _, drow in dmnd_hits.iterrows():
                        mask = ann["protein_id"] == drow["protein_id"]
                        fam  = drow["cazy_family"]
                        ann.loc[mask, "hmm"]           = f"DIAMOND:{fam}"
                        ann.loc[mask, "class"]         = "DIAMOND"
                        ann.loc[mask, "is_pl"]         = fam.startswith("PL")
                        ann.loc[mask, "is_gag_related"]= fam in GAG_CAZY
                else:
                    print("  DIAMOND: no hits above threshold")

            # ── 4. hmmscan (Pfam) ─────────────────────────────────────────
            if args.hmm_db and unann_faa.exists():
                hmm_hits = run_hmmscan(unann_faa, args.hmm_db,
                                       args.hmmscan_bin, out_dir)
                if not hmm_hits.empty:
                    print(f"\n  Pfam hits ({len(hmm_hits)}):")
                    print(hmm_hits.to_string(index=False))
                    hmm_hits.to_csv(
                        out_dir / f"{safe_name}_pfam.tsv",
                        sep="\t", index=False)
                else:
                    print("  Pfam hmmscan: no hits")

            # ── 5. NCBI BLAST ─────────────────────────────────────────────
            if args.blast and unann_faa.exists():
                print(f"\n  Running NCBI BLAST on {len(found)} proteins …")
                blast_hits = run_ncbi_blast(unann_faa, args.blast_db,
                                            args.blast_max_hits)
                if not blast_hits.empty:
                    blast_hits.to_csv(
                        out_dir / f"{safe_name}_blast.tsv",
                        sep="\t", index=False)
                    print(blast_hits.to_string(index=False))
                else:
                    print("  BLAST: no hits above threshold")

        # ── 6. Highlight likely depolymerases ──────────────────────────────
        depoly = ann[ann["is_pl"] | ann["is_gag_related"]]
        if not depoly.empty:
            print(f"\n  Likely depolymerases / GAG-related:")
            print(depoly[["protein_id","protein_name","hmm","class"]].to_string(index=False))

        summary_rows.append(summarize_pul(pul_name, ann, svm_score, n_prot))

    # ── 7. Overall summary ────────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "candidate_summary.tsv"
    summary.to_csv(summary_csv, sep="\t", index=False)

    print(f"\n{'='*70}")
    print("CANDIDATE SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved → {summary_csv}")
    print(f"Per-PUL files → {out_dir}/")

    # ── 8. Localization prediction on all GH + PL proteins ───────────────────
    if not (args.signalp_bin or args.tmhmm_bin):
        return

    print(f"\n{'='*70}")
    print("LOCALIZATION PREDICTION  (GH + PL proteins)")
    print(f"{'='*70}")

    # Collect unique GH/PL protein IDs across all annotated PULs
    ghpl_ids: list = []
    ghpl_pul: dict = {}   # protein_id -> pul_name (first seen)
    for pul_name, (ann, _) in all_annotations.items():
        for _, r in ann.iterrows():
            if _is_gh_or_pl(r.get("hmm", "")):
                pid = r["protein_id"]
                if pid not in ghpl_pul:
                    ghpl_ids.append(pid)
                    ghpl_pul[pid] = pul_name

    print(f"  {len(ghpl_ids)} GH/PL proteins found across {len(all_annotations)} PULs")
    if not ghpl_ids:
        print("  Nothing to do.")
        return

    if seq_index is None:
        seq_index = build_seq_index(args.proteins_faa)

    ghpl_faa = out_dir / "all_ghpl_proteins.faa"
    extract_sequences(ghpl_ids, seq_index, ghpl_faa)

    # ── SignalP ───────────────────────────────────────────────────────────────
    sp_df = pd.DataFrame(columns=["protein_id", "has_sp", "sp_type", "sp_score"])
    if args.signalp_bin and ghpl_faa.exists():
        sp_df = run_signalp(ghpl_faa, args.signalp_bin, args.signalp_org, out_dir)
        if sp_df.empty:
            print("  [warn] SignalP produced no results")

    # ── TMHMM ─────────────────────────────────────────────────────────────────
    tm_df = pd.DataFrame(columns=["protein_id", "n_tm", "topology"])
    if args.tmhmm_bin and ghpl_faa.exists():
        tm_df = run_tmhmm(ghpl_faa, args.tmhmm_bin, out_dir)
        if tm_df.empty:
            print("  [warn] TMHMM produced no results")

    # ── Merge & infer ─────────────────────────────────────────────────────────
    loc = pd.DataFrame({"protein_id": ghpl_ids})
    loc["pul_name"] = loc["protein_id"].map(ghpl_pul)

    if not sp_df.empty:
        loc = loc.merge(sp_df[["protein_id", "has_sp", "sp_type", "sp_score"]],
                        on="protein_id", how="left")
        loc["has_sp"]   = loc["has_sp"].fillna(False)
        loc["sp_type"]  = loc["sp_type"].fillna("—")
        loc["sp_score"] = loc["sp_score"].fillna(0.0)
    else:
        loc[["has_sp", "sp_type", "sp_score"]] = False, "—", 0.0

    if not tm_df.empty:
        loc = loc.merge(tm_df[["protein_id", "n_tm", "topology"]],
                        on="protein_id", how="left")
        loc["n_tm"]     = loc["n_tm"].fillna(0).astype(int)
        loc["topology"] = loc["topology"].fillna("—")
    else:
        loc[["n_tm", "topology"]] = 0, "—"

    loc["localization"] = loc.apply(
        lambda r: infer_localization(bool(r["has_sp"]), int(r["n_tm"])), axis=1)

    # Merge hmm info for context
    hmm_map: dict = {}
    for pul_name, (ann, _) in all_annotations.items():
        for _, r in ann.iterrows():
            if _is_gh_or_pl(r.get("hmm", "")):
                hmm_map[r["protein_id"]] = r["hmm"]
    loc.insert(2, "hmm", loc["protein_id"].map(hmm_map).fillna("—"))

    loc_csv = out_dir / "ghpl_localization.tsv"
    loc.to_csv(loc_csv, sep="\t", index=False)

    print(f"\n  Overall localization counts:")
    print(loc["localization"].value_counts().to_string())

    print(f"\n  Per-PUL GH/PL localization:")
    for pul_name in all_annotations:
        sub = loc[loc["pul_name"] == pul_name]
        if sub.empty:
            continue
        print(f"\n  ── {pul_name}")
        cols = ["protein_id", "hmm", "localization"]
        if not sp_df.empty:
            cols += ["sp_score"]
        if not tm_df.empty:
            cols += ["n_tm"]
        print(sub[cols].to_string(index=False))

    print(f"\n  Saved → {loc_csv}")


if __name__ == "__main__":
    main()
