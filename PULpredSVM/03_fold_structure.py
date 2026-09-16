"""
03_fold_structure.py

Structural annotation of hypothetical proteins in top GAG/alginate candidates.

Pipeline:
  1. Extract hypothetical proteins from top-50 candidate CGC FASTAs
  2. Predict 3D structures with ESMFold (GPU required)
  3. Search predicted structures against PDB + AlphaFold/Swiss-Prot with FoldSeek
  4. Filter hits by polysaccharide-degradation-related keywords
  5. Output annotated FASTA (hypothetical proteins with structural evidence)
     and full FoldSeek hit table

One-time database setup (run once, ~10 GB total):
  DB=data/foldseek_db
  mkdir -p $DB
  conda run -p /work3/zhayu/envs/pulpred foldseek databases PDB $DB/pdb $DB/tmp
  conda run -p /work3/zhayu/envs/pulpred foldseek databases Alphafold/Swiss-Prot $DB/swissprot $DB/tmp

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python PULpredSVM/03_fold_structure.py --task gag
      python PULpredSVM/03_fold_structure.py --task alginate
      python PULpredSVM/03_fold_structure.py --task gag --top_n 100
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

DBCAN_DIR  = Path("data/dbCAN_seq")
FS_DB_DIR  = Path("data/foldseek_db")
FS_DBS     = ["pdb", "swissprot"]   # databases to search
THREADS    = 8
TOP_HITS   = 5   # FoldSeek hits to report per protein

# Keywords for polysaccharide-degradation-related structural homologs
KEEP_KEYWORDS = {
    "lyase", "hydrolase", "glycosidase", "glycoside", "glycosyl",
    "heparinase", "chondroitinase", "sulfatase", "sulfohydrolase",
    "polysaccharide", "carbohydrate", "alginate", "pectin",
    "mannuronate", "guluronate", "glucuronidase", "galactosidase",
    "fucosidase", "sialidase", "amylase", "cellulase", "xylanase",
    "DUF",
}

HYPO_TERMS = {
    "hypothetical protein", "uncharacterized protein",
    "predicted protein", "unknown function", "putative protein",
}


def is_hypothetical(annotation: str) -> bool:
    a = annotation.lower()
    return any(t in a for t in HYPO_TERMS) or annotation.strip() in {"", "null", "-"}


def read_fasta(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (protein_id, annotation, sequence)."""
    entries, cur_pid, cur_ann, cur_seq = [], None, "", []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur_pid:
                entries.append((cur_pid, cur_ann, "".join(cur_seq)))
            parts = line[1:].split("|")
            cur_pid = parts[2].split()[0] if len(parts) >= 3 else line[1:].split()[0]
            cur_ann = " ".join(line[1:].split()[1:]) if len(line[1:].split()) > 1 else ""
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_pid:
        entries.append((cur_pid, cur_ann, "".join(cur_seq)))
    return entries


def cgc_id_to_fasta(cgc_id: str, env: str) -> Path:
    stem   = cgc_id.replace("|", "#")
    genome = stem.split("#")[0].rsplit("_", 1)[0]
    return DBCAN_DIR / env / genome / "dbcan_out" / "CGC_fasta" / f"{stem}.fasta"


def predict_structures_esmfold(proteins: list[tuple[str, str]],
                                struct_dir: Path) -> list[Path]:
    """
    Predict PDB structures for (pid, seq) pairs with ESMFold.
    Returns list of output PDB file paths.
    """
    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    print(f"  Loading ESMFold via transformers…")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model     = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )
    model = model.eval().cuda()
    model.esm = model.esm.half()   # save GPU memory

    out_paths = []
    for pid, seq in proteins:
        pdb_path = struct_dir / f"{pid.replace('|','_').replace('/','_')}.pdb"
        if pdb_path.exists():
            print(f"    {pid}: cached")
            out_paths.append(pdb_path)
            continue

        succeeded = False
        for max_len in [1024, 512]:
            seq_in = seq[:max_len]
            try:
                tokenized = tokenizer([seq_in], return_tensors="pt",
                                      add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    output = model(**tokenized)
                # Convert output to PDB string
                positions = atom14_to_atom37(output["positions"][-1], output)
                pdb_str   = to_pdb(OFProtein(
                    aatype         = output["aatype"][0].cpu().numpy(),
                    atom_positions = positions[0].cpu().numpy(),
                    atom_mask      = output["atom37_atom_exists"][0].cpu().numpy(),
                    residue_index  = output["residue_index"][0].cpu().numpy(),
                    b_factors      = output["plddt"][0].cpu().numpy(),
                    chain_index    = output.get("chain_index", [None])[0],
                ))
                pdb_path.write_text(pdb_str)
                plddt = output["plddt"][0].mean().item()
                print(f"    {pid}: {len(seq_in)} AA  pLDDT={plddt:.1f} → {pdb_path.name}")
                out_paths.append(pdb_path)
                succeeded = True
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if max_len == 512:
                    print(f"    {pid}: SKIPPED (OOM at 512 AA — sequence too long)")
                else:
                    print(f"    {pid}: OOM at {max_len} AA, retrying at 512…")
        if not succeeded:
            pass  # protein skipped, not added to out_paths
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return out_paths


def find_foldseek() -> str:
    fs = shutil.which("foldseek")
    if fs:
        return fs
    candidate = Path("/work3/zhayu/envs/pulpred/bin/foldseek")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("foldseek not found")


def run_foldseek(struct_dir: Path, db_path: Path,
                 out_tsv: Path, threads: int) -> None:
    fs = find_foldseek()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([
            fs, "easy-search",
            str(struct_dir), str(db_path), str(out_tsv), tmp,
            "--format-output",
            "query,target,pident,alnlen,qlen,tlen,prob,evalue,bits,theader",
            "--exhaustive-search", "0",
            "--num-iterations", "2",
            "-e", "0.001",
            "--threads", str(threads),
            "-v", "1",
        ], check=True)


def hits_match_keywords(theader: str) -> bool:
    h = theader.lower()
    return any(kw.lower() in h for kw in KEEP_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",    required=True, choices=["gag", "alginate"])
    ap.add_argument("--top_n",   type=int, default=50,
                    help="Use top N candidates from CSV (default 50)")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    res_dir    = Path(f"PULpredSVM/results/{args.task}_minprot5")
    cands_csv  = res_dir / f"novel_{args.task}_candidates.csv"
    struct_dir = res_dir / "structures"
    struct_dir.mkdir(exist_ok=True)

    if not cands_csv.exists():
        raise FileNotFoundError(f"Run 01_score_svm.py first: {cands_csv}")

    import pandas as pd
    cands = pd.read_csv(cands_csv).head(args.top_n)
    print(f"[1/4] {len(cands)} candidates from {cands_csv}\n")

    # ── Step 1: Collect hypothetical proteins ─────────────────────────────────
    print("[2/4] Extracting hypothetical proteins…")
    hypo_proteins: list[tuple[str, str, str, str]] = []  # (cgc_id, pid, ann, seq)

    for _, row in cands.iterrows():
        cgc_id = row["cgc_id"]
        env    = row["environment"]
        fa     = cgc_id_to_fasta(cgc_id, env)
        if not fa.exists():
            continue
        for pid, ann, seq in read_fasta(fa):
            if is_hypothetical(ann) and len(seq) >= 50:
                hypo_proteins.append((cgc_id, pid, ann, seq))

    print(f"  {len(hypo_proteins)} hypothetical proteins (≥50 AA) across {len(cands)} CGCs")

    if not hypo_proteins:
        print("  Nothing to annotate.")
        return

    # ── Step 2: ESMFold structure prediction ─────────────────────────────────
    print("\n[3/4] Predicting structures with ESMFold…")
    proteins_for_esmfold = [
        (pid, seq) for _, pid, _, seq in hypo_proteins
        if not (struct_dir / f"{pid.replace('|','_').replace('/','_')}.pdb").exists()
    ]
    if proteins_for_esmfold:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{len(proteins_for_esmfold)} structures not yet cached — "
                "ESMFold requires a CUDA GPU. Submit via run_foldseek_annotate.sh"
            )
        predict_structures_esmfold(proteins_for_esmfold, struct_dir)
    else:
        print(f"  All {len(hypo_proteins)} structures already cached — skipping ESMFold.")

    # ── Step 3: FoldSeek search ───────────────────────────────────────────────
    print("\n[4/4] FoldSeek search…")
    all_hits: dict[str, list[dict]] = {}  # pid → list of hit dicts

    for db_name in FS_DBS:
        db_path = FS_DB_DIR / db_name
        if not db_path.exists():
            print(f"  WARNING: {db_path} not found — skipping. Run database setup first.")
            continue

        out_tsv = res_dir / f"foldseek_{db_name}.tsv"
        if not out_tsv.exists():
            print(f"  Searching {db_name}…")
            run_foldseek(struct_dir, db_path, out_tsv, args.threads)
        else:
            print(f"  {db_name}: cached")

        with open(out_tsv) as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) < 10:
                    continue
                query, target, pident, alnlen, qlen, tlen, prob, evalue, bits, theader = parts[:10]
                # Skip hits where query covers less than 50% of the template length —
                # these indicate the query is missing large structural domains present
                # in the reference (e.g. a short domain fragment matching a full lyase).
                try:
                    if int(qlen) / int(tlen) < 0.5:
                        continue
                except (ValueError, ZeroDivisionError):
                    pass
                pid = Path(query).stem
                all_hits.setdefault(pid, []).append({
                    "db": db_name, "target": target,
                    "pident": pident, "prob": prob, "evalue": evalue,
                    "qlen": qlen, "tlen": tlen,
                    "theader": theader,
                })

    # ── Output: annotated FASTA + hit table ──────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────────────")

    hit_rows = []
    kept_seqs: list[tuple[str, str, str, list[dict]]] = []  # (cgc_id, pid, seq, hits)

    for cgc_id, pid, ann, seq in hypo_proteins:
        tsv_key = pid.replace("|", "_")  # matches FoldSeek query stem (structure filename)
        all_pid_hits = all_hits.get(tsv_key, [])
        # Sort all hits by FoldSeek probability descending
        all_pid_hits.sort(key=lambda h: float(h["prob"]), reverse=True)

        relevant = [h for h in all_pid_hits if hits_match_keywords(h["theader"])]
        top_hits  = (relevant if relevant else all_pid_hits)[:TOP_HITS]

        # Keep protein if any hit is polysaccharide-related
        if relevant:
            kept_seqs.append((cgc_id, pid, seq, top_hits))

        for h in top_hits:
            hit_rows.append({
                "cgc_id": cgc_id, "protein_id": pid,
                "annotation": ann,
                "db": h["db"], "target": h["target"],
                "pident": h["pident"], "prob": h["prob"],
                "evalue": h["evalue"],
                "hit_description": h["theader"],
                "polysaccharide_related": hits_match_keywords(h["theader"]),
            })

    # Save full hit table
    hits_tsv = res_dir / "hypothetical_foldseek_hits.tsv"
    if hit_rows:
        pd.DataFrame(hit_rows).to_csv(hits_tsv, sep="\t", index=False)
        print(f"Full hit table → {hits_tsv}")

    # Save filtered FASTA (only polysaccharide-related)
    out_fa = res_dir / "hypothetical_foldseek_filtered.fasta"
    with open(out_fa, "w") as f:
        for cgc_id, pid, seq, hits in kept_seqs:
            top = hits[0]
            f.write(f">{pid} [{cgc_id}] top_hit={top['target']} "
                    f"prob={top['prob']} | {top['theader'][:80]}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    print(f"\nHypothetical proteins with polysaccharide-related structural homologs:")
    print(f"  {len(kept_seqs)} / {len(hypo_proteins)} retained → {out_fa}")

    if kept_seqs:
        print(f"\n{'CGC':<35} {'Protein':<25} {'Top hit (prob)'}")
        print("-" * 90)
        for cgc_id, pid, seq, hits in kept_seqs:
            top = hits[0]
            desc = top["theader"][:45]
            print(f"  {cgc_id:<35} {pid:<25} {top['prob']:>5}  {desc}")


if __name__ == "__main__":
    main()
