"""
05_blast_consensus.py

NCBI remote BLAST for all hypothetical proteins in the high-confidence
consensus candidates (found by both ESM2mean and CLS models, min_proteins=5).

Targets:
  PULpredSVM/results/gag_highconf_minprot5.fasta
  PULpredSVM/results/alginate_highconf_minprot5.fasta

Output:
  PULpredSVM/results/gag_highconf_minprot5_blast.tsv
  PULpredSVM/results/alginate_highconf_minprot5_blast.tsv

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python PULpredSVM/05_blast_consensus.py --task gag
      python PULpredSVM/05_blast_consensus.py --task alginate
      python PULpredSVM/05_blast_consensus.py --task both
"""

import argparse
import time
from pathlib import Path

TASKS = {
    "gag":      Path("PULpredSVM/results/gag_highconf_minprot5.fasta"),
    "alginate": Path("PULpredSVM/results/alginate_highconf_minprot5.fasta"),
}

BATCH_SIZE  = 5
SLEEP_SEC   = 15
EVALUE      = "1e-5"
MAX_HITS    = 5


def parse_fasta(fa_path):
    """Returns list of (header, seq) for all entries."""
    entries, cur_hdr, cur_seq = [], None, []
    for line in fa_path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur_hdr is not None:
                entries.append((cur_hdr, "".join(cur_seq)))
            cur_hdr = line[1:].rstrip()
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_hdr is not None:
        entries.append((cur_hdr, "".join(cur_seq)))
    return entries


def is_hypothetical(header):
    h = header.lower()
    return ("hypothetical protein" in h or
            "uncharacterized protein" in h or
            "predicted protein" in h or
            "unknown function" in h or
            (header.rstrip().endswith("|") and "hypothetical" in h))


def blast_batch(seqs, evalue, max_hits):
    """seqs: list of (pid, seq). Returns list of result dicts."""
    from Bio.Blast import NCBIWWW, NCBIXML

    fasta_str = "\n".join(f">{pid}\n{seq}" for pid, seq in seqs)
    result_handle = NCBIWWW.qblast(
        "blastp", "nr", fasta_str,
        expect=evalue, hitlist_size=max_hits,
        format_type="XML"
    )
    records = list(NCBIXML.parse(result_handle))
    rows = []
    for rec in records:
        qid = rec.query.split()[0]
        if not rec.alignments:
            rows.append({"query": qid, "hit_id": "—", "hit_desc": "No significant hits",
                         "identity": "—", "evalue": "—", "score": "—"})
            continue
        for aln in rec.alignments[:max_hits]:
            hsp = aln.hsps[0]
            rows.append({
                "query":    qid,
                "hit_id":   aln.accession,
                "hit_desc": aln.hit_def[:120],
                "identity": f"{100*hsp.identities/hsp.align_length:.1f}%",
                "evalue":   f"{hsp.expect:.2e}",
                "score":    hsp.score,
            })
    return rows


def run_task(task, fa_path, out_tsv):
    print(f"\n{'='*60}")
    print(f"  {task.upper()}  →  {fa_path.name}")
    print(f"{'='*60}")

    entries    = parse_fasta(fa_path)
    hypo       = [(hdr, seq) for hdr, seq in entries if is_hypothetical(hdr)]
    print(f"  Total proteins  : {len(entries)}")
    print(f"  Hypothetical    : {len(hypo)}\n")

    if not hypo:
        print("  Nothing to BLAST.")
        return

    # Load cache
    cached = {}
    if out_tsv.exists():
        import csv
        with open(out_tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                cached[row["query"]] = True
        print(f"  Cache: {len(cached)} queries already done\n")

    # Build work list (use first field of header as query ID)
    todo = [(hdr.split()[0], seq) for hdr, seq in hypo
            if hdr.split()[0] not in cached]
    print(f"  Queries to BLAST: {len(todo)}\n")

    if not todo:
        print("  All done (cached).")
        _print_summary(out_tsv)
        return

    # Write header if new file
    write_header = not out_tsv.exists()
    with open(out_tsv, "a") as f:
        if write_header:
            f.write("query\thit_id\thit_desc\tidentity\tevalue\tscore\n")

        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            pids  = [p for p, _ in batch]
            print(f"  Batch {i//BATCH_SIZE + 1}: {pids}")
            try:
                rows = blast_batch(batch, EVALUE, MAX_HITS)
                for row in rows:
                    f.write(
                        f"{row['query']}\t{row['hit_id']}\t{row['hit_desc']}\t"
                        f"{row['identity']}\t{row['evalue']}\t{row['score']}\n"
                    )
                f.flush()
                for row in rows:
                    if row["hit_id"] != "—":
                        print(f"    {row['query'][:30]:30s}  "
                              f"{row['evalue']:>10}  {row['hit_desc'][:60]}")
            except Exception as e:
                print(f"    ERROR: {e}")
            if i + BATCH_SIZE < len(todo):
                print(f"    Sleeping {SLEEP_SEC}s…")
                time.sleep(SLEEP_SEC)

    print(f"\n  Saved → {out_tsv}")
    _print_summary(out_tsv)


def _print_summary(tsv_path):
    import csv
    from collections import Counter
    if not tsv_path.exists():
        return
    rows = list(csv.DictReader(open(tsv_path), delimiter="\t"))
    hits = [r for r in rows if r.get("hit_id", "—") != "—"]
    print(f"\n  === Summary ===")
    print(f"  Total BLAST rows  : {len(rows)}")
    print(f"  With hits         : {len(hits)}")
    # Show unique top hits
    descs = Counter(r["hit_desc"][:60] for r in hits)
    print(f"  Top annotations:")
    for desc, n in descs.most_common(10):
        print(f"    {n:>3}x  {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["gag","alginate","both"])
    args = ap.parse_args()

    tasks = ["gag","alginate"] if args.task == "both" else [args.task]
    for task in tasks:
        fa_path = TASKS[task]
        out_tsv = fa_path.with_suffix("").parent / f"{fa_path.stem}_blast.tsv"
        if not fa_path.exists():
            print(f"  FASTA not found: {fa_path}"); continue
        run_task(task, fa_path, out_tsv)


if __name__ == "__main__":
    main()
