import contextlib
import json
import os
import sys
from functools import partial, reduce

prj_path = os.path.dirname(os.path.dirname(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import pathlib
import random
from typing import List, Optional, Union

import experiments.simple_search as S


def generating(files: List[pathlib.Path],
               dbbase: pathlib.Path,
               base: pathlib.Path,
               uniref_db: pathlib.Path,
               mmseqs: pathlib.Path = pathlib.Path("mmseqs"),
               s: float = 5.7,
               e: float = 1e-4,
               db_load_mode: int = 2,
               threads: int = 32):
    """
    msa generating with mmseqs
    """
    tmp_query_fapath = base.joinpath("tmp_queries.fa")

    with contextlib.ExitStack() as stack:
        readers = [stack.enter_context(open(f, "r"))
                   for f in files]
        with open(tmp_query_fapath, "w") as h:
            for reader in readers:
                h.write(reader.read())

    query_file = S.mmseqs_preprocess(tmp_query_fapath, base, mmseqs)
    S.mmseqs_simple_search(dbbase, base, uniref_db,
                           mmseqs, s, e, db_load_mode, threads)
    S.mmseqs_end_process(query_file, base, mmseqs)

def batch_search(ori: Union[pathlib.Path, List[str]],
                 dbbase: pathlib.Path,
                 base: pathlib.Path,
                 uniref_db: pathlib.Path,
                 seq_dir: Optional[pathlib.Path] = None,
                 seq_suffix: str = "fa",
                 batch_size: int = 200,
                 mmseqs: pathlib.Path = pathlib.Path("mmseqs"),
                 s: float = 5.7,
                 e: float = 1e-4,
                 db_load_mode: int = 2,
                 threads: int = 32
                 ):
    """
    ori: sequence file (fasta format) directory or sequence name list, the latter need a seq_dir and a seq_suffix
    dbbase: the database directory
    base: the target directory
    uniref_db: the database contenct path
    seq_dir: if ori is a name list, this parameter is necessary
    """

    existed_fnames = (p.removesuffix(".a3m") for p in os.listdir(base)
                      if p.endswith(".a3m"))
    efname_set = set(existed_fnames)

    if isinstance(ori, pathlib.Path):
        filenames = [p.removesuffix(f".{seq_suffix}") 
                     for p in os.listdir(ori)]
        seq_dir = ori
    else:
        assert seq_dir is not None, "when ori is a name list, the seq_dir is necessary"
        filenames = ori
    
    fname_set = efname_set.symmetric_difference(filenames)
    files = [seq_dir.joinpath(f"{n}.{seq_suffix}") for n in fname_set]
    
    fs_size = len(files)
    idx_lst = range(fs_size)
    random.shuffle(list(idx_lst))
    batch_files = [[files[j] for j in idx_lst[i:i+batch_size]] 
                   for i in range(0, fs_size, batch_size)]

    partial_generation = partial(generating, dbbase=dbbase, base=base,
                                 uniref_db=uniref_db, mmseqs=mmseqs,
                                 s=s,e=e,db_load_mode=db_load_mode, threads=threads)
    reduce(lambda _, ps: partial_generation(ps), batch_files, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ori",
        type=pathlib.Path,
        help="fasta files dir or name list pickle file.",
    )
    parser.add_argument(
        "dbbase",
        type=pathlib.Path,
        help="The path to the database and indices you downloaded and created with setup_databases.sh",
    )
    parser.add_argument(
        "base", type=pathlib.Path, help="Directory for the results (and intermediate files)"
    )
    parser.add_argument(
        "-s",
        type=float,
        default=5.7,
        help="mmseqs sensitivity. Lowering this will result in a much faster search but possibly sparser msas",
    )
    parser.add_argument(
        "-e",
        type=float,
        default=1e-4,
        help="mmseqs evalue. Lowering this will result in a smaller number of sequence in multiple sequence results"
    )
    # dbs are uniref, templates and environmental
    # We normally don't use templates
    parser.add_argument(
        "--db1", type=pathlib.Path, default=pathlib.Path("uniref30_2103_db"), help="UniRef database"
    )

    parser.add_argument(
        "--mmseqs",
        type=pathlib.Path,
        default=pathlib.Path("mmseqs"),
        help="Location of the mmseqs binary",
    )

    parser.add_argument("--db-load-mode", type=int, default=2)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seq-dir", type=pathlib.Path, 
                        help="when ori is a json file containing names, this option is required")
    parser.add_argument("--seq-suffix", type=str, default="fa",
                        help="the suffix of a sequence filename")
    args = parser.parse_args()

    if os.path.isfile(args.ori):
        with open(args.ori, "r") as h:
            ori = json.load(h)
    else:
        ori = args.ori

    batch_search(ori, args.dbbase, args.base, 
                 args.db1, args.seq_dir, args.seq_suffix,
                 args.batch_size, args.mmseqs, args.s, args.e,
                 args.db_load_mode, args.threads)

if __name__ == "__main__":
    main()