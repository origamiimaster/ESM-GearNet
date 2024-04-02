import csv
import glob
import os
import sys
import math
import pprint
import random

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils import data as torch_data

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.datasets import AlphaFoldDB
from torchdrug.utils import comm
from torchdrug.core import Registry as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, task, siamdiff
from gearnet import model, cdconv, gvp, dataset, task, protbert


@R.register("datasets.MyDataSet")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class MyDataSet(data.ProteinDataset):
    """
    A set of proteins with their 3D structures and EC numbers, which describes their
    catalysis of biochemical reactions.

    Statistics (test_cutoff=0.95):
        - #Train: 15,011
        - #Valid: 1,664
        - #Test: 1,840

    Parameters:
        path (str): the path to store the dataset
        test_cutoff (float, optional): the test cutoff used to split the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://zenodo.org/record/6622158/files/EnzymeCommission.zip"
    md5 = "33f799065f8ad75f87b709a87293bc65"
    processed_file = "enzyme_commission.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    def __init__(self, path, test_cutoff=0.95, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if test_cutoff not in self.test_cutoffs:
            raise ValueError(
                "Unknown test cutoff `%.2f` for EnzymeCommission dataset" % test_cutoff)
        self.test_cutoff = test_cutoff

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(utils.extract(zip_file), "EnzymeCommission")
        pkl_file = os.path.join(path, self.processed_file)

        csv_file = os.path.join(path, "nrPDB-EC_test.csv")
        pdb_ids = []
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=",")
            idx = self.test_cutoffs.index(test_cutoff) + 1
            _ = next(reader)
            for line in reader:
                if line[idx] == "0":
                    pdb_ids.append(line[0])

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in ["train", "valid", "test"]:
                split_path = utils.extract(
                    os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(
                    glob.glob(os.path.join(split_path, split, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        if len(pdb_ids) > 0:
            self.filter_pdb(pdb_ids)

        tsv_file = os.path.join(path, "nrPDB-EC_annot.tsv")
        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in
                   self.pdb_files]
        self.load_annotation(tsv_file, pdb_ids)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file
                  in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"),
                            splits.count("test")]

    def filter_pdb(self, pdb_ids):
        pdb_ids = set(pdb_ids)
        sequences = []
        pdb_files = []
        data = []
        for sequence, pdb_file, protein in zip(self.sequences,
                                               self.pdb_files, self.data):
            if os.path.basename(pdb_file).split("_")[0] in pdb_ids:
                continue
            sequences.append(sequence)
            pdb_files.append(pdb_file)
            data.append(protein)
        self.sequences = sequences
        self.pdb_files = pdb_files
        self.data = data

    def load_annotation(self, tsv_file, pdb_ids):
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            _ = next(reader)
            tasks = next(reader)
            task2id = {task: i for i, task in enumerate(tasks)}
            _ = next(reader)
            pos_targets = {}
            for pdb_id, pos_target in reader:
                pos_target = [task2id[t] for t in pos_target.split(",")]
                pos_target = torch.tensor(pos_target)
                pos_targets[pdb_id] = pos_target

        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self,
                                      range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index],
                                            self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        item["targets"] = utils.sparse_coo_tensor(indices, values,
                                                  (len(self.tasks),)).to_dense()
        return item


@R.register("datasets.MiniAlphaFoldDB")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class MiniAlphaFoldDB(AlphaFoldDB):
    """
    3D protein structures predicted by AlphaFold.
    This dataset covers proteomes of 48 organisms, as well as the majority of Swiss-Prot.

    Statistics:
        See https://alphafold.ebi.ac.uk/download

    Parameters:
        path (str): path to store the dataset
        species_id (int, optional): the id of species to be loaded. The species are numbered
            by the order appeared on https://alphafold.ebi.ac.uk/download (0-20 for model
            organism proteomes, 21 for Swiss-Prot)
        split_id (int, optional): the id of split to be loaded. To avoid large memory consumption
            for one dataset, we have cut each species into several splits, each of which contains
            at most 22000 proteins.
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(self, path, species_id=0, split_id=0, verbose=1, **kwargs):
        super().__init__(path, species_id=0, split_id=0, verbose=1, **kwargs)

        print(self.__len__())





@R.register("transforms.CustomNoiseTransform")
class CustomNoiseTransform(core.Configurable):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, item):
        # Graph item needs specific perturbations
        graph = item["graph"].clone()
        b_factor = graph.b_factor.unsqueeze(0) / 100
        perturb_noise = torch.randn_like(graph.node_position)
        perturb_noise = b_factor * perturb_noise
        graph.node_position = graph.node_position + perturb_noise * self.sigma
        item["graph2"] = graph
        return item


if __name__ == "__main__":
    # args, vars = util.parse_args()
    cfg = util.load_config("config/custom/custom.yaml", context={})
    working_dir = util.create_working_directory(cfg)

    # Set seed:
    seed = 1024 # args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = core.Configurable.load_config_dict(cfg.dataset)

    solver, scheduler = util.build_downstream_solver(cfg, dataset)
