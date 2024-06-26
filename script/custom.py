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
class MiniAlphaFoldDB(data.ProteinDataset):
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

    urls = [
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006548_3702_ARATH_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001940_6239_CAEEL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000559_237561_CANAL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000437_7955_DANRE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002195_44689_DICDI_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000803_7227_DROME_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000625_83333_ECOLI_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008827_3847_SOYBN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000005640_9606_HUMAN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008153_5671_LEIIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000805_243232_METJA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000589_10090_MOUSE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001584_83332_MYCTU_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000059680_39947_ORYSJ_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001450_36329_PLAF7_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002494_10116_RAT_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002311_559292_YEAST_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002485_284812_SCHPO_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008816_93061_STAA8_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002296_353153_TRYCC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000007305_4577_MAIZE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/swissprot_pdb_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001631_447093_AJECG_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006672_6279_BRUMA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000799_192222_CAMJE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000094526_86049_9EURO1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000274756_318479_DRAME_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000325664_1352_ENTFC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000053029_1442368_9EURO2_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000579_71421_HAEIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000429_85962_HELPY_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000007841_1125630_KLEPH_v2.tar",
        # "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008153_5671_LEIIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000078237_100816_9PEZI1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000806_272631_MYCLE_v2.tar",
        # "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001584_83332_MYCTU_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000020681_1299332_MYCUL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000535_242231_NEIG1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006304_1133849_9NOCA1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000024404_6282_ONCVO_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002059_502779_PARBA_v2.tar",
        # "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001450_36329_PLAF7_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002438_208964_PSEAE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001014_99287_SALTY_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008854_6183_SCHMA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002716_300267_SHIDS_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000018087_1391915_SPOS1_v2.tar",
        # "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008816_93061_STAA8_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000586_171101_STRR6_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000035681_6248_STRER_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000030665_36087_TRITR_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008524_185431_TRYB2_v2.tar",
        # "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002296_353153_TRYCC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000270924_6293_WUCBA_v2.tar"
    ]
    md5s = [
        "4cd5f596ebfc3d45d9f6b647dc5684af", "b89bee5507f78f971417cc8fd75b40f7", "a6459a1f1a0a22fbf25f1c05c2889ae3",
        "24dfba8ab93dbf3f51e7db6b912dd6b4", "6b81b3086ed9e57e04a54f148ecf974c", "a50f4fd9f581c89e79e1b2857e54b786",
        "fdd16245769bf1f7d91a0e285ac00e52", "66b9750c511182bc5f8ee71fe2ab2a17", "5dadeb5aac704025cac33f7557794858",
        "99b22e0f050d845782d914becbfe4d2f", "da938dfae4fabf6e144f4b5ede5885ec", "2003c09d437cfb4093552c588a33e06d",
        "fba59f386cfa33af3f70ae664b7feac0", "d7a1a6c02213754ee1a1ffb3b41ad4ba", "8a0e8deadffec2aba3b7edd6534b7481",
        "1854d0bbcf819de1de7b0cfdb6d32b2e", "d9720e3809db6916405db096b520c236", "6b918e9e4d645b12a80468bcea805f1f",
        "ed0eefe927eb8c3b81cf87eaabbb8d6e", "051369e0dc8fed4798c8b2c68e6cbe2e", "b05ff57164167851651c625dca66ed28",
        "68e7a6e57bd43cb52e344b3190073387", "75d027ac7833f284fda65ea620353e8a", "7d85bb2ee4130096a6d905ab8d726bcc",
        "63498210c88e8bfb1a7346c4ddf73bb1", "5bf2211304ef91d60bb3838ec12d89cd", "4981758eb8980e9df970ac6113e4084c",
        "322431789942595b599d2b86670f41b3", "35d7b32e37bcc23d02b12b03b1e0c093", "1b8847dd786fa41b5b38f5e7aa58b813",
        "126bdbe59fa82d55bfa098b710bdf650", "6c6d3248ed943dd7137637fc92d7ba37", "532203c6877433df5651b95d27685825",
        "6e7112411da5843bec576271c44e0a0a", "0e4f913a9b4672b0ad3cc9c4f2de5c8d", "a138d0060b2e8a0ef1f90cf3ab7b7ca0",
        "04d491dd1c679e91b5a2f3b9f14db555", "889c051e39305614accdff00414bfa67", "cd87cf24e5135c9d729940194ccc65c8",
        "75eb8bfe866cf3040f4c08a566c32bc1", "fd8e6ddb9c159aab781a11c287c85feb", "b91a2e103980b96f755712f2b559ad66",
        "26187d09b093649686d7c158aa4fd113", "62e16894bb4b8951a82befd24ad4ee21", "85c001df1d91788bf3cc1f97230b1dac",
        "91a25af808351757b101a8c9c787db9e", "8b3e8645cc4c2484c331759b9d1df5bc", "e8a76a6ab290e6743233510e8d1eb4a5",
        "38280bd7804f4c060b0775c4abed9b89"
    ]
    species_nsplit = [
        2, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 20,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, #1, 1, 1, 1, 1
    ]
    split_length = 22000

    def __init__(self, path, species_id=0, split_id=0, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        species_name = os.path.basename(self.urls[species_id])[:-4]
        if split_id >= self.species_nsplit[species_id]:
            raise ValueError("Split id %d should be less than %d in species %s" %
                            (split_id, self.species_nsplit[species_id], species_name))
        self.processed_file = "%s_%d.pkl.gz" % (species_name, split_id)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            tar_file = utils.download(self.urls[species_id], path, md5=self.md5s[species_id])
            pdb_path = utils.extract(tar_file)
            gz_files = sorted(glob.glob(os.path.join(pdb_path, "*.pdb.gz")))
            pdb_files = []
            index = slice(split_id * self.split_length, (split_id + 1) * self.split_length)
            for gz_file in gz_files[index]:
                pdb_files.append(utils.extract(gz_file))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))





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
    print("Starting")
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
