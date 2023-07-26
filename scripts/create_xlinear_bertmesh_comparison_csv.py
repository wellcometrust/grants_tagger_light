import xml.etree.ElementTree as ET
import argparse
import awswrangler as wr
import pandas as pd
import random
import torch

from tqdm import tqdm
from grants_tagger_light.predict import predict_tags as predict_tags_bertmesh
from grants_tagger_light.models.xlinear import MeshXLinear
from loguru import logger

random.seed(42)
torch.manual_seed(42)

WELLCOME_FUNDER_ID = "grid.52788.30"


def load_mesh_terms_from_file(mesh_terms_list_path):
    with open(mesh_terms_list_path, "r") as f:
        mesh_terms = f.readlines()
    mesh_terms = [term.strip() for term in mesh_terms]
    return mesh_terms


def _extract_data(mesh_elem):
    # TreeNumberList e.g. A11.118.637.555.567.550.500.100
    tree_number = mesh_elem[-2][0].text
    # DescriptorUI e.g. M000616943
    code = mesh_elem[0].text
    # DescriptorName e.g. Mucosal-Associated Invariant T Cells
    name = mesh_elem[1][0].text

    return tree_number, code, name


def find_subnames(mesh_metadata_path: str, mesh_terms_list_path: str):
    """
    Given a path to a file containing a list of MeSH terms and a path to a file
    containing MeSH metadata, returns a list of all MeSH terms that are subnames
    of the MeSH terms in the input file.

    Args:
        mesh_metadata_path (str): The path to the MeSH metadata file.
        mesh_terms_list_path (str): The file containing the list of MeSH terms
                                    to filter by.

    Returns:
        List[str]: A list of all MeSH terms that are subnames
                   of the MeSH terms in the input file.
    """
    mesh_tree = ET.parse(mesh_metadata_path)
    mesh_terms = load_mesh_terms_from_file(mesh_terms_list_path)
    root = mesh_tree.getroot()

    # Do 1st pass to get all their codes
    top_level_tree_numbers = []
    pbar = tqdm(root)
    pbar.set_description("Finding top level tree numbers")
    for mesh_elem in pbar:
        try:
            tree_number, _, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        if name not in mesh_terms:
            continue
        else:
            top_level_tree_numbers.append(tree_number)

    # Do 2nd pass to collect all names that are in the same tree as the ones we found
    all_subnames = []
    pbar = tqdm(root)
    pbar.set_description("Finding subnames")
    for mesh_elem in pbar:
        try:
            curr_tree_number, _, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        for top_level_tree_number in top_level_tree_numbers:
            if curr_tree_number.startswith(top_level_tree_number):
                all_subnames.append(name)
                break

    return all_subnames


def create_comparison_csv(
    s3_url: str,
    num_parquet_files_to_consider: int,
    num_samples_per_cat: int,
    num_wellcome_grants_to_add: int,
    mesh_metadata_path: str,
    mesh_terms_list_path: str,
    pre_annotate_bertmesh: bool,
    bertmesh_path: str,
    bertmesh_thresh: float,
    pre_annotate_xlinear: bool,
    xlinear_path: str,
    xlinear_label_binarizer_path: str,
    xlinear_thresh: float,
    output_path: str,
):
    subnames = find_subnames(mesh_metadata_path, mesh_terms_list_path)

    parquet_files = wr.s3.list_objects(s3_url)
    random.shuffle(parquet_files)

    grants = []

    for idx in tqdm(range(num_parquet_files_to_consider)):
        grants.append(
            wr.s3.read_parquet(
                parquet_files[idx],
            )
        )

    all_grants = pd.concat(grants)

    # Filter out rows where abstract is na
    all_grants = all_grants[~all_grants["abstract"].isna()]

    # Reshuffle parquet files, search for Wellcome grants to add
    random.shuffle(parquet_files)
    wellcome_grants = []
    idx = 0
    total_len = 0

    while len(wellcome_grants) < num_wellcome_grants_to_add and idx <= len(
        parquet_files
    ):
        pq_file = parquet_files[idx]
        pq_df = wr.s3.read_parquet(pq_file)
        pq_df = pq_df[pq_df["funder"] == WELLCOME_FUNDER_ID]

        if len(pq_df.index) > 0:
            wellcome_grants.append(pq_df)
            total_len += len(pq_df.index)
            logger.info("Added {} Wellcome grants".format(len(pq_df)))
        idx += 1

        # If already exceeds num_wellcome_grants_to_add, break
        if total_len >= num_wellcome_grants_to_add:
            break

    # Do stratified sampling based on for_first_level_name column
    grants_sample = all_grants.groupby("for_first_level_name", group_keys=False).apply(
        lambda x: x.sample(min(len(x), num_samples_per_cat))
    )

    wellcome_grants = pd.concat(wellcome_grants)
    wellcome_grants = wellcome_grants[~wellcome_grants["abstract"].isna()]
    wellcome_grants = wellcome_grants[:num_wellcome_grants_to_add]

    grants_sample = pd.concat([grants_sample, wellcome_grants])

    abstracts = grants_sample["abstract"].tolist()

    # Annotate with bertmesh
    if pre_annotate_bertmesh:
        tags = predict_tags_bertmesh(
            abstracts,
            bertmesh_path,
            return_labels=True,
            threshold=bertmesh_thresh,
        )

        grants_sample["bertmesh_terms"] = tags
        # Keep 1st elem of each list (above func returns them nested)
        grants_sample["bertmesh_terms"] = grants_sample["bertmesh_terms"].apply(
            lambda x: x[0]
        )

    # Annotate with xlinear
    if pre_annotate_xlinear:
        model = MeshXLinear(
            model_path=xlinear_path,
            label_binarizer_path=xlinear_label_binarizer_path,
        )

        tags = model(X=abstracts, threshold=xlinear_thresh)

        grants_sample["xlinear_terms"] = tags

    if pre_annotate_bertmesh:
        # Filter out rows where none of the bertmesh tags are in the subnames list
        grants_sample = grants_sample[
            grants_sample["bertmesh_terms"].apply(
                lambda x: any([tag in subnames for tag in x])
            )
        ]

    if pre_annotate_bertmesh and pre_annotate_xlinear:
        # Add column with common terms
        grants_sample["common_terms"] = grants_sample.apply(
            lambda x: set(x["bertmesh_terms"]).intersection(set(x["xlinear_terms"])),
            axis=1,
        )

        # Add columns with bertmesh_only and xlinear_only
        grants_sample["bertmesh_only"] = grants_sample.apply(
            lambda x: set(x["bertmesh_terms"]).difference(set(x["xlinear_terms"])),
            axis=1,
        )

        grants_sample["xlinear_only"] = grants_sample.apply(
            lambda x: set(x["xlinear_terms"]).difference(set(x["bertmesh_terms"])),
            axis=1,
        )

    # Add column for whether it is a Wellcome grant or not
    grants_sample["is_wellcome_grant"] = grants_sample["funder"].apply(
        lambda x: x == "grid.52788.30"
    )

    # Output df to csv
    grants_sample.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-url", type=str)
    parser.add_argument("--num-parquet-files-to-consider", type=int, default=1)
    parser.add_argument("--num-samples-per-cat", type=int, default=10)
    parser.add_argument("--num-wellcome-grants-to-add", type=int, default=100)
    parser.add_argument("--mesh-metadata-path", type=str)
    parser.add_argument("--mesh-terms-list-path", type=str)
    parser.add_argument("--pre-annotate-bertmesh", action="store_true")
    parser.add_argument(
        "--bertmesh-path", type=str, default="Wellcome/WellcomeBertMesh"
    )
    parser.add_argument("--bertmesh-thresh", type=float, default=0.5)
    parser.add_argument("--pre-annotate-xlinear", action="store_true")
    parser.add_argument("--xlinear-path", type=str)
    parser.add_argument("--xlinear-label-binarizer-path", type=str)
    parser.add_argument("--xlinear-thresh", type=float, default=0.5)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    create_comparison_csv(
        s3_url=args.s3_url,
        num_parquet_files_to_consider=args.num_parquet_files_to_consider,
        num_samples_per_cat=args.num_samples_per_cat,
        num_wellcome_grants_to_add=args.num_wellcome_grants_to_add,
        mesh_metadata_path=args.mesh_metadata_path,
        mesh_terms_list_path=args.mesh_terms_list_path,
        pre_annotate_bertmesh=args.pre_annotate_bertmesh,
        bertmesh_path=args.bertmesh_path,
        bertmesh_thresh=args.bertmesh_thresh,
        pre_annotate_xlinear=args.pre_annotate_xlinear,
        xlinear_path=args.xlinear_path,
        xlinear_label_binarizer_path=args.xlinear_label_binarizer_path,
        xlinear_thresh=args.xlinear_thresh,
        output_path=args.output_path,
    )
