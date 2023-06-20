import xml.etree.ElementTree as ET
import argparse
import awswrangler as wr
import pandas as pd
import random
import torch

from tqdm import tqdm
from grants_tagger_light.predict import predict_tags as predict_tags_bertmesh
from xlinear_predict import predict_tags as predict_tags_xlinear

random.seed(42)
torch.manual_seed(42)


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
    mesh_metadata_path: str,
    mesh_terms_list_path: str,
    bertmesh_path: str,
    xlinear_path: str,
    xlinear_label_binarizer_path: str,
    output_path: str,
):
    subnames = find_subnames(mesh_metadata_path, mesh_terms_list_path)

    parquet_files = wr.s3.list_objects(s3_url)
    random.shuffle(parquet_files)

    all_dfs = []

    for idx in tqdm(range(num_parquet_files_to_consider)):
        df = wr.s3.read_parquet(
            parquet_files[idx],
        )

        all_dfs.append(df)

    df = pd.concat(all_dfs)

    # Filter out rows where abstract is na
    df = df[~df["abstract"].isna()]

    # Do stratified sampling based on for_first_level_name column
    df_sample = df.groupby("for_first_level_name", group_keys=False).apply(
        lambda x: x.sample(min(len(x), num_samples_per_cat))
    )

    # Annotate with bertmesh
    abstracts = df_sample["abstract"].tolist()

    tags = predict_tags_bertmesh(abstracts, bertmesh_path, return_labels=True)

    df_sample["bertmesh_terms"] = tags
    # Keep 1st elem of each list (above func returns them nested)
    df_sample["bertmesh_terms"] = df_sample["bertmesh_terms"].apply(lambda x: x[0])

    # Annotate with xlinear
    tags = predict_tags_xlinear(
        X=abstracts,
        model_path=xlinear_path,
        label_binarizer_path=xlinear_label_binarizer_path,
    )

    df_sample["xlinear_terms"] = tags

    # Filter out rows where none of the bertmesh tags are in the subnames list
    df_sample = df_sample[
        df_sample["bertmesh_terms"].apply(lambda x: any([tag in subnames for tag in x]))
    ]

    # Output df to csv
    df_sample.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-url", type=str)
    parser.add_argument("--num-parquet-files-to-consider", type=int, default=1)
    parser.add_argument("--num-samples-per-cat", type=int, default=10)
    parser.add_argument("--mesh-metadata-path", type=str)
    parser.add_argument("--mesh-terms-list-path", type=str)
    parser.add_argument(
        "--bertmesh-path", type=str, default="Wellcome/WellcomeBertMesh"
    )
    parser.add_argument("--bertmesh-thresh", type=float, default=0.5)
    parser.add_argument("--xlinear-path", type=str)
    parser.add_argument("--xlinear-label-binarizer-path", type=str)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    create_comparison_csv(
        s3_url=args.s3_url,
        num_parquet_files_to_consider=args.num_parquet_files_to_consider,
        num_samples_per_cat=args.num_samples_per_cat,
        mesh_metadata_path=args.mesh_metadata_path,
        mesh_terms_list_path=args.mesh_terms_list_path,
        bertmesh_path=args.bertmesh_path,
        xlinear_path=args.xlinear_path,
        xlinear_label_binarizer_path=args.xlinear_label_binarizer_path,
        output_path=args.output_path,
    )
