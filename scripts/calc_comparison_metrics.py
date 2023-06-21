import pandas as pd
import argparse
import ast
from grants_tagger_light.models.bert_mesh import BertMesh
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import cohen_kappa_score


def calc_comparison_metrics(csv_path: str):
    comparison = pd.read_csv(csv_path)

    # Load model to get its id2label
    model = BertMesh.from_pretrained("Wellcome/WellcomeBertMesh")
    id2label = model.id2label

    mlb = MultiLabelBinarizer(classes=list(id2label.values()))
    mlb.fit([list(id2label.values())])

    bertmesh_terms = comparison["bertmesh_terms"].values.tolist()
    bertmesh_terms = [ast.literal_eval(x) for x in bertmesh_terms]
    bertmesh_terms = mlb.transform(bertmesh_terms)

    xlinear_terms = comparison["xlinear_terms"].values.tolist()
    xlinear_terms = [ast.literal_eval(x) for x in xlinear_terms]
    xlinear_terms = mlb.transform(xlinear_terms)

    # Calc Cohen Kappa for each label and then average
    cohen_kappas = []

    # For every label, create a list of the predictions for that label
    for i in range(bertmesh_terms.shape[1]):
        bertmesh_label_preds = bertmesh_terms[:, i]
        xlinear_label_preds = xlinear_terms[:, i]

        # If there are no predictions for this label, skip it
        if sum(bertmesh_label_preds) == 0 and sum(xlinear_label_preds) == 0:
            continue

        cohen_kappa = cohen_kappa_score(bertmesh_label_preds, xlinear_label_preds)
        cohen_kappas.append(cohen_kappa)

    print("Average Cohen Kappa: ", sum(cohen_kappas) / len(cohen_kappas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str)
    args = parser.parse_args()

    calc_comparison_metrics(args.csv_path)
