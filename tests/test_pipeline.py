from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from grants_tagger_light.models.bert_mesh import BertMeshHFCompat, BertMeshPipeline

PIPELINE_REGISTRY.register_pipeline(
    "grants-tagging", pipeline_class=BertMeshPipeline, pt_model=BertMeshHFCompat
)


def test_pipeline():
    pipe = pipeline("grants-tagging", model="Wellcome/WellcomeBertMesh")

    out = pipe("This grant is about malaria")
    assert "Malaria" in out[0]
    assert "Neoplasms" not in out[0]
