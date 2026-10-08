"""Build the small reviewable common/geometry-only transfer bundle locally."""
import argparse
from pathlib import Path
import zipfile
from .io import file_sha256,write_json

def build(output):
    repo=Path(__file__).parents[2];output=Path(output).resolve()
    manifest_path=output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists(): raise ValueError("Bundle output or manifest already exists; use a new name")
    paths=[]
    for folder in ("src/pipeline_common","src/path_mapping","tests/pipeline_common","tests/path_mapping"):
        paths.extend(p for p in (repo/folder).rglob("*.py") if "__pycache__" not in p.parts)
    for relative in ("src/pipelines/__init__.py","src/pipelines/geometry_only.py","src/prepare_sequence.py","src/run_pipeline.py","src/evaluate_pipeline.py","src/compare_pipelines.py",
        "VLM_evaluation/src/traversability_hazard_inference/preparation.py","VLM_evaluation/src/traversability_hazard_segmentation/sam3_adapter.py",
        "configs/pipelines/geometry_only.json","configs/experiments/prepare_video_smoke.json",
        "configs/fixtures/robot.synthetic.json","configs/fixtures/run.synthetic.json",
        "docs/implementation/shared_contract.md","docs/implementation/geometry_only_runtime.md",
        "docs/run_storage.md",
        "requirements/path_mapping_cpu.txt","requirements/geometry_only_gpu.txt"):
        path=repo/relative
        if path.is_file(): paths.append(path)
    paths=sorted(set(paths))
    rows=[{"path":p.relative_to(repo).as_posix(),"sha256":file_sha256(p),"size_bytes":p.stat().st_size} for p in paths]
    output.parent.mkdir(parents=True,exist_ok=True)
    try: manifest_path.touch(exist_ok=False)
    except FileExistsError: raise ValueError("Bundle manifest already exists; use a new name") from None
    try:
        with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as archive:
            for path,row in zip(paths,rows): archive.write(path,row["path"])
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    manifest={"contract_id":"semantic_mapping_v1","schema_version":1,"bundle_path":str(output),"bundle_relative_path":output.name,"bundle_sha256":file_sha256(output),"file_count":len(rows),"files":rows,
        "scope":"common infrastructure and geometry_only; production adapters/configs for 2-4 are owned and bundled separately",
        "excluded":["weights","videos","datasets","credentials","upstream submodules","completed server runs","other sessions' production adapters"],
        "transfer_note":"review target file hashes and preserve server local changes before applying the full bridge reader files; supervisor handles transfer"}
    write_json(manifest_path,manifest);return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();result=build(args.output);print(f"{result['file_count']} files: {result['bundle_path']} ({result['bundle_sha256']})")
