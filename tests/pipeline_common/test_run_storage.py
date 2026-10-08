"""Attempt ownership and diagnostic durability, without model execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Barrier, Event, Lock
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from pipeline_common.bundle import build
from pipeline_common import bundle
from pipeline_common.fixture import create_fixture
from pipeline_common.fixture_adapters import FixtureAdapter
from pipeline_common.io import digest_json, project_code_identity, read_json, safe_path, write_json, write_jsonl, write_npz
from pipeline_common.runtime import new_attempt_path, run


@pytest.fixture
def scene(tmp_path):
    return create_fixture(tmp_path/"fixture")


def fixture_run(scene, output, *, cache=True):
    return run("fixed_hazards",scene/"sequence/sequence.json",scene/"run.json",output,
        fixture=True,cache=scene/"geometry_cache" if cache else None)


@pytest.mark.parametrize("kind",["empty","file","running","failed","complete","malformed"])
def test_existing_output_is_refused_byte_for_byte(scene,tmp_path,kind):
    output=tmp_path/"attempt"
    if kind=="file": output.write_bytes(b"existing file")
    else:
        output.mkdir()
        if kind in {"running","failed","complete"}: write_json(output/"run.json",{"status":kind})
        if kind=="malformed": (output/"run.json").write_text("{broken",encoding="utf-8")
    before={p.name:p.read_bytes() for p in output.iterdir()} if output.is_dir() else output.read_bytes()
    with pytest.raises(ValueError,match="completed|already exists"):
        fixture_run(scene,output)
    after={p.name:p.read_bytes() for p in output.iterdir()} if output.is_dir() else output.read_bytes()
    assert after==before


@pytest.mark.parametrize("failure",["missing_config","invalid_config","missing_sequence"])
def test_early_failure_keeps_owned_attempt_diagnostics(scene,tmp_path,failure):
    config=scene/"run.json";sequence=scene/"sequence/sequence.json"
    if failure=="missing_config": config=tmp_path/"missing.json"
    elif failure=="invalid_config":
        config=tmp_path/"invalid.json";write_json(config,{"contract_id":"wrong"})
    else: sequence=tmp_path/"missing-sequence.json"
    output=tmp_path/"attempt"
    with pytest.raises((ValueError,FileNotFoundError)):
        run("geometry_only",sequence,config,output,fixture=True)
    metadata=read_json(output/"run.json");detail=read_json(output/"failure.json")
    assert metadata["status"]=="failed" and metadata["stage"]=="validation"
    assert detail["error_type"] in {"ValueError","FileNotFoundError"}
    assert detail["traceback"] and metadata["attempt_id"] and metadata["finished_at_utc"]


def test_same_output_loser_cannot_mark_winner_failed(scene,tmp_path):
    started=Event();release=Event()
    class HeldCollector:
        def __init__(self,**kwargs): pass
        def start(self):
            started.set()
            assert release.wait(10),"test did not release owning run"
        def stop(self): return {}
    output=tmp_path/"collision"
    with patch("pipeline_common.resources.ResourceCollector",HeldCollector),ThreadPoolExecutor(max_workers=1) as pool:
        winner=pool.submit(fixture_run,scene,output)
        try:
            assert started.wait(10)
            before=(output/"run.json").read_bytes()
            with pytest.raises(ValueError,match="already exists"):
                fixture_run(scene,output)
            assert (output/"run.json").read_bytes()==before
        finally: release.set()
        assert winner.result(timeout=10)["status"]=="complete"
    assert read_json(output/"run.json")["status"]=="complete"


def test_interrupt_marks_failed_and_preserves_completed_geometry(scene,tmp_path):
    output=tmp_path/"interrupted"
    with patch("pipeline_common.fixture_adapters.create_fixture_adapter",side_effect=KeyboardInterrupt("stop")):
        with pytest.raises(KeyboardInterrupt): fixture_run(scene,output)
    assert read_json(output/"run.json")["status"]=="failed"
    assert read_json(output/"failure.json")["error_type"]=="KeyboardInterrupt"
    assert read_json(output/"geometry/manifest.json")["geometry_fingerprint"]
    assert "geometry_stage" in (output/"events.jsonl").read_text(encoding="utf-8")


def test_cleanup_failure_keeps_original_error_and_partial_artifacts(scene,tmp_path):
    class BrokenAdapter:
        def observe(self,frame): return "invalid SemanticFrame"
        def close(self): raise RuntimeError("cleanup also failed")
    output=tmp_path/"cleanup"
    with patch("pipeline_common.fixture_adapters.create_fixture_adapter",return_value=BrokenAdapter()):
        result=fixture_run(scene,output)
    assert result["status"]=="failed"
    assert result["reason"]=="Adapter must return SemanticFrame"
    events=[json.loads(line) for line in (output/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"]=="cleanup_error" for event in events)
    assert len((output/"frames.jsonl").read_text(encoding="utf-8").splitlines())==3
    assert (output/"map/voxels.npz").is_file()


def test_finalization_failure_preserves_prior_events_and_frame_records(scene,tmp_path):
    output=tmp_path/"finalization"
    with patch("pipeline_common.planning.build_costmap",side_effect=ValueError("planned failure")):
        with pytest.raises(ValueError,match="planned failure"): fixture_run(scene,output)
    assert read_json(output/"run.json")["status"]=="failed"
    assert read_json(output/"failure.json")["stage"]=="finalizing"
    events=[json.loads(line) for line in (output/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(event["event"]=="semantic_fusion" for event in events)==3
    assert len((output/"frames.jsonl").read_text(encoding="utf-8").splitlines())==3


def test_attempt_ids_vary_but_config_and_provenance_ids_are_stable(scene,tmp_path):
    # Keep code frozen even while another chat works in this shared checkout.
    snapshot=project_code_identity()
    with patch("pipeline_common.runtime.project_code_identity",return_value=snapshot):
        results=[fixture_run(scene,tmp_path/name) for name in ("first","second")]
    assert results[0]["attempt_id"]!=results[1]["attempt_id"]
    for key in ("requested_config_digest","initial_config_digest","config_digest","provenance_id"):
        assert results[0][key]==results[1][key]
    resolved=read_json(tmp_path/"first/config.resolved.json")
    assert results[0]["config_digest"]==digest_json({k:v for k,v in resolved.items() if k!="config_digest"})
    assert read_json(tmp_path/"first/provenance.json")["hardware_budget"]==resolved["hardware_budget"]


def test_project_code_fingerprint_detects_source_change_and_excludes_bulk(tmp_path):
    files={"src/pipeline_common/common.py":"common", "src/path_mapping/bridge.py":"bridge",
        "src/pipelines/adapter.py":"adapter", "src/run_pipeline.py":"cli",
        "VLM_evaluation/src/traversability_hazard_inference/qwen.py":"vlm",
        "src/lingbot-map/vendor.py":"vendor", "src/sam3-robot/vendor.py":"vendor",
        "data/image.py":"data", "outputs/generated.py":"output", "tests/check.py":"test"}
    for relative,value in files.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    initial=project_code_identity(tmp_path)
    assert initial["recipe"]=="sha256_project_python_sources_v1" and initial["file_count"]==5
    assert initial["sha256"]==digest_json(initial["files"])
    assert all(not Path(row["path"]).is_absolute() for row in initial["files"])
    relocated=tmp_path/"relocated"
    shutil.copytree(tmp_path/"src",relocated/"src")
    shutil.copytree(tmp_path/"VLM_evaluation",relocated/"VLM_evaluation")
    assert project_code_identity(relocated)==initial
    (tmp_path/"src/lingbot-map/vendor.py").write_text("changed excluded vendor",encoding="utf-8")
    assert project_code_identity(tmp_path)==initial
    (tmp_path/"src/pipeline_common/common.py").write_text("changed common implementation",encoding="utf-8")
    changed=project_code_identity(tmp_path)
    assert changed["sha256"]!=initial["sha256"] and changed["file_count"]==5


def test_code_change_alters_provenance_without_changing_config_gates(scene,tmp_path):
    source_root=tmp_path/"sources";source=source_root/"src/run_pipeline.py"
    source.parent.mkdir(parents=True);source.write_text("first code revision",encoding="utf-8")
    with patch("pipeline_common.runtime.project_code_identity",side_effect=lambda:project_code_identity(source_root)):
        first=fixture_run(scene,tmp_path/"first_code")
        source.write_text("second code revision",encoding="utf-8")
        second=fixture_run(scene,tmp_path/"second_code")
    assert first["config_digest"]==second["config_digest"]
    assert first["requested_config_digest"]==second["requested_config_digest"]
    assert first["provenance_id"]!=second["provenance_id"]
    for name,result in (("first_code",first),("second_code",second)):
        provenance=read_json(tmp_path/name/"provenance.json")
        assert provenance["provenance_id"]==result["provenance_id"]
        assert provenance["project_code"]["files"][0]["path"]=="src/run_pipeline.py"


def test_internal_artifact_references_survive_directory_move(scene,tmp_path):
    output=tmp_path/"original";fixture_run(scene,output,cache=False)
    moved=tmp_path/"moved";shutil.copytree(output,moved)
    reference=read_json(moved/"geometry/manifest.json")["cache_reference"]
    assert reference=={"kind":"run_relative","path":"geometry/cache"}
    assert (safe_path(moved,reference["path"])/"geometry.npz").is_file()
    rows=[json.loads(line) for line in (moved/"semantics/frames.jsonl").read_text(encoding="utf-8").splitlines()]
    masks=[instance for row in rows for query in row["queries"] for instance in query["instances"]]
    assert masks
    for instance in masks:
        with np.load(safe_path(moved,instance["mask_path"]),allow_pickle=False) as archive:
            assert archive[instance["mask_key"]].dtype==np.bool_


def test_output_root_cli_allocates_fresh_attempts_and_reports_paths(scene,tmp_path):
    repo=Path(__file__).resolve().parents[2];root=tmp_path/"runs"
    args=[sys.executable,str(repo/"src/run_pipeline.py"),"--pipeline","geometry_only",
        "--sequence",str(scene/"sequence/sequence.json"),"--config",str(scene/"run.json"),
        "--geometry-cache",str(scene/"geometry_cache"),"--output-root",str(root),"--fixture"]
    for _ in range(2):
        completed=subprocess.run(args,cwd=repo,capture_output=True,text=True,timeout=30)
        assert completed.returncode==0,completed.stderr
        assert "geometry_only: complete" in completed.stdout and str(root) in completed.stdout
    attempts=list(root.iterdir());assert len(attempts)==2
    assert all("geometry_only" in path.name and read_json(path/"run.json")["status"]=="complete" for path in attempts)
    assert new_attempt_path(root,"geometry_only") not in attempts


@pytest.mark.parametrize("writer",["json","jsonl","npz"])
def test_serialization_failure_preserves_destination_and_cleans_temps(tmp_path,writer):
    path=tmp_path/f"artifact.{writer}";path.write_bytes(b"previous")
    with pytest.raises((ValueError,TypeError)):
        if writer=="json": write_json(path,{"invalid":float("nan")})
        elif writer=="jsonl": write_jsonl(path,[{"ok":True},{"invalid":object()}])
        else: write_npz(path,bad=np.array([object()],dtype=object))
    assert path.read_bytes()==b"previous"
    assert list(tmp_path.iterdir())==[path]


def test_replace_failure_preserves_destination_and_cleans_temp(tmp_path):
    path=tmp_path/"artifact.json";write_json(path,{"before":True});before=path.read_bytes()
    with patch.object(Path,"replace",side_effect=OSError("replace unavailable")):
        with pytest.raises(OSError,match="replace unavailable"): write_json(path,{"after":True})
    assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]


def test_concurrent_atomic_writers_use_private_temporary_files(tmp_path):
    path=tmp_path/"shared.json";barrier=Barrier(2);replace=Path.replace;seen=set();lock=Lock()
    def rendezvous(temporary,destination):
        with lock:
            first=temporary not in seen;seen.add(temporary)
        if first: barrier.wait(timeout=10)
        return replace(temporary,destination)
    rows=[{"writer":index,"payload":[index]*1000} for index in range(2)]
    with patch.object(Path,"replace",rendezvous),ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(write_json,path,row) for row in rows]
        for future in futures: future.result(timeout=10)
    assert read_json(path) in rows and list(tmp_path.iterdir())==[path]
    assert len(seen)==2


def test_transient_windows_replace_conflict_is_retried(tmp_path):
    path=tmp_path/"retry.json";replace=Path.replace;calls=[]
    def transient(temporary,destination):
        calls.append(temporary)
        if len(calls)==1:
            error=PermissionError("temporary sharing violation");error.winerror=32
            raise error
        return replace(temporary,destination)
    with patch.object(Path,"replace",transient): write_json(path,{"complete":True})
    assert len(calls)==2 and read_json(path)=={"complete":True}
    assert list(tmp_path.iterdir())==[path]


def test_persistent_windows_replace_error_is_bounded_and_preserves_file(tmp_path):
    path=tmp_path/"locked.json";write_json(path,{"before":True});before=path.read_bytes()
    error=PermissionError("persistent access denial");error.winerror=5
    with patch.object(Path,"replace",side_effect=error) as replacement:
        with pytest.raises(PermissionError,match="persistent access denial"):
            write_json(path,{"after":True})
    assert replacement.call_count==4
    assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]


def test_bundle_refuses_existing_companion_manifest(tmp_path):
    manifest=tmp_path/"bundle.manifest.json";manifest.write_bytes(b"preserve")
    with pytest.raises(ValueError,match="already exists"): build(tmp_path/"bundle.zip")
    assert manifest.read_bytes()==b"preserve" and not (tmp_path/"bundle.zip").exists()


def test_concurrent_bundle_names_with_shared_manifest_cannot_overwrite(tmp_path):
    repo=tmp_path/"repo";source=repo/"src/pipeline_common/example.py"
    source.parent.mkdir(parents=True);source.write_text("# tiny local bundle fixture\n",encoding="utf-8")
    guide=repo/"docs/run_storage.md";guide.parent.mkdir();guide.write_text("# Storage\n",encoding="utf-8")
    barrier=Barrier(2);touch=Path.touch
    def rendezvous(path,*args,**kwargs):
        barrier.wait(timeout=10)
        return touch(path,*args,**kwargs)
    outputs=[tmp_path/"bundle.zip",tmp_path/"bundle.other"]
    results=[];errors=[]
    with patch.object(bundle,"__file__",str(repo/"src/pipeline_common/bundle.py")),\
            patch.object(Path,"touch",rendezvous),ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(build,output) for output in outputs]
        for future in futures:
            try: results.append(future.result(timeout=10))
            except ValueError as error: errors.append(error)
    assert len(results)==len(errors)==1
    manifest=read_json(tmp_path/"bundle.manifest.json")
    assert manifest==results[0]
    assert sum(output.exists() for output in outputs)==1
    assert manifest["bundle_relative_path"]==Path(manifest["bundle_path"]).name
    with zipfile.ZipFile(manifest["bundle_path"]) as archive:
        assert set(archive.namelist())=={"src/pipeline_common/example.py","docs/run_storage.md"}
        for row in manifest["files"]:
            content=archive.read(row["path"])
            assert hashlib.sha256(content).hexdigest()==row["sha256"]
            assert len(content)==row["size_bytes"]


def test_diagnostic_flush_failure_does_not_replace_interrupt(scene,tmp_path):
    output=tmp_path/"flush-failure";writer=write_jsonl
    def fail_final_flush(path,rows):
        if Path(path).name=="events.jsonl" and any(row.get("event")=="geometry_stage" for row in rows):
            # The first stage checkpoint succeeds; fail the flush during unwinding.
            if Path(path).exists(): raise OSError("event persistence unavailable")
        return writer(path,rows)
    with patch("pipeline_common.runtime.write_jsonl",side_effect=fail_final_flush),\
            patch("pipeline_common.fixture_adapters.create_fixture_adapter",side_effect=KeyboardInterrupt("original interrupt")):
        with pytest.raises(KeyboardInterrupt,match="original interrupt") as caught:
            fixture_run(scene,output)
    assert any("events.jsonl" in note for note in caught.value.__notes__)
    assert read_json(output/"failure.json")["error_type"]=="KeyboardInterrupt"
