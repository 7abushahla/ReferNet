"""Fixed full-population training service, not a CLI or launch authorization.

Consume a context authenticated by the diagnostic staging loader and an
externally hashed successful same-source both-arm diagnostic receipt. Adding
this source requires new diagnostics. Existing receipts never waive source
identity. No diagnostic checkpoint, resume path or epoch override is accepted.
Training receipts must include full provenance bound by their original full
initialization hash. Only base.checkpoint_path is excluded from cross-allocation
comparison; the current path still must match its authenticated weights authority.

prepare_training builds an independent 1778-train/222-val registration table,
with training entries first. Pass its returned prepared value to both calls of
run_training to share that frozen table without refitting. Neither call selects
a family or authorizes a job. All output directories must be new.

Selected replay constructs another fresh original model and checks all 222
validation rows through the existing core. It is an in-process replay, not a
claim of fresh-process execution. Allocated execution and independent review
remain coordinator responsibilities.
"""
from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import direct_reference as bridge
from .data.query import load_role_rows
from .direct_reference_diagnostic import (
    _checkpoint_identities, _dataset, _model, _preprocess, bind, load_context,
    pixel_audit, require, write_receipt,
)
from .direct_reference_pilot import (
    _validate_contract, _validate_model, evaluate_direct_reference,
    read_direct_reference_checkpoint, replay_direct_reference_checkpoint,
    save_direct_reference_checkpoint, train_direct_reference_epoch,
)
from .pilots import _selection_record, _write_csv, json_sha256, make_pilot_optimizer, pilot_scheduler, tensor_sha256
from .train import selection_key


def _inputs(context, gate_binding):
    authenticated = load_context(Path(context["request"]["path"]), context["request"]["sha256"],
                                 context["run_dir"], context["device"])
    for key in ("manifest", "config", "identities", "weights_sha256", "run_dir", "device", "request"):
        require(context[key] == authenticated[key], f"authenticated context mismatch: {key}")
    config = authenticated["config"]
    fixed = dict(epochs=100, seed=42, head_seed=42, adapter_seed=42, fold=0, batch_size=8, eval_batch_size=1,
                 accumulation_steps=1, drop_last=True, eval_drop_last=False, input_shape=[4,512,800], mask_weight=0.0)
    require(all(config.get(key) == value and type(config.get(key)) is type(value) for key,value in fixed.items()), "full training recipe mismatch")
    require(config.get("selection") == dict(primary="accuracy",tie_break="auc",exact_tie="earlier_epoch"), "selection recipe mismatch")
    require(config.get("scheduler") == dict(name="warmup_cosine",warmup_iters=1000,warmup_cap_fraction=.1), "scheduler recipe mismatch")
    receipt = bridge._json(gate_binding)
    require(receipt.get("schema_version") == "direct_reference_diagnostic_v1" and receipt.get("diagnostic_only") is True,
            "same-source diagnostic receipt required")
    require(receipt.get("passed") is True and receipt.get("no_test_open") is True, "diagnostic gate failed")
    require(receipt.get("identities") == authenticated["identities"], "diagnostic source/context identity mismatch")
    for name, smoke in (("training_gates",False),("smoke_gates",True)):
        gates = receipt.get(name)
        require(isinstance(gates,dict) and set(gates) == {"off","on"}, f"both-mode {name} required")
        for arm, enabled in (("off",False),("on",True)):
            bridge._gate(gates[arm],authenticated["identities"],authenticated["weights_sha256"],enabled,smoke)
            if not smoke: _receipt_initialization(gates[arm])
    authorities = authenticated["manifest"]["authorities"]
    records, split = bridge._bytes(authorities["records"]), bridge._bytes(authorities["split"])
    rows = {role: load_role_rows(BytesIO(records),BytesIO(split),role) for role in ("train","val")}
    for role,count in (("train",1778),("val",222)):
        require(len(rows[role]) == count and rows[role].role.eq(role).all() and not rows[role].image_id.duplicated().any(),
                f"full official {role} membership required")
    require(not set(rows["train"].image_id) & set(rows["val"].image_id), "train/validation membership overlaps")
    return authenticated, receipt, rows


def _phase(context, receipt, rows, gate_binding):
    return dict(schema_version="direct_reference_phase_v1",phase="train",**context["identities"],
                authorized_image_ids={role: sorted(frame.image_id.tolist()) for role,frame in rows.items()},
                training_gates=receipt["training_gates"],smoke_gates=receipt["smoke_gates"],diagnostic_gate=gate_binding)


def _load_prepared(context, receipt, rows, gate_binding, prepared):
    union = pd.concat([rows["train"],rows["val"]],ignore_index=True)
    preprocess = prepared["preprocess"]
    authority = preprocess["phase_authority"]
    require(bridge._json(authority) == _phase(context,receipt,rows,gate_binding), "full training phase authority mismatch")
    require(preprocess == _preprocess(context,authority), "full training preprocessing mismatch")
    enrollment = bridge._json(preprocess["enrollment_authority"])
    table_binding = prepared["table_binding"]
    content = bridge._json(table_binding)
    require(isinstance(content.get("entries"),dict) and set(content["entries"]) == set(union.image_id),
            "full independent registration population required")
    table = bridge.load_registration_table(Path(table_binding["path"]),table_binding["sha256"],rows=union,
                                            enrollment_manifest=enrollment,preprocess=preprocess)
    return dict(preprocess=preprocess,enrollment=enrollment,table_binding=table_binding,table=table,rows=union)


def prepare_training(context, gate_binding):
    """Authenticate gates first, then create one shared full registration table."""
    context, receipt, rows = _inputs(context,gate_binding)
    directory = context["run_dir"] / "training" / "shared"
    directory.mkdir(parents=True,exist_ok=False)
    phase_binding = write_receipt(directory / "phase-authority.json",_phase(context,receipt,rows,gate_binding))
    preprocess = _preprocess(context,phase_binding)
    enrollment = bridge._json(preprocess["enrollment_authority"])
    union = pd.concat([rows["train"],rows["val"]],ignore_index=True)
    with pixel_audit(context,union) as opened:
        table = bridge.build_registration_table(union,Path(preprocess["raw_root"]),enrollment,preprocess=preprocess)
    require(len(opened) == 2000 and {(entry["image_id"],entry["role"]) for entry in opened}
            == {(row.image_id,row.role) for row in union.itertuples()}, "full registration query read membership mismatch")
    table_binding = write_receipt(directory / "registration.json",table)
    write_receipt(directory / "registration-query-reads.json",dict(phase="train",shared=True,query_reads=opened,
                  scope="query_decode_helper_reads",registration=table_binding,phase_authority=phase_binding))
    return _load_prepared(context,receipt,rows,gate_binding,dict(preprocess=preprocess,table_binding=table_binding))


def _receipt_initialization(binding):
    initialization = bridge._json(binding).get("initialization")
    require(isinstance(initialization,dict), "full initialization receipt required")
    provenance = initialization.get("provenance")
    require(isinstance(provenance,dict) and json_sha256(provenance) == initialization.get("initialization_sha256"),
            "initialization provenance payload/hash mismatch")
    base = provenance.get("base")
    require(isinstance(base,dict) and isinstance(base.get("checkpoint_path"),str)
            and Path(base["checkpoint_path"]).is_absolute(), "initialization checkpoint location required")
    return initialization


def _fresh_model(context, enabled, receipt):
    initialization = _receipt_initialization(receipt["training_gates"]["on" if enabled else "off"])
    model = _model(context,enabled)
    require(tensor_sha256(model.state_dict()) == initialization["state_sha256"], "training must reset original model tensors")
    current = copy.deepcopy(model.initialization_provenance)
    prior = copy.deepcopy(initialization["provenance"])
    weights = context["manifest"]["authorities"]["weights"]
    require(current["base"].get("checkpoint_path") == str(bridge._path(weights["path"])), "current initialization weights path mismatch")
    require(current["base"].get("checkpoint_sha256") == weights["sha256"] == context["weights_sha256"], "current initialization weights hash mismatch")
    current["base"].pop("checkpoint_path")
    prior["base"].pop("checkpoint_path")
    require(current == prior, "fresh initialization scientific provenance mismatch")
    return model


def _predictions(summary, predictions, rows):
    require(type(summary.get("n_images")) is int and summary["n_images"] == 222 and len(predictions) == 222,
            "complete selected validation population required")
    require([row["image_id"] for row in predictions] == rows.image_id.tolist(), "validation prediction membership/order mismatch")
    for prediction,row in zip(predictions,rows.to_dict("records")):
        require(all(prediction[key] == row[key] for key in ("design_id","attack_family","label")), "validation prediction metadata mismatch")


def run_training(context, enabled, gate_binding, *, prepared=None):
    """Run the fixed 100-epoch recipe only after externally bound diagnostics."""
    require(type(enabled) is bool,"reference_enabled must be boolean")
    context, receipt, rows = _inputs(context,gate_binding)
    if prepared is None:
        prepared = prepare_training(context,gate_binding)
    prepared = _load_prepared(context,receipt,rows,gate_binding,prepared)
    arm = "on" if enabled else "off"
    directory = context["run_dir"] / "training" / arm
    directory.mkdir(parents=True,exist_ok=False)
    config = copy.deepcopy(context["config"])
    config.update(arm=arm,reference_enabled=enabled)
    with pixel_audit(context,prepared["rows"]):
        model = _fresh_model(context,enabled,receipt)
        identities = _checkpoint_identities(context,config,prepared["preprocess"],prepared["table_binding"],
                                           json_sha256(model.initialization_provenance))
        fresh_state_sha256 = tensor_sha256(model.state_dict())
        _validate_contract(config,identities)
        _validate_model(model,config,identities)
        optimizer = make_pilot_optimizer(model,{"track":"D_running_pilot"},"train")
        require(not optimizer.state,"fresh optimizer state required")
        scheduler = pilot_scheduler(optimizer,"D_running_pilot",22200)
        train_dataset = _dataset(context,prepared,rows["train"],enabled,augmented=True)
        val_dataset = _dataset(context,prepared,rows["val"],enabled)
        val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=False,num_workers=0,
                                generator=torch.Generator().manual_seed(config["seed"]+3001))
        sampler = torch.Generator().manual_seed(config["seed"]+1001)
        history, omissions = [], []
        best_key, best_epoch, selected_predictions = None,0,None
        for epoch in range(1,101):
            order = torch.randperm(1778,generator=sampler).tolist()
            omissions.append(dict(epoch=epoch,image_ids=rows["train"].iloc[order[1776:]].image_id.tolist()))
            loader = DataLoader(train_dataset,batch_size=8,sampler=order,drop_last=True,num_workers=0,
                                generator=torch.Generator().manual_seed(config["seed"]+2001+epoch))
            stats = train_direct_reference_epoch(model,loader,optimizer,scheduler,context["device"])
            require(stats.get("n_images") == 1776 and stats.get("optimizer_steps") == 222,"full epoch update/drop-last mismatch")
            summary, predictions = evaluate_direct_reference(model,val_loader,context["device"])
            _predictions(summary,predictions,rows["val"])
            key = selection_key("accuracy",summary["accuracy"],summary["auc"],summary["image_loss"],stats["loss"])
            history.append(dict(epoch=epoch,**stats,val_accuracy=summary["accuracy"],val_auc=summary["auc"],
                                val_image_loss=summary["image_loss"],lr_pretrained=optimizer.param_groups[0]["lr"],
                                lr_new=optimizer.param_groups[-1]["lr"]))
            _write_csv(directory / "log.csv",history)
            save_direct_reference_checkpoint(directory / "last.pt",model,config,identities,epoch,key,"train")
            if best_key is None or key > best_key:
                best_key,best_epoch = key,epoch
                selected_predictions = copy.deepcopy(predictions)
                save_direct_reference_checkpoint(directory / "best.pt",model,config,identities,epoch,key,"train")
        require(scheduler.last_epoch == 22200,"scheduler did not complete 22200 updates")
        checkpoint = read_direct_reference_checkpoint(directory / "best.pt",config,identities)
        require(checkpoint["phase"] == "train" and checkpoint["epoch"] == best_epoch and checkpoint["selection_key"] == _selection_record(best_key),
                "selected training checkpoint mismatch")
        replay_model = _fresh_model(context,enabled,receipt)
        summary,predictions,replay = replay_direct_reference_checkpoint(directory / "best.pt",replay_model,val_loader,context["device"],
                    config=config,identities=identities,expected_predictions=selected_predictions)
        _predictions(summary,predictions,rows["val"])
        require(replay.get("passed") is True,"selected full-validation replay failed")
        replay = dict(replay,fresh_process=False,n_images=222,phase="train")
        _write_csv(directory / "val_predictions.csv",predictions)
        write_receipt(directory / "val_summary.json",dict(summary,role="val",selected_epoch=best_epoch))
        write_receipt(directory / "replay.json",replay)
    result = dict(passed=True,phase="train",diagnostic_only=False,epochs_completed=100,selected_epoch=best_epoch,
                  optimizer_steps=22200,role="val",arm=arm,fresh_initial_state_sha256=fresh_state_sha256,
                  scheduler=dict(total_updates=22200,warmup_updates=1000),identities=identities,selection_key=_selection_record(best_key),
                  omitted_train_ids=omissions,replay=replay,checkpoint=bind(directory / "best.pt"),
                  predictions=bind(directory / "val_predictions.csv"),registration=prepared["table_binding"],diagnostic_gate=gate_binding)
    write_receipt(directory / "train.json",result)
    return result
