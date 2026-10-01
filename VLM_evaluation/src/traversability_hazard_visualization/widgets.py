"""Small notebook interface; callbacks only select indexed evidence and render."""

import html
import json

from .rendering import (render_discovery, render_image, render_segmentation,
                        render_deployment, image_diagnostics, export_figure)


def create_viewer(saved):
    """Return an ipywidgets Tab. Missing frontend dependencies leave headless APIs usable."""
    import ipywidgets as w
    from IPython.display import display
    import matplotlib.pyplot as plt

    def details(value):
        return w.HTML("<details><summary>Saved evidence and diagnostics</summary><pre style='white-space:pre-wrap'>"
                      + html.escape(json.dumps(value, indent=2, ensure_ascii=True)) + "</pre></details>")

    def dropdown(label, values, preferred=None, all_value=False):
        values = list(values)
        options = ([("all", None)] if all_value else []) + [(str(v), v) for v in values]
        if not options:
            options = [("unavailable", None)]
        selected = preferred if preferred in [v for _, v in options] else options[0][1]
        return w.Dropdown(description=label, options=options, value=selected,
                          layout=w.Layout(width="auto", min_width="200px"))

    def panel(controls, render, evidence=None):
        output = w.Output()
        state = {}
        export_name = w.Text(value="view.png", description="Filename")
        export_button = w.Button(description="Export PNG/SVG", disabled=not saved.config.get("export_enabled", False))
        export_status = w.HTML("Export disabled by configuration." if export_button.disabled else "Explicit export only; existing files are never overwritten.")
        def refresh(change=None):
            with output:
                output.clear_output(wait=True)
                if state.get("figure") is not None:
                    plt.close(state["figure"])
                try:
                    fig = render()
                    state["figure"] = fig
                    display(fig)
                    plt.close(fig)  # Avoid duplicate automatic display; figure remains exportable.
                    if evidence:
                        display(details(evidence()))
                except (ValueError, KeyError, OSError) as error:
                    state["figure"] = None
                    display(w.HTML("<b>Saved view unavailable:</b> " + html.escape(str(error))))
        def export_clicked(button):
            try:
                if state.get("figure") is None:
                    raise ValueError("No available rendered figure")
                path = export_figure(saved, state["figure"], export_name.value, enabled=True)
                export_status.value = "Saved " + html.escape(str(path))
            except (ValueError, OSError) as error:
                export_status.value = html.escape(str(error))
        for control in controls:
            control.observe(refresh, names="value")
        export_button.on_click(export_clicked)
        refresh()
        return w.VBox([w.HBox(controls, layout=w.Layout(flex_flow="row wrap")), output,
                       w.HBox([export_name, export_button]), export_status])

    sources = sorted({r["source"] for r in saved.rows("concepts") if r["source"] != "all"})
    splits = sorted({r["split"] for r in saved.rows("concepts")})
    models = sorted({r["model_key"] for r in saved.rows("concepts")})
    ds = dropdown("Source", ["all", *sources], "all")
    dt = dropdown("Split", splits, "test")
    dm = dropdown("Model", models, all_value=True)
    discovery = panel([ds, dt, dm], lambda: render_discovery(saved, source=ds.value, split=dt.value, model=dm.value),
                      lambda: saved.rows("concepts", source=ds.value, split=dt.value, model_key=dm.value))

    es = dropdown("Source", sorted({f["source"] for f in saved.frames.values()}))
    et = dropdown("Split", sorted({f["split"] for f in saved.frames.values()}), "test")
    ef = dropdown("Frame", [])
    em = dropdown("Model", models)
    ec = dropdown("SAM condition", [])
    eq = dropdown("Query", [])
    eg = dropdown("Reference", ["aggregate union"])
    diagnostic = dropdown("Overlay", ["scored", "all_query", "failed_partial"], "scored")
    opacity = w.FloatSlider(description="Opacity", value=.45, min=0, max=1, step=.05, continuous_update=False)
    updating = {"active": False}
    def update_frames(change=None):
        if updating["active"]:
            return
        updating["active"] = True
        identifiers = sorted(k for k, f in saved.frames.items() if f["source"] == es.value and f["split"] == et.value)
        previous = ef.value
        ef.options = [(k, k) for k in identifiers] or [("unavailable", None)]
        if previous in identifiers:
            ef.value = previous
        else:
            ef.value = identifiers[0] if identifiers else None
        updating["active"] = False
        update_sam()
    def update_sam(change=None):
        if updating["active"]:
            return
        updating["active"] = True
        conditions = sorted(c for c, f in saved.sam_audits if f == ef.value)
        prior = ec.value
        ec.options = [("no saved SAM", None), *[(c, c) for c in conditions]]
        match = "vlm__" + str(em.value)
        ec.value = match if match in conditions else (prior if prior in conditions else None)
        vocabulary = saved.references.get(ef.value, {}).get("scored_concepts", [])
        eg.options = [("aggregate union", None), *[(c, c) for c in vocabulary]]
        updating["active"] = False
        update_queries()
    def update_queries(change=None):
        if updating["active"]:
            return
        queries = saved.sam_audits.get((ec.value, ef.value), {}).get("queries", [])
        eq.options = [("union view", None), *[(q["phrase"] + " / " + q["status"] + " / " + q["query_id"].rsplit(":", 1)[-1], q["query_id"]) for q in queries]]
    es.observe(update_frames, names="value")
    et.observe(update_frames, names="value")
    ef.observe(update_sam, names="value")
    em.observe(update_sam, names="value")
    ec.observe(update_queries, names="value")
    update_frames()
    explorer = panel([es, et, ef, em, ec, eq, eg, diagnostic, opacity],
                     lambda: render_image(saved, ef.value, model=em.value, condition=ec.value,
                                          query_id=eq.value, concept=eg.value, opacity=opacity.value,
                                          diagnostic=None if diagnostic.value == "scored" else diagnostic.value),
                     lambda: image_diagnostics(saved, ef.value, em.value, ec.value))

    ss = dropdown("Source", sources)
    st = dropdown("Split", splits, "test")
    sc = dropdown("Condition", sorted({r["condition_key"] for r in saved.rows("segmentation")}), all_value=True)
    sam = panel([ss, st, sc], lambda: render_segmentation(saved, source=ss.value, split=st.value, condition=sc.value),
                lambda: {"rows": saved.rows("segmentation", source=ss.value, split=st.value, condition_key=sc.value),
                         "per_concept": saved.rows("segmentation", "per_concept", source=ss.value, split=st.value, condition_key=sc.value)})

    profiles = w.SelectMultiple(description="Profiles", options=[(" / ".join(k), k) for k in saved.profiles],
                                value=(), rows=min(8, max(2, len(saved.profiles))), layout=w.Layout(width="95%"))
    deployment = panel([profiles], lambda: render_deployment(saved, profiles.value),
                       lambda: [saved.profiles[k] for k in profiles.value])
    tabs = w.Tab(children=[discovery, explorer, sam, deployment])
    for i, title in enumerate(["VLM discovery", "Image explorer", "SAM handoff", "Deployment"]):
        tabs.set_title(i, title)
    banner = w.HTML("<b style='color:#a33a13'>SYNTHETIC DATA — SOFTWARE CHECK ONLY</b>" if saved.fixture else "<b>Saved hazard evidence</b>")
    return w.VBox([banner, details(saved.status()), tabs])
