import torch.cuda.nvtx as nvtx
from minisgl.utils.torch_utils import nvtx_annotate, nvtx_pause_current_ranges


def test_nvtx_pause_current_ranges_reopens_annotated_ranges(monkeypatch):
    calls = []

    monkeypatch.setattr(nvtx, "range_push", lambda name: calls.append(("push", name)))
    monkeypatch.setattr(nvtx, "range_pop", lambda: calls.append(("pop", None)))

    class Op:
        @nvtx_annotate("Layer_{}", layer_id_field="layer_id")
        def forward(self):
            with nvtx_pause_current_ranges():
                calls.append(("wait", None))

        layer_id = 7

    Op().forward()

    assert calls == [
        ("push", "Layer_7"),
        ("pop", None),
        ("wait", None),
        ("push", "Layer_7"),
        ("pop", None),
    ]
