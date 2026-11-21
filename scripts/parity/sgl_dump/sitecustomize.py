"""Per-visit residual dump for SGLang's Inkling2 loop model (debug only).

Mounted into the container and put on PYTHONPATH; does nothing unless INK_DUMP_DIR is set.
After each decoder-layer call in a prefill (extend) forward, saves state.partial (the running
block-residual, same quantity HF's decoder layers return) keyed by layer visit:
  visit = layer for 0..5; 6 + pass*12 + (layer-6) for looped layers; 42 + (layer-18) for 18..23.
One file per forward: $INK_DUMP_DIR/fwd_<n>.pt = {visit: [T, H] float32}. Skipped during CUDA graph capture.
"""
import os

DUMP = os.environ.get("INK_DUMP_DIR")

if DUMP:
    import builtins
    import sys

    _orig_import = builtins.__import__
    _state = {"patched": False, "n": 0, "cur": None}

    def _patch(mod):
        import torch

        cls = mod.Inkling2CausalLLM
        orig_init, orig_fwd = cls.__init__, cls.forward

        def init(self, config, *a, **k):
            orig_init(self, config, *a, **k)
            core = config.loop_core_layers

            def make_hook(layer_id):
                def hook(_m, args, kwargs, _out):
                    if _state["cur"] is None or torch.cuda.is_current_stream_capturing():
                        return
                    p = kwargs.get("pass_idx", 0)
                    if layer_id < core.start:
                        visit = layer_id
                    elif layer_id < core.stop:
                        visit = core.start + p * len(core) + (layer_id - core.start)
                    else:
                        visit = core.start + 3 * len(core) + (layer_id - core.stop)
                    _state["cur"][visit] = args[0].partial.detach().float().cpu()
                return hook

            for i, layer in enumerate(self.layers):
                layer.register_forward_hook(make_hook(i), with_kwargs=True)

        def forward(self, input_ids, positions, forward_batch, input_embeds=None):
            fm = forward_batch.forward_mode
            on = fm.is_extend() and not torch.cuda.is_current_stream_capturing()
            if on:
                _state["cur"] = {}
            out = orig_fwd(self, input_ids, positions, forward_batch, input_embeds)
            if on:
                os.makedirs(DUMP, exist_ok=True)
                ids = input_ids if input_ids is not None else forward_batch.input_ids
                rec = {"visits": _state["cur"], "input_ids": ids.detach().cpu(),
                       "positions": positions.detach().cpu()}
                torch.save(rec, f"{DUMP}/fwd_{os.getpid()}_{_state['n']}.pt")
                _state["n"] += 1
                _state["cur"] = None
            return out

        cls.__init__, cls.forward = init, forward

    def _import(name, *a, **k):
        m = _orig_import(name, *a, **k)
        if not _state["patched"] and "sglang.srt.models.inkling2" in sys.modules:
            mod = sys.modules["sglang.srt.models.inkling2"]
            if hasattr(mod, "Inkling2CausalLLM"):
                _state["patched"] = True
                _patch(mod)
        return m

    builtins.__import__ = _import
