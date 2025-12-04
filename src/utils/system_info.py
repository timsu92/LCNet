from __future__ import annotations

from typing import Callable, Tuple

import torch


def auto_batch_size(
    device: torch.device,
    build_model: Callable[[], torch.nn.Module],
    sample_shape: Tuple[int, int, int, int],
    max_batch: int = 1024,
    base_batch: int = 64,
    max_trials_between: int = 3,
) -> int:
    """Heuristically search for a suitable batch size.

    Strategy:
      1) Try exponentially increasing sizes: base_batch, base_batch*2, ... up to max_batch
         or until OOM occurs.
      2) When the first OOM is hit, binary-search-ish between last_ok and failed size,
         but limit to at most `max_trials_between` extra attempts.

    This is a rough heuristic meant to be called once before training.
    It assumes the dominant memory usage comes from holding a single batch
    and doing one forward/backward pass.
    """

    if device.type != "cuda":
        # For CPU, just return a conservative value based on base_batch.
        return base_batch

    last_ok = 0
    current = base_batch

    def can_run(batch_size: int) -> bool:
        model = build_model().to(device)
        # Simulate optimizer memory (AdamW has 2 states per param, roughly doubling model size requirement)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn((batch_size,) + sample_shape[1:], device=device)
        try:
            # Removed autocast to match train.py (full precision)
            out = model(x)
            loss = out.mean()
            loss.backward()
            optimizer.step()  # Triggers optimizer state initialization
            optimizer.zero_grad()

            # If we reached here, batch_size is feasible.
            ok = True
        except RuntimeError as e:  # likely OOM
            if "out of memory" in str(e).lower():
                ok = False
            else:
                raise
        finally:
            # Clean up
            del model, x
            try:
                del loss  # type: ignore[name-defined]
            except UnboundLocalError:
                pass
            try:
                del out  # type: ignore[name-defined]
            except UnboundLocalError:
                pass

            torch.cuda.empty_cache()
        return ok

    # 1) Exponential growth until failure or max_batch
    while current <= max_batch:
        if can_run(current):
            last_ok = current
            current *= 2
        else:
            break

    if last_ok == 0:
        # Even the smallest base batch doesn't fit; fall back to 1.
        return 1

    # If we never failed up to max_batch, just return last_ok.
    if current > max_batch:
        return last_ok

    # 2) Refine between last_ok and failed (current), with limited probes.
    low = last_ok
    high = current
    trials = 0
    best = last_ok

    while trials < max_trials_between and high - low > 1:
        mid = (low + high) // 2
        if can_run(mid):
            best = mid
            low = mid
        else:
            high = mid
        trials += 1

    # Apply safety margin (93%) to avoid OOM due to fragmentation
    return int(best * 0.93)
