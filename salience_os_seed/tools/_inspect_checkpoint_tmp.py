import argparse
import sys
from pathlib import Path

import torch


def _load_session_module(root: Path):
    import importlib.util
    import sys as _sys

    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    session_path = root / "conversation" / "session.py"
    spec = importlib.util.spec_from_file_location("conversation.session", session_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load session module at {session_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_payload(path_str: str, *, check_nan: bool = False) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"missing: {path}")
        return -1
    payload = torch.load(path, map_location="cpu")
    step = int(payload.get("step", 0))
    print(f"checkpoint: {path}")
    print(f"step: {step}")
    vocab = payload.get("vocab", {})
    if isinstance(vocab, dict):
        print(f"vocab tokens: {len(vocab.get('tokens', []))}")
    model_state = payload.get("model")
    if isinstance(model_state, dict):
        embed_weight = model_state.get("embed.weight")
        output_weight = model_state.get("output.weight")
        if embed_weight is not None:
            print(f"embed.weight shape: {tuple(embed_weight.shape)}")
        if output_weight is not None:
            print(f"output.weight shape: {tuple(output_weight.shape)}")
        if check_nan:
            nan_tensors: list[str] = []
            inf_tensors: list[str] = []
            for name, tensor in model_state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if torch.isnan(tensor).any():
                    nan_tensors.append(name)
                if torch.isinf(tensor).any():
                    inf_tensors.append(name)
            if nan_tensors:
                print("model tensors containing NaN:")
                for name in nan_tensors:
                    print(f"  {name}")
            if inf_tensors:
                print("model tensors containing Inf:")
                for name in inf_tensors:
                    print(f"  {name}")
    optimizer = payload.get("optimizer")
    if optimizer:
        print("optimizer keys:", sorted(optimizer.keys()))
        states = optimizer.get("state", {})
        step_shapes: dict[str, int] = {}
        for value in states.values():
            step = value.get("step") if isinstance(value, dict) else None
            if step is None:
                continue
            if isinstance(step, torch.Tensor):
                shape = tuple(step.shape)
            else:
                shape = ()
            shape_key = str(shape)
            step_shapes[shape_key] = step_shapes.get(shape_key, 0) + 1
        if step_shapes:
            print("optimizer step shape counts:")
            for shape_key, count in sorted(step_shapes.items()):
                print(f"  {shape_key}: {count}")
    return step


def inspect_session(path_str: str, *, prompt: str | None, max_tokens: int) -> None:
    root = Path(__file__).resolve().parent.parent
    module = _load_session_module(root)
    cfg = module.ConversationConfig(
        lm=module.TrainingConfig(checkpoint_path=path_str),
        learning_enabled=False,
        archive_checkpoint_on_start=False,
        response_tokens=max_tokens,
    )
    session = module.ConversationSession(config=cfg)
    print(f"session proto_lm.step: {session.proto_lm.step}")
    print(f"session vocab size: {session.proto_lm.vocab.size()}")
    if prompt is not None:
        sample = session.proto_lm.sample(prompt, max_tokens=max_tokens)
        print("sample:")
        print(sample)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect proto LM checkpoint and optionally sample text")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--session",
        action="store_true",
        help="Instantiate ConversationSession to verify load",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample text from the proto LM (implies --session)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt prefix when sampling (requires --sample)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="Maximum tokens to sample",
    )
    parser.add_argument(
        "--check-nan",
        action="store_true",
        help="Scan model tensors for NaN/Inf values",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inspect_payload(args.checkpoint, check_nan=args.check_nan)
    if args.session or args.sample:
        prompt = args.prompt if args.sample else None
        try:
            inspect_session(args.checkpoint, prompt=prompt, max_tokens=args.tokens)
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
