import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _read_command(args: argparse.Namespace) -> tuple[object, bool]:
    sources = [bool(args.cmd), bool(args.cmd_file), bool(args.command)]
    if sum(1 for flag in sources if flag) != 1:
        raise ValueError("specify exactly one of --cmd, --cmd-file, or a trailing command")
    if args.cmd:
        return args.cmd, True
    if args.cmd_file:
        command_text = Path(args.cmd_file).read_text(encoding="utf-8").strip()
        if not command_text:
            raise ValueError("command file is empty")
        return command_text, True
    command_parts = list(args.command)
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]
    if not command_parts:
        raise ValueError("no command provided")
    if args.shell:
        return " ".join(command_parts), True
    return command_parts, False


def _stream_process(proc: subprocess.Popen, timeout: float | None) -> int:
    deadline = time.time() + timeout if timeout else None
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
            if line == "" and proc.poll() is not None:
                break
            if deadline and time.time() > deadline:
                proc.kill()
                print("\n[command_runner] timeout exceeded", file=sys.stderr, flush=True)
                return 124
        return proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a subprocess with streamed output.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute; precede with -- to terminate parser options.")
    parser.add_argument("--cmd", help="Command string to execute.")
    parser.add_argument("--cmd-file", help="Read command string from file.")
    parser.add_argument("--cwd", help="Working directory for the command.")
    parser.add_argument("--timeout", type=float, help="Timeout in seconds.")
    parser.add_argument("--shell", action="store_true", help="Force shell execution for trailing command arguments.")
    args = parser.parse_args()

    try:
        command, force_shell = _read_command(args)
    except Exception as exc:
        print(f"[command_runner] {exc}", file=sys.stderr)
        return 2

    use_shell = force_shell or args.shell
    if isinstance(command, str) and not use_shell:
        command = shlex.split(command)
    print(f"[command_runner] running: {command}")

    try:
        proc = subprocess.Popen(
            command,
            cwd=args.cwd,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        print(f"[command_runner] failed to start command: {exc}", file=sys.stderr)
        return 127

    try:
        returncode = _stream_process(proc, args.timeout)
    except KeyboardInterrupt:
        proc.kill()
        print("\n[command_runner] interrupted", file=sys.stderr, flush=True)
        return 130

    print(f"[command_runner] exit code: {returncode}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
