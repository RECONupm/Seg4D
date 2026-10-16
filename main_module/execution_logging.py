"""Shared execution logs for Seg4D console runtimes."""

from datetime import datetime
import os
import platform
import subprocess
import sys
import traceback


_SEPARATOR = "=" * 78


def _command_text(command):
    if isinstance(command, str):
        return command
    return subprocess.list2cmdline([str(argument) for argument in command])


def start_execution_log(
    output_directory,
    filename,
    operation,
    command=None,
    working_directory=None,
    details=None,
):
    """Create a new UTF-8 log and write reproducibility information."""
    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, filename)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"{_SEPARATOR}\n")
        log_file.write(f"Seg4D execution log: {operation}\n")
        log_file.write(f"Started: {datetime.now().astimezone().isoformat()}\n")
        log_file.write(f"Platform: {platform.platform()}\n")
        log_file.write(f"Seg4D Python: {sys.executable}\n")
        log_file.write(
            "Working directory: "
            f"{working_directory or os.getcwd()}\n"
        )
        if command is not None:
            log_file.write(f"Command: {_command_text(command)}\n")
        if details:
            for key, value in details.items():
                log_file.write(f"{key}: {value}\n")
        log_file.write(f"{_SEPARATOR}\n")
        log_file.write("Console output (stdout + stderr):\n\n")
    return log_path


def append_execution_log(log_path, text):
    """Append diagnostic text to an existing execution log."""
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(str(text))
        if not str(text).endswith("\n"):
            log_file.write("\n")


def append_execution_result(log_path, return_code):
    """Record the final process status."""
    append_execution_log(
        log_path,
        (
            f"\n{_SEPARATOR}\n"
            f"Finished: {datetime.now().astimezone().isoformat()}\n"
            f"Exit code: {return_code}\n"
            f"{_SEPARATOR}\n"
        ),
    )


def append_execution_exception(log_path, context, error=None):
    """Append the active Python traceback, including failures outside an EXE."""
    formatted_traceback = traceback.format_exc()
    if formatted_traceback.strip() == "NoneType: None":
        formatted_traceback = repr(error) if error is not None else "Unknown error"
    append_execution_log(
        log_path,
        (
            f"\n{_SEPARATOR}\n"
            f"Python exception while {context}:\n"
            f"{formatted_traceback.rstrip()}\n"
            f"{_SEPARATOR}\n"
        ),
    )


def read_execution_log_tail(log_path, line_count=30):
    """Return the last log lines for a concise error dialog."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            return "".join(log_file.readlines()[-line_count:]).strip()
    except OSError:
        return ""


def execution_failure_message(summary, log_path, return_code=None):
    """Build an error dialog message containing the real console failure."""
    message = summary
    if return_code is not None:
        message += f"\nExit code: {return_code}"
    details = read_execution_log_tail(log_path)
    if details:
        message += f"\n\nCause reported by the process:\n{details}"
    message += f"\n\nFull detailed log:\n{log_path}"
    return message


def run_logged_command(
    command,
    output_directory,
    filename,
    operation,
    working_directory=None,
    creationflags=0,
    details=None,
):
    """Run a console command and capture both output streams in a detailed log."""
    log_path = start_execution_log(
        output_directory,
        filename,
        operation,
        command=command,
        working_directory=working_directory,
        details=details,
    )
    try:
        with open(
            log_path, "a", encoding="utf-8", errors="replace"
        ) as log_file:
            completed = subprocess.run(
                command,
                cwd=working_directory,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                creationflags=creationflags,
            )
    except Exception as error:
        append_execution_exception(log_path, "starting the console process", error)
        return subprocess.CompletedProcess(command, -1), log_path
    append_execution_result(log_path, completed.returncode)
    return completed, log_path
