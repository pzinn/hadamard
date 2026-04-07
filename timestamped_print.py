import builtins
import datetime
import sys


_print_at_line_start = True


def print(*args, **kwargs):
    global _print_at_line_start
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", sys.stdout)
    if _print_at_line_start:
        stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        builtins.print(stamp, end="", file=file)
    builtins.print(*args, **kwargs)
    _print_at_line_start = end.endswith("\n")
