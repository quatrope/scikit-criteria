import inspect
import pathlib
from typing import Text

import attr

import typer

# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "0.1"

# =============================================================================
# FUNCTIONS
# =============================================================================


def check_test_structure(test_dir, reference_dir):

    test_dir = pathlib.Path(test_dir)
    reference_dir = pathlib.Path(reference_dir)

    if not test_dir.exists():
        raise OSError(f"'{test_dir}' do no exist")
    if not reference_dir.exists():
        raise OSError(f"'{reference_dir}' do no exist")

    reference = list(reference_dir.glob("**/*.py"))

    result = {}
    for ref in reference:
        if ref.name.startswith("_"):
            continue

        # essentially we remove the parent dir
        *dirs, ref_name = ref.relative_to(reference_dir).parts

        search_dir = test_dir
        for subdir in dirs:
            search_dir /= subdir

        search = search_dir / f"test_{ref_name}"

        result[str(ref)] = (str(search), search.exists())

    return result


# =============================================================================
# CLI
# =============================================================================


@attr.s(frozen=True)
class CLI:
    """Check if the structure of test directory is equivalent to those of the
    project.

    """

    footnotes = "\n".join(
        [
            "This software is under the BSD 3-Clause License.",
            "Copyright (c) 2021, Juan Cabral.",
            "For bug reporting or other instructions please check:"
            " https://github.com/quatrope/scikit-criteria-next",
        ]
    )

    run = attr.ib(init=False)

    @run.default
    def _set_run_default(self):
        app = typer.Typer()
        for k in dir(self):
            if k.startswith("_"):
                continue
            v = getattr(self, k)
            if inspect.ismethod(v):
                decorator = app.command()
                decorator(v)
        return app

    def version(self):
        """Print checktestdir.py version."""
        typer.echo(f"{__file__ } v.{VERSION}")

    def check(
        self,
        test_dir: str = typer.Argument(
            ..., help="Path to the test structure."
        ),
        reference_dir: str = typer.Option(
            ..., help="Path to the reference structure."
        ),
        verbose: bool = typer.Option(
            default=False, help="Show all the result"
        ),
    ):
        try:
            check_result = check_test_structure(test_dir, reference_dir)
        except Exception as err:
            typer.echo(typer.style(str(err), fg=typer.colors.RED))
            raise typer.Exit(code=1)

        all_tests_exists = True
        for ref, test_result in check_result.items():

            test, test_exists = test_result

            if test_exists:
                fg = typer.colors.GREEN
                status = ""
            else:
                all_tests_exists = False
                fg = typer.colors.RED
                status = typer.style("[NOT FOUND]", fg=typer.colors.YELLOW)

            if verbose or not test_exists:
                msg = f"{ref} -> {test} {status}"
                typer.echo(typer.style(msg, fg=fg))

        if all_tests_exists:
            final_fg = typer.colors.GREEN
            final_status = "Test structure ok!"
            exit_code = 0
        else:
            final_fg = typer.colors.RED
            final_status = "Structure not equivalent!"
            exit_code = 1

        typer.echo("-------------------------------------")
        typer.echo(typer.style(final_status, fg=final_fg))
        raise typer.Exit(code=exit_code)


def main():
    """Run the checktestdir.py cli interface."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
