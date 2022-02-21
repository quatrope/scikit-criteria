#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tool to check if each python module has a corresponding API docs."""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect
import pathlib

import attr

import typer

# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "0.1"

# =============================================================================
# FUNCTIONS
# =============================================================================


def check_apidoc_structure(apidoc_dir, reference_dir):

    apidoc_dir = pathlib.Path(apidoc_dir)
    reference_dir = pathlib.Path(reference_dir)

    if not apidoc_dir.exists():
        raise OSError(f"'{apidoc_dir}' do no exist")
    if not reference_dir.exists():
        raise OSError(f"'{reference_dir}' do no exist")

    reference = list(reference_dir.glob("**/*.py"))

    result = {}
    for ref in reference:

        # essentially we remove the parent dir
        *dirs, ref_name = ref.relative_to(reference_dir).parts

        if ref_name == "__init__.py":
            ref_name = "index.py"

        search_dir = apidoc_dir
        for subdir in dirs:
            search_dir /= subdir

        search = search_dir / f"{ref_name[:-3]}.rst"

        result[str(ref)] = (str(search), search.exists())

    return result


# =============================================================================
# CLI
# =============================================================================


@attr.s(frozen=True)
class CLI:
    """Check if the structure of API doc directory is equivalent to those of
    the project.

    """

    footnotes = "\n".join(
        [
            "This software is under the BSD 3-Clause License.",
            "Copyright (c) 2021, Juan Cabral.",
            "For bug reporting or other instructions please check:"
            " https://github.com/quatrope/scikit-criteria",
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
            ..., help="Path to the api-doc structure."
        ),
        reference_dir: str = typer.Option(
            ..., help="Path to the reference structure."
        ),
        verbose: bool = typer.Option(
            default=False, help="Show all the result"
        ),
    ):
        """Check if the structure of test directory is equivalent to those
        of the project.

        """
        try:
            check_result = check_apidoc_structure(test_dir, reference_dir)
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
    """Run the checkapidocdir.py cli interface."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
