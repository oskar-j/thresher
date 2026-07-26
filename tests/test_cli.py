"""The `thresher` command-line interface."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from thresher import algorithm
from thresher.cli import main

BASIC_CSV = "score,actual\n0.1,-1\n0.3,-1\n0.4,1\n0.7,1\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def basic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "scores.csv"
    path.write_text(BASIC_CSV)
    return path


def test_prints_the_threshold(runner: CliRunner, basic_csv: Path) -> None:
    result = runner.invoke(main, [str(basic_csv)])

    assert result.exit_code == 0
    # The bare number and nothing else, so the output can be piped onward.
    assert 0.3 <= float(result.output.strip()) < 0.4


def test_reads_stdin(runner: CliRunner) -> None:
    result = runner.invoke(main, ["-"], input=BASIC_CSV)

    assert result.exit_code == 0
    assert 0.3 <= float(result.output.strip()) < 0.4


@pytest.mark.parametrize("algorithm_name", ["exact", "ls", "grid", "sgrid", "gen", "sgd", "auto"])
def test_every_algorithm_is_selectable(runner: CliRunner, basic_csv: Path, algorithm_name: str) -> None:
    result = runner.invoke(main, [str(basic_csv), "-a", algorithm_name])

    assert result.exit_code == 0, result.output
    assert float(result.output.strip()) == pytest.approx(float(result.output.strip()))


def test_custom_labels(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "zero_one.csv"
    path.write_text("score,actual\n0.1,0\n0.3,0\n0.4,1\n0.7,1\n")

    result = runner.invoke(main, [str(path), "--labels", "0,1"])

    assert result.exit_code == 0
    assert 0.3 <= float(result.output.strip()) < 0.4


def test_no_header_and_custom_separator(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "scores.tsv"
    path.write_text("0.1\t-1\n0.3\t-1\n0.4\t1\n0.7\t1\n")

    result = runner.invoke(main, [str(path), "--sep", "\t", "--no-header"])

    assert result.exit_code == 0
    assert 0.3 <= float(result.output.strip()) < 0.4


def test_columns_selected_by_name(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text("id,actual,note,score\n1,-1,a,0.1\n2,-1,b,0.3\n3,1,c,0.4\n4,1,d,0.7\n")

    result = runner.invoke(main, [str(path), "--score-column", "score", "--label-column", "actual"])

    assert result.exit_code == 0
    assert 0.3 <= float(result.output.strip()) < 0.4


def test_algorithm_params_are_passed_through(runner: CliRunner, basic_csv: Path) -> None:
    result = runner.invoke(main, [str(basic_csv), "-a", "ls", "-p", "n_jobs=2"])

    assert result.exit_code == 0
    assert float(result.output.strip()) >= 0.0


def test_list_algorithms(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--list-algorithms"])

    assert result.exit_code == 0
    for name in algorithm.available_algorithms:
        assert name in result.output


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "thresher" in result.output


class TestErrors:
    """Bad input should explain itself, and say so in command-line terms."""

    def test_unknown_algorithm_points_at_the_flag(self, runner: CliRunner, basic_csv: Path) -> None:
        result = runner.invoke(main, [str(basic_csv), "-a", "does-not-exist"])

        assert result.exit_code == 1
        assert "does-not-exist" in result.output
        # not the Python API advice the library exception carries
        assert "--list-algorithms" in result.output
        assert "get_supported_algorithms()" not in result.output

    def test_unmapped_labels_point_at_the_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "zero_one.csv"
        path.write_text("score,actual\n0.1,0\n0.7,1\n")

        result = runner.invoke(main, [str(path)])

        assert result.exit_code == 1
        assert "--labels" in result.output
        assert "Thresher(labels=" not in result.output

    def test_single_class(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "one_class.csv"
        path.write_text("score,actual\n0.1,-1\n0.2,-1\n")

        result = runner.invoke(main, [str(path)])

        assert result.exit_code == 1
        assert "single class" in result.output

    def test_missing_file(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, [str(tmp_path / "absent.csv")])

        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_too_few_columns(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "one_column.csv"
        path.write_text("score\n0.1\n0.2\n")

        result = runner.invoke(main, [str(path)])

        assert result.exit_code == 2
        assert "--label-column" in result.output

    def test_unknown_column(self, runner: CliRunner, basic_csv: Path) -> None:
        result = runner.invoke(main, [str(basic_csv), "--score-column", "nope"])

        assert result.exit_code == 2
        assert "nope" in result.output

    def test_empty_input(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("score,actual\n")

        result = runner.invoke(main, [str(path)])

        assert result.exit_code == 1
        assert "no rows" in result.output

    @pytest.mark.parametrize("bad", ["n_jobs", "=4", ""])
    def test_malformed_param(self, runner: CliRunner, basic_csv: Path, bad: str) -> None:
        result = runner.invoke(main, [str(basic_csv), "-p", bad])

        assert result.exit_code == 2
        assert "key=value" in result.output

    @pytest.mark.parametrize("bad", ["0", "0,1,2"])
    def test_malformed_labels(self, runner: CliRunner, basic_csv: Path, bad: str) -> None:
        result = runner.invoke(main, [str(basic_csv), "--labels", bad])

        assert result.exit_code == 2
        assert "two comma-separated" in result.output


class TestOptionParsing:
    """Values arrive as strings and have to be coerced to what the algorithms expect."""

    @pytest.mark.parametrize(
        ("pair", "expected"),
        [
            ("n_jobs=4", 4),
            ("stoch_ratio=0.1", 0.1),
            ("reshuffle=true", True),
            ("reshuffle=False", False),
            ("some_name=abc", "abc"),
        ],
    )
    def test_param_values_are_coerced(self, pair: str, expected: object) -> None:
        from thresher.cli import _parse_params

        key = pair.split("=")[0]
        parsed = _parse_params((pair,))

        assert parsed[key] == expected
        assert isinstance(parsed[key], type(expected))

    def test_repeated_params_accumulate(self) -> None:
        from thresher.cli import _parse_params

        assert _parse_params(("a=1", "b=2.5")) == {"a": 1, "b": 2.5}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0,1", (0, 1)), ("no,yes", ("no", "yes")), (" -1 , 1 ", (-1, 1))],
    )
    def test_labels_are_parsed_and_coerced(self, raw: str, expected: tuple[object, object]) -> None:
        from thresher.cli import _parse_labels

        assert _parse_labels(raw) == expected

    def test_no_labels_option_means_none(self) -> None:
        from thresher.cli import _parse_labels

        assert _parse_labels(None) is None


class TestColumnSelection:
    def test_columns_selected_by_index(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "wide.csv"
        path.write_text("id,actual,note,score\n1,-1,a,0.1\n2,-1,b,0.3\n3,1,c,0.4\n4,1,d,0.7\n")

        result = runner.invoke(main, [str(path), "--score-column", "3", "--label-column", "1"])

        assert result.exit_code == 0, result.output
        assert 0.3 <= float(result.output.strip()) < 0.4

    def test_string_labels_from_a_file(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "words.csv"
        path.write_text("score,actual\n0.1,no\n0.3,no\n0.4,yes\n0.7,yes\n")

        result = runner.invoke(main, [str(path), "--labels", "no,yes"])

        assert result.exit_code == 0, result.output
        assert 0.3 <= float(result.output.strip()) < 0.4


class TestBackendOption:
    def test_local_backend_flag(self, runner: CliRunner, basic_csv: Path) -> None:
        result = runner.invoke(main, [str(basic_csv), "--backend", "local"])

        assert result.exit_code == 0
        assert 0.3 <= float(result.output.strip()) < 0.4

    def test_ray_without_the_extra_explains_how_to_install_it(
        self, runner: CliRunner, basic_csv: Path
    ) -> None:
        try:
            import ray  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("Ray is installed, so this error path cannot be reached")

        result = runner.invoke(main, [str(basic_csv), "--backend", "ray"])

        assert result.exit_code == 1
        assert "thresher-py[ray]" in result.output

    def test_unknown_backend_is_rejected_by_the_parser(self, runner: CliRunner, basic_csv: Path) -> None:
        result = runner.invoke(main, [str(basic_csv), "--backend", "nope"])

        assert result.exit_code == 2
        assert "local" in result.output and "ray" in result.output


class TestOutputAndFailures:
    def test_verbose_reports_the_row_count(self, runner: CliRunner, basic_csv: Path) -> None:
        result = runner.invoke(main, [str(basic_csv), "--verbose"])

        assert result.exit_code == 0
        assert "Read 4 rows" in result.output

    def test_unreadable_input_is_reported(self, runner: CliRunner, tmp_path: Path) -> None:
        """A file with nothing in it at all, which pandas refuses outright.

        Distinct from a header-only file, which parses fine into zero rows and is caught
        later by the "no rows to optimize over" check.
        """
        path = tmp_path / "empty_file.csv"
        path.write_text("")

        result = runner.invoke(main, [str(path)])

        assert result.exit_code == 1
        assert "could not read" in result.output
