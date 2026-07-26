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


@pytest.mark.parametrize("algorithm_name", ["ls", "grid", "sgrid", "gen", "sgd", "auto"])
def test_every_algorithm_is_selectable(
    runner: CliRunner, basic_csv: Path, algorithm_name: str
) -> None:
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

    result = runner.invoke(
        main, [str(path), "--score-column", "score", "--label-column", "actual"]
    )

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
