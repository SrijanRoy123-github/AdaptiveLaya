from tools.registry import TOOL_NAMES, execute

SAMPLE = "def add(a, b):\n    return a + b\n"
TEST_PASS = "def test_add():\n    from sample import add\n    assert add(1, 2) == 3\n"
TEST_FAIL = "def test_add():\n    from sample import add\n    assert add(1, 2) == 4\n"


def make_repo(tmp_path, test_body):
    (tmp_path / "sample.py").write_text(SAMPLE)
    (tmp_path / "test_sample.py").write_text(test_body)
    return tmp_path


def test_tool_names():
    assert TOOL_NAMES == ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]


def test_run_tests_pass(tmp_path):
    repo = make_repo(tmp_path, TEST_PASS)
    result = execute("RUN_TESTS", {"target": "test_sample.py", "timeout_s": 60}, str(repo))
    assert result["ok"] is True
    assert "passed" in result["output"]


def test_run_tests_fail(tmp_path):
    repo = make_repo(tmp_path, TEST_FAIL)
    result = execute("RUN_TESTS", {"target": "test_sample.py", "timeout_s": 60}, str(repo))
    assert result["ok"] is False


def test_run_tests_timeout(tmp_path):
    (tmp_path / "test_slow.py").write_text("import time\ndef test_slow():\n    time.sleep(30)\n")
    result = execute("RUN_TESTS", {"target": "test_slow.py", "timeout_s": 2}, str(tmp_path))
    assert result["ok"] is False
    assert result.get("timeout") is True


def test_lint(tmp_path):
    (tmp_path / "bad.py").write_text("import os\n")
    result = execute("LINT", {"path": "bad.py"}, str(tmp_path))
    assert "output" in result


def test_read_context(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n" * 10)
    result = execute("READ_CONTEXT", {"path": "a.py"}, str(tmp_path))
    assert result["ok"] is True
    assert "x = 1" in result["output"]


def test_read_context_missing(tmp_path):
    result = execute("READ_CONTEXT", {"path": "nope.py"}, str(tmp_path))
    assert result["ok"] is False


def test_search_repo(tmp_path):
    (tmp_path / "lib.py").write_text("def find_me():\n    pass\n")
    result = execute("SEARCH_REPOSITORY", {"pattern": "find_me"}, str(tmp_path))
    assert result["ok"] is True
    assert "lib.py" in result["output"]


def test_unknown_tool(tmp_path):
    result = execute("SHELL", {}, str(tmp_path))
    assert result["ok"] is False


def test_read_context_escape(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = execute("READ_CONTEXT", {"path": "../esc.py"}, str(tmp_path))
    assert result["ok"] is False
    result2 = execute("READ_CONTEXT", {"path": "a.py"}, str(tmp_path))
    assert result2["ok"] is True

