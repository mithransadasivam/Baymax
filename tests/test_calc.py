import pytest

from baymax.calc import CALC_NOT_CHECKED, extract_expression, lookup


class TestExtractExpression:
    @pytest.mark.parametrize(
        ("question", "expression"),
        [
            ("What is 47 * 89?", "47 * 89"),
            ("what's 12 + 7", "12 + 7"),
            ("Calculate 100 / 4", "100 / 4"),
            ("What is 47 times 89?", "47  *  89"),
            ("What's 15 plus 6?", "15  +  6"),
            ("What is 9 minus 4?", "9  -  4"),
            ("What is 20 divided by 5?", "20  /  5"),
            ("What is 2 to the power of 8?", "2  **  8"),
            ("What's 15 percent of 200?", "(15/100)*200"),
            ("(12 + 5) / 3", "(12 + 5) / 3"),
        ],
    )
    def test_recognizes_pure_arithmetic_questions(self, question, expression):
        assert extract_expression(question) == expression

    @pytest.mark.parametrize(
        "question",
        [
            "What is the capital of Australia?",
            "What's the weather today?",
            "What's on my schedule?",
            "Why does a Wheatstone bridge balance at zero?",
            "What's the plus side of moving to Chennai?",  # contains "plus" but isn't arithmetic
        ],
    )
    def test_leaves_non_arithmetic_questions_alone(self, question):
        assert extract_expression(question) is None


class TestLookup:
    def test_computes_the_exact_answer(self):
        check = lookup("What is 47 * 89?")

        assert "4183" in check.results
        assert "exact" in check.results.lower()
        assert "calculated" in check.record.lower()

    def test_a_non_arithmetic_question_is_not_checked(self):
        assert lookup("What is the capital of Australia?") == CALC_NOT_CHECKED

    def test_percent_of_is_computed_correctly(self):
        check = lookup("What's 15 percent of 200?")

        assert "30" in check.results

    def test_division_by_zero_is_reported_rather_than_crashing(self):
        check = lookup("What is 5 / 0?")

        assert "not sure" in check.results.lower() or "didn't come out" in check.results.lower()

    def test_only_numeric_literals_and_arithmetic_operators_are_ever_evaluated(self):
        # __import__ or any other name would never pass the ARITHMETIC_ONLY gate to begin with,
        # but this pins the deeper guarantee: the evaluator itself has no path to run anything
        # other than +, -, *, /, **, % on plain numbers.
        from baymax.calc import _safe_eval
        import ast

        with pytest.raises(ValueError):
            _safe_eval(ast.parse("__import__('os')", mode="eval").body)
