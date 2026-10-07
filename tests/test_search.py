import pytest

from baymax.search import NOT_CHECKED, lookup, search, search_query


@pytest.fixture
def fake_ddgs(monkeypatch):
    """Replaces DuckDuckGo with a stub, so the tests never touch the network."""
    state = {"results": [], "error": None, "calls": [], "backends": []}

    class FakeDDGS:
        def __init__(self, **kwargs) -> None:
            pass

        def text(self, query, max_results=None, **kwargs):
            state["calls"].append((query, max_results))
            state["backends"].append(kwargs.get("backend"))
            if state["error"]:
                raise state["error"]
            return state["results"]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    return state


class TestSearchQuery:
    @pytest.mark.parametrize(
        "question",
        [
            "What's the weather in Las Vegas today?",
            "Any news about the election?",
            "What's the current price of petrol?",
            "Who won the game last night?",
            "What are the latest headlines?",
            # Casual weather phrasings, which the first version of the gate let through unsearched.
            "What is the temp in Chennai?",
            "How hot is it in Chennai?",
            "Is it raining in Chennai?",
            "Will it rain in London?",
            "What time is it in Tokyo?",
        ],
    )
    def test_questions_about_live_information_are_searched(self, question):
        assert search_query(question) == question

    @pytest.mark.parametrize(
        "question",
        [
            # Each of these got a vague or invented answer from llama3.2:3b in testing.
            "Why does a Wheatstone bridge go to zero output when balanced?",
            "Is it safe to weld a fuel tank that has been drained but not purged of vapor?",
            "What is the second moment of area formula for a solid circular cross-section?",
            "What causes cavitation in a centrifugal pump?",
            "What's the difference between yield strength and ultimate tensile strength?",
            "How do you calculate torque from horsepower and RPM?",
            "Derive the relationship between force and acceleration.",
            "How does a jet engine work?",
            "What is the formula for kinetic energy?",
        ],
    )
    def test_technical_questions_the_model_tends_to_get_vague_or_wrong_are_searched(self, question):
        assert search_query(question) == question

    @pytest.mark.parametrize(
        "question",
        [
            "What is the capital of Australia?",
            "Explain how a jet engine works.",
            "What is 17 times 23?",
            "Who wrote Brave New World?",
            "Why is the sky blue?",
            # "Why does" and "is it safe to" alone would also match these; domain vocabulary is
            # what keeps everyday questions like these from triggering a needless search.
            "Why does my dog bark at the mailman?",
            "Is it safe to eat raw cookie dough?",
            "What's the difference between a crocodile and an alligator?",
        ],
    )
    def test_general_knowledge_never_touches_the_network(self, question):
        assert search_query(question) is None

    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            ("Baymax, search for cheap flights to Tokyo", "cheap flights to Tokyo"),
            ("look up the Titanfall 2 release date", "the Titanfall 2 release date"),
            ("Could you google the Ollama changelog?", "the Ollama changelog"),
        ],
    )
    def test_an_explicit_order_searches_only_what_follows_it(self, order, expected):
        assert search_query(order) == expected


class TestSearch:
    def test_only_duckduckgo_is_ever_queried(self, fake_ddgs):
        # ddgs's default backend scrapes Google among others from this machine's IP, which got the
        # whole home connection captcha'd by Google. A single pinned engine is the whole fix.
        fake_ddgs["results"] = [{"title": "A site", "body": "text", "href": "https://a.site"}]

        search("anything")

        assert fake_ddgs["backends"] == ["duckduckgo"]

    def test_each_result_becomes_one_line_naming_its_site(self, fake_ddgs):
        fake_ddgs["results"] = [
            {"title": "BBC News", "body": "Something happened today.", "href": "https://www.bbc.co.uk/news"},
            {"title": "Reuters", "body": "It happened twice.", "href": "https://reuters.com/world"},
        ]

        assert search("what happened").splitlines() == [
            "- BBC News (bbc.co.uk): Something happened today.",
            "- Reuters (reuters.com): It happened twice.",
        ]

    def test_ragged_whitespace_in_snippets_is_collapsed(self, fake_ddgs):
        fake_ddgs["results"] = [{"title": "A site", "body": "spread\n  over   lines", "href": "https://a.site"}]

        assert search("anything") == "- A site (a.site): spread over lines"

    def test_a_failed_search_costs_the_answer_not_the_conversation(self, fake_ddgs):
        fake_ddgs["error"] = OSError("no network")

        assert search("anything") == ""

    def test_results_without_a_title_are_dropped(self, fake_ddgs):
        fake_ddgs["results"] = [{"body": "orphaned snippet"}, {"title": "Real", "body": "kept", "href": "https://real.org"}]

        assert search("anything") == "- Real (real.org): kept"


class TestLookup:
    def test_general_knowledge_is_never_looked_up_or_announced(self, fake_ddgs):
        announced = []

        web = lookup("What is the capital of Australia?", announce=lambda: announced.append(True))

        assert web == NOT_CHECKED
        assert fake_ddgs["calls"] == [], "no search should have been attempted"
        assert announced == []

    def test_live_questions_are_announced_then_looked_up(self, fake_ddgs):
        fake_ddgs["results"] = [{"title": "Forecast", "body": "Hot and dry.", "href": "https://www.accuweather.com/x"}]
        announced = []

        web = lookup("What's the weather today?", announce=lambda: announced.append(True))

        assert announced == [True]
        assert fake_ddgs["calls"][0][0] == "What's the weather today?"
        assert "- Forecast (accuweather.com): Hot and dry." in web.results

    def test_the_record_names_the_real_sources_once_each_and_none_of_the_bulk(self, fake_ddgs):
        fake_ddgs["results"] = [
            {"title": "Now", "body": "32 degrees.", "href": "https://www.accuweather.com/now"},
            {"title": "Hourly", "body": "33 degrees.", "href": "https://www.accuweather.com/hourly"},
            {"title": "Forecast", "body": "Rain later.", "href": "https://www.ndtv.com/weather"},
        ]

        web = lookup("What's the temp in Chennai?")

        assert web.record == "You checked the web to answer that, using accuweather.com, ndtv.com."
        assert "degrees" not in web.record

    def test_a_failed_search_tells_the_model_rather_than_letting_it_guess(self, fake_ddgs):
        fake_ddgs["error"] = OSError("no network")

        web = lookup("What's the weather today?")

        assert "search failed" in web.results and "couldn't check" in web.results
        assert "search failed" in web.record
