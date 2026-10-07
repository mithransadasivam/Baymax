from baymax.greeting import strip_greeting


class TestGreetingsAreRecognized:
    def test_the_name_alone_counts_as_a_greeting(self):
        assert strip_greeting("Baymax, what time is it?") == "what time is it?"

    def test_hey_baymax_is_recognized(self):
        assert strip_greeting("Hey Baymax, what is the capital of France?") == "what is the capital of France?"

    def test_hey_baymax_without_a_comma_still_works(self):
        assert strip_greeting("Hey Baymax what is the capital of France") == "what is the capital of France"

    def test_plain_hello_counts(self):
        assert strip_greeting("Hello, what is the weather like?") == "what is the weather like?"

    def test_a_time_of_day_greeting_counts(self):
        assert strip_greeting("Good morning, what do I have scheduled today?") == "what do I have scheduled today?"

    def test_matching_is_case_insensitive(self):
        assert strip_greeting("HEY BAYMAX, what time is it?") == "what time is it?"

    def test_a_bare_greeting_with_nothing_after_it_returns_an_empty_string(self):
        assert strip_greeting("Hey Baymax.") == ""
        assert strip_greeting("Baymax") == ""


class TestNonGreetingsAreIgnored:
    def test_a_question_with_no_greeting_at_all_returns_none(self):
        assert strip_greeting("What time is it?") is None

    def test_ordinary_speech_with_no_greeting_returns_none(self):
        assert strip_greeting("I think it might rain later.") is None

    def test_a_greeting_word_appearing_mid_sentence_does_not_count(self):
        assert strip_greeting("I said hello to my neighbour this morning.") is None


class TestTheAcceptedTradeOff:
    """Without a trained acoustic model, a common word said to someone else in the room can still
    match -- this is the real cost of not training a custom phrase, documented rather than hidden."""

    def test_a_greeting_meant_for_someone_else_still_matches(self):
        assert strip_greeting("Hey, did you catch the game last night?") == "did you catch the game last night?"

    def test_a_greeting_naming_someone_else_still_matches(self):
        assert strip_greeting("Hey Sarah, are you free later?") == "Sarah, are you free later?"


class TestLongestMatchWins:
    def test_hey_baymax_is_matched_whole_not_as_hey_plus_leftover_baymax(self):
        # If "hey" matched alone, "baymax, what's..." would be left sitting in front of the question.
        assert strip_greeting("Hey Baymax, what's 2 plus 2?") == "what's 2 plus 2?"
