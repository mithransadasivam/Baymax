from baymax.text import SentenceBuffer, clean_for_speech


def feed_all(buffer: SentenceBuffer, tokens: list[str]) -> list[str]:
    sentences = [s for token in tokens for s in buffer.feed(token)]
    return sentences + buffer.flush()


class TestSentenceBuffer:
    def test_releases_a_sentence_once_the_next_one_starts(self):
        buffer = SentenceBuffer()
        assert buffer.feed("The reactor is stable, sir.") == []
        assert buffer.feed(" Shall I") == ["The reactor is stable, sir."]
        assert buffer.flush() == ["Shall I"]

    def test_merges_short_fragments_into_the_following_sentence(self):
        buffer = SentenceBuffer()
        assert feed_all(buffer, ["Yes. That is entirely correct, sir. Anything else?"]) == [
            "Yes. That is entirely correct, sir.",
            "Anything else?",
        ]

    def test_does_not_split_decimals(self):
        buffer = SentenceBuffer()
        assert feed_all(buffer, ["It costs 3.50 dollars at the moment, sir."]) == [
            "It costs 3.50 dollars at the moment, sir."
        ]

    def test_token_by_token_streaming_matches_feeding_everything_at_once(self):
        text = "Good evening, sir. The weather is dreadful. I would stay inside tonight!\nShall I dim the lights?"
        whole = feed_all(SentenceBuffer(), [text])
        streamed = feed_all(SentenceBuffer(), list(text))
        assert streamed == whole
        assert len(whole) == 3

    def test_splits_on_newlines(self):
        buffer = SentenceBuffer()
        assert feed_all(buffer, ["Here are the options available\nThe first is to wait"]) == [
            "Here are the options available",
            "The first is to wait",
        ]

    def test_keeps_closing_quotes_with_their_sentence(self):
        buffer = SentenceBuffer()
        assert feed_all(buffer, ['He said "stand down immediately." Then he left.']) == [
            'He said "stand down immediately."',
            "Then he left.",
        ]

    def test_flush_on_empty_buffer_returns_nothing(self):
        assert SentenceBuffer().flush() == []


class TestCleanForSpeech:
    def test_strips_markdown_emphasis_and_headings(self):
        assert clean_for_speech("## Status\n**All** systems are *nominal*.") == "Status All systems are nominal."

    def test_keeps_link_text_and_drops_urls(self):
        assert clean_for_speech("See [the docs](https://example.com) or https://example.org now.") == "See the docs or now."

    def test_strips_list_markers(self):
        assert clean_for_speech("- first\n2. second\n* third") == "first second third"

    def test_strips_emoji(self):
        assert clean_for_speech("Done 🚀 sir ✅") == "Done sir"

    def test_leaves_ordinary_sentences_alone(self):
        sentence = "It's 3.5 degrees outside, sir, so take a coat."
        assert clean_for_speech(sentence) == sentence
