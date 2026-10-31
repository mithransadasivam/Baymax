import time
from types import SimpleNamespace

import pytest

from baymax import brain as brain_module
from baymax.brain import Brain
from baymax.vitals import VitalsStore, asks_about_heart, band, describe, token, token_matches


@pytest.fixture
def store(tmp_path) -> VitalsStore:
    return VitalsStore(tmp_path)


class TestStore:
    def test_a_reading_is_saved_and_read_back(self, store):
        store.add(72)

        assert store.latest().bpm == 72
        assert store.path.exists()

    def test_the_latest_of_a_kind_is_found(self, store):
        store.add(60, kind="resting_heart_rate", ts=time.time() - 100)
        store.add(95, kind="heart_rate")

        assert store.latest("resting_heart_rate").bpm == 60
        assert store.latest().bpm == 95

    @pytest.mark.parametrize("bad", [0, 10, 24, 251, 900, -5])
    def test_a_value_no_heart_could_have_is_refused(self, store, bad):
        with pytest.raises(ValueError):
            store.add(bad)
        assert store.latest() is None

    def test_an_unknown_kind_is_refused(self, store):
        with pytest.raises(ValueError):
            store.add(70, kind="blood_pressure")

    def test_a_timestamp_in_the_future_is_treated_as_now(self, store):
        reading = store.add(70, ts=time.time() + 10_000)

        assert reading.ts <= time.time() + 1

    def test_a_damaged_line_in_the_file_is_skipped_not_fatal(self, store):
        store.add(70)
        with open(store.path, "a", encoding="utf-8") as handle:
            handle.write("this is not json\n")
        store.add(80)

        assert [r.bpm for r in store.readings()] == [70, 80]

    def test_the_numbers_stay_on_disk_in_plain_text(self, store):
        store.add(72)

        assert '"bpm": 72' in store.path.read_text(encoding="utf-8")


class TestBands:
    @pytest.mark.parametrize(
        "bpm, label",
        [(35, "very low"), (50, "below the usual range"), (60, "normal"), (72, "normal"), (100, "normal"),
         (110, "a little high"), (130, "high")],
    )
    def test_each_resting_range_is_labelled(self, bpm, label):
        assert band(bpm)[0] == label

    def test_a_mildly_high_reading_leads_with_ordinary_causes_not_alarm(self):
        _, why = band(110)

        assert "caffeine" in why and "stress" in why
        assert "cancer" not in why.lower()


class TestIntent:
    @pytest.mark.parametrize(
        "text",
        ["Check my heart rate", "What's my pulse?", "Is my resting heart rate okay?", "My heart is racing",
         "How's my heart rate looking", "I keep getting palpitations", "scan my pulse"],
    )
    def test_questions_about_the_users_own_heart_rate_are_recognised(self, text):
        assert asks_about_heart(text)

    @pytest.mark.parametrize(
        "text", ["What is the capital of France?", "How do I take ibuprofen?", "I have a headache", "Hello"]
    )
    def test_other_questions_are_not(self, text):
        assert not asks_about_heart(text)


class TestDescribe:
    def test_with_no_readings_it_says_so_and_forbids_guessing(self):
        note = describe([])

        assert "no reading" in note.lower() and "do not guess" in note.lower()

    def test_the_latest_number_and_its_band_appear(self, store):
        store.add(72)

        note = describe(store.readings())

        assert "72 beats per minute" in note and "normal" in note

    def test_an_old_reading_is_flagged_as_possibly_out_of_date(self, store):
        store.add(72, ts=time.time() - 3 * 3600)

        assert "not recent" in describe(store.readings())

    def test_a_day_of_readings_is_summarised(self, store):
        for bpm in (62, 70, 90, 75):
            store.add(bpm)

        assert "from 62 to 90" in describe(store.readings())

    def test_the_note_tells_the_model_not_to_diagnose_a_rhythm(self, store):
        store.add(72)

        assert "don't diagnose" in describe(store.readings())


class TestUrgentAlert:
    def test_a_very_fast_fresh_reading_with_chest_pain_gets_the_911_script(self, store):
        store.add(168)

        alert = store.urgent_alert("My heart is pounding and I have chest pain")

        assert alert and "911" in alert and "168" in alert

    def test_a_very_slow_fresh_reading_with_fainting_gets_the_script(self, store):
        store.add(38)

        assert "911" in store.urgent_alert("I feel faint and dizzy")

    def test_an_extreme_reading_without_a_symptom_is_not_escalated(self, store):
        store.add(160)

        assert store.urgent_alert("Just curious what my heart rate was after my run") is None

    def test_a_symptom_with_an_ordinary_reading_is_not_escalated_by_this_rule(self, store):
        store.add(80)

        assert store.urgent_alert("I feel a bit dizzy") is None

    def test_a_stale_reading_is_never_used(self, store):
        store.add(170, ts=time.time() - 3 * 3600)

        assert store.urgent_alert("I have chest pain") is None

    def test_a_negated_symptom_does_not_trigger_it(self, store):
        store.add(170)

        assert store.urgent_alert("My pulse was high but I have no chest pain and I'm not dizzy") is None

    def test_no_readings_means_no_alert(self, store):
        assert store.urgent_alert("I have chest pain") is None


class TestToken:
    def test_it_is_created_once_and_then_stable(self, tmp_path):
        first = token(tmp_path)

        assert first and token(tmp_path) == first
        assert (tmp_path / "vitals-token").exists()

    def test_matching_is_exact(self):
        assert token_matches("abc", "abc")
        assert not token_matches("abc", "abd")
        assert not token_matches("abc", None)
        assert not token_matches("abc", "")


class TestBrainIntegration:
    @pytest.fixture
    def sent(self, monkeypatch):
        conversations = []

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            def chat(self, model, messages, stream, **kwargs):
                conversations.append([dict(m) for m in messages])
                yield SimpleNamespace(message=SimpleNamespace(content="Your heart rate looks normal."))

        monkeypatch.setattr(brain_module.ollama, "Client", FakeClient)
        return conversations

    def test_the_reading_is_handed_to_the_model_when_asked(self, sent, store):
        store.add(72)
        brain = Brain(model="m", host="h", vitals=store)

        "".join(brain.reply("Check my heart rate"))

        assert any("72 beats per minute" in m["content"] for m in sent[0] if m["role"] == "system")

    def test_it_is_not_added_to_unrelated_questions(self, sent, store):
        store.add(72)
        brain = Brain(model="m", host="h", vitals=store)

        "".join(brain.reply("What is the usual dose of ibuprofen?"))

        assert not any("beats per minute" in m["content"] for m in sent[0])

    def test_an_urgent_reading_with_a_symptom_bypasses_the_model(self, sent, store):
        store.add(170)
        brain = Brain(model="m", host="h", vitals=store)

        reply = "".join(brain.reply("My heart is racing and I feel faint"))

        assert "911" in reply
        assert sent == [], "the model must not be consulted"

    def test_without_a_vitals_store_nothing_changes(self, sent):
        brain = Brain(model="m", host="h")

        "".join(brain.reply("Check my heart rate"))

        assert not any("Apple Watch" in m["content"] for m in sent[0])
