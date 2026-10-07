import pytest

from baymax import emergency


@pytest.mark.parametrize(
    "text",
    [
        "I have crushing chest pain",
        "My chest feels tight and my left arm is numb",
        "my dad's face is drooping and his speech is slurred",
        "I can't breathe",
        "he's choking on a grape",
        "my throat is swelling after eating peanuts",
        "the bleeding won't stop",
        "I think I overdosed",
        "I took too many pills",
        "she is having a seizure right now",
        "my baby is unresponsive",
        "I want to kill myself",
        "sometimes I think everyone would be better off without me",
    ],
)
def test_real_emergencies_are_caught(text):
    assert emergency.check(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "I don't have any chest pain",
        "What causes a stroke?",
        "What are the symptoms of a heart attack?",
        "my dad had a stroke ten years ago",
        "How much ibuprofen can I take?",
        "My knee hurts when I run",
        "I have a mild sore throat",
        "Explain what anaphylaxis is",
        "I've had a headache since yesterday",
    ],
)
def test_ordinary_questions_are_not_treated_as_emergencies(text):
    assert emergency.check(text) is None


def test_suicidal_thoughts_get_the_crisis_script_with_988():
    alert = emergency.check("I want to end my life")

    assert alert.kind == emergency.CRISIS
    assert "9 8 8" in alert.script
    assert "911" in alert.script


def test_suicidal_thoughts_are_never_filtered_by_tense_or_phrasing():
    # Unlike a physical emergency, "I used to want to die" or "why do I want to die" still deserves a reply.
    assert emergency.check("Why do I want to die?") is not None


def test_poisoning_points_to_poison_control():
    assert "2 2 2" in emergency.check("my toddler swallowed bleach").script
