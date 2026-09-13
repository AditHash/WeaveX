"""Phase 11 tests — extract_candidate_entities.

Uses the real FastAPICorpus paragraphs. Sets below were verified with a
standalone script before writing these assertions (this exact process
caught a real bug: an earlier ASCII-only regex silently dropped
"Sebastián Ramírez" entirely — fixed to allow Unicode letters after the
first character).
"""

from app.chunking import extract_candidate_entities


def test_extracts_single_word_entities():
    result = extract_candidate_entities("FastAPI is a modern Python web framework.")
    assert result == {"FastAPI", "Python"}


def test_extracts_multi_word_entity_phrase():
    result = extract_candidate_entities("Sebastián Ramírez created FastAPI.")
    assert result == {"Sebastián Ramírez", "FastAPI"}


def test_extracts_accented_names_correctly():
    # The specific bug this test guards against: an ASCII-only character
    # class matches "Sebasti" then fails at the accented "á", silently
    # dropping the whole name instead of raising — verified with a
    # standalone script before the regex was fixed.
    result = extract_candidate_entities("Sebastián Ramírez")
    assert "Sebastián Ramírez" in result
    assert "Sebasti" not in result


def test_no_entities_in_lowercase_only_text():
    assert extract_candidate_entities("this sentence has no proper nouns") == set()


def test_known_limitation_sentence_starter_capitalization():
    # Documents the actual false-positive behavior rather than hiding it:
    # a purely generic sentence-initial word is captured as a "candidate
    # entity" just because it's capitalized.
    result = extract_candidate_entities("The quick fox jumps.")
    assert "The" in result
