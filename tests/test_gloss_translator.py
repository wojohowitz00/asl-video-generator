"""Tests for gloss translator module."""



def test_gloss_sequence_model():
    """Test GlossSequence Pydantic model."""
    from asl_video_generator.gloss_translator import GlossSequence

    seq = GlossSequence(
        english="Hello",
        gloss=["HELLO"],
        estimated_duration_ms=500,
    )

    assert seq.english == "Hello"
    assert seq.gloss == ["HELLO"]
    assert seq.nmm is not None


def test_nmm_question_types():
    """Test NonManualMarkers for different question types."""
    from asl_video_generator.gloss_translator import NonManualMarkers

    # Yes/No question
    yn = NonManualMarkers(is_question=True, question_type="yes_no", eyebrow_position="raised")
    assert yn.is_question
    assert yn.question_type == "yes_no"

    # WH question
    wh = NonManualMarkers(is_question=True, question_type="wh", eyebrow_position="furrowed")
    assert wh.question_type == "wh"

    # Negation
    neg = NonManualMarkers(is_negation=True, head_movement="shake")
    assert neg.is_negation


def test_vocabulary_set_exists():
    """Test ASL vocabulary set is populated."""
    from asl_video_generator.gloss_translator import ASL_CORE_VOCABULARY

    assert len(ASL_CORE_VOCABULARY) > 100
    assert "HELLO" in ASL_CORE_VOCABULARY
    assert "THANK-YOU" in ASL_CORE_VOCABULARY
    assert "WHERE" in ASL_CORE_VOCABULARY


def test_translator_fallback_parsing():
    """Test translator fallback when LLM is unavailable."""
    from asl_video_generator.gloss_translator import GlossTranslator

    # Create translator without API key (will use fallback)
    translator = GlossTranslator(provider="openai", enable_cache=False)

    # Test the fallback parsing directly
    result = translator._parse_response("Hello?", "{invalid json}")

    assert result.english == "Hello?"
    assert len(result.gloss) > 0
    assert result.nmm.is_question  # Detected from ?


def test_translator_wh_question_detection():
    """Test WH-question detection in fallback."""
    from asl_video_generator.gloss_translator import GlossTranslator

    translator = GlossTranslator(provider="openai", enable_cache=False)

    result = translator._parse_response("Where is the library?", "{bad}")

    assert result.nmm.is_question
    assert result.nmm.question_type == "wh"
    assert result.nmm.eyebrow_position == "furrowed"


def test_translator_yes_no_question_detection():
    """Test yes/no question detection in fallback."""
    from asl_video_generator.gloss_translator import GlossTranslator

    translator = GlossTranslator(provider="openai", enable_cache=False)

    result = translator._parse_response("Do you understand?", "{bad}")

    assert result.nmm.is_question
    assert result.nmm.question_type == "yes_no"
    assert result.nmm.eyebrow_position == "raised"


def test_vocabulary_validation():
    """Test vocabulary validation keeps known signs."""
    from asl_video_generator.gloss_translator import GlossSequence, GlossTranslator

    translator = GlossTranslator(provider="openai", enable_cache=False)

    seq = GlossSequence(
        english="test",
        gloss=["HELLO", "UNKNOWN_SIGN", "THANK-YOU"],
        estimated_duration_ms=1500,
    )

    validated = translator._validate_vocabulary(seq)

    # Known signs should be kept
    assert "HELLO" in validated.gloss
    assert "THANK-YOU" in validated.gloss


def test_fingerspelling_detection():
    """Test fingerspelling format detection."""
    from asl_video_generator.gloss_translator import GlossTranslator

    translator = GlossTranslator(provider="openai", enable_cache=False)

    seq_with_fs = translator._validate_vocabulary(
        type("MockSeq", (), {"gloss": ["J-O-H-N", "HELLO"]})()
    )

    # Fingerspelled names should be kept
    assert "J-O-H-N" in seq_with_fs.gloss
