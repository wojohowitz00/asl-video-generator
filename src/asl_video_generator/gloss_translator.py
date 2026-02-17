"""LLM-based English to ASL Gloss translator with NMM prediction and caching.

Supports multiple LLM providers:
- OpenAI (GPT-4o) - highest quality
- Gemini - good quality, different pricing
- Ollama - local, offline, free

Includes vocabulary restriction to ensure valid ASL glosses and
explicit Non-Manual Marker (NMM) prediction for grammatical accuracy.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from .config import PipelineConfig, load_config_from_env


class NonManualMarkers(BaseModel):  # type: ignore[misc]
    """Non-manual markers for ASL grammatical expression.

    NMMs convey critical grammatical information in ASL:
    - Question type (yes/no vs wh-question)
    - Negation
    - Conditional statements
    - Topic markers
    """

    facial_expression: str = Field(
        default="neutral",
        description="Primary facial expression: neutral, happy, sad, surprised, confused",
    )
    head_movement: str | None = Field(
        default=None,
        description="Head movement: nod, shake, tilt-right, tilt-left, forward",
    )
    eyebrow_position: str | None = Field(
        default=None,
        description="Eyebrow position: raised (yes/no Q), furrowed (wh-Q), neutral",
    )
    eye_gaze: str | None = Field(
        default=None,
        description="Eye gaze direction for referencing: left, right, up, addressee",
    )
    mouth_morpheme: str | None = Field(
        default=None,
        description="Mouth morpheme: mm (casual), oo (small), cha (large), pah (success)",
    )
    is_question: bool = False
    question_type: Literal["yes_no", "wh", "rhetorical", "none"] = "none"
    is_negation: bool = False
    is_conditional: bool = False
    is_topic: bool = False
    # Span information - which gloss indices this NMM applies to
    start_index: int = 0
    end_index: int | None = None  # None means until end


class GlossSequence(BaseModel):  # type: ignore[misc]
    """ASL gloss sequence with timing and NMM annotations."""

    english: str
    gloss: list[str]
    # Multiple NMM spans can overlay the gloss sequence
    nmm_spans: list[NonManualMarkers] = Field(default_factory=list)
    # Primary NMM for backwards compatibility
    nmm: NonManualMarkers = Field(default_factory=NonManualMarkers)
    estimated_duration_ms: int = 0
    difficulty: Literal["beginner", "intermediate", "advanced"] = "beginner"

    def model_post_init(self, __context: Any) -> None:
        """Ensure backwards compatibility with single NMM."""
        if not self.nmm_spans and self.nmm:
            self.nmm_spans = [self.nmm]


# Common ASL vocabulary for validation
# This list restricts LLM output to known signs in typical pose dictionaries
ASL_CORE_VOCABULARY = {
    # Pronouns
    "I", "ME", "MY", "MINE", "YOU", "YOUR", "HE", "SHE", "IT", "WE", "THEY", "THEIR",
    # Questions
    "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH",
    # Common verbs
    "GO", "COME", "WANT", "NEED", "LIKE", "LOVE", "HATE", "HAVE", "GIVE", "GET",
    "TAKE", "MAKE", "DO", "SAY", "TELL", "ASK", "KNOW", "THINK", "SEE", "LOOK",
    "WATCH", "HEAR", "LISTEN", "EAT", "DRINK", "SLEEP", "WAKE-UP", "WORK", "PLAY",
    "LEARN", "TEACH", "STUDY", "READ", "WRITE", "HELP", "TRY", "FINISH", "START",
    "STOP", "WAIT", "MEET", "VISIT", "LIVE", "DIE", "BORN", "GROW", "CHANGE",
    "UNDERSTAND", "REMEMBER", "FORGET", "FEEL", "HURT", "CRY", "LAUGH", "SMILE",
    # Time
    "NOW", "TODAY", "TOMORROW", "YESTERDAY", "MORNING", "AFTERNOON", "EVENING",
    "NIGHT", "WEEK", "MONTH", "YEAR", "ALWAYS", "NEVER", "SOMETIMES", "OFTEN",
    "BEFORE", "AFTER", "DURING", "SINCE", "UNTIL", "AGAIN", "STILL", "ALREADY",
    "EARLY", "LATE", "SOON", "LATER", "RECENTLY", "FUTURE", "PAST", "TIME",
    # Greetings and common phrases
    "HELLO", "HI", "GOODBYE", "BYE", "PLEASE", "THANK-YOU", "THANKS", "SORRY",
    "EXCUSE-ME", "YES", "NO", "MAYBE", "OK", "FINE", "GOOD", "BAD", "BETTER",
    "WORSE", "BEST", "WORST", "NICE", "WELCOME",
    # People
    "PERSON", "PEOPLE", "MAN", "WOMAN", "BOY", "GIRL", "CHILD", "CHILDREN",
    "BABY", "FAMILY", "MOTHER", "FATHER", "PARENT", "SON", "DAUGHTER", "BROTHER",
    "SISTER", "GRANDMOTHER", "GRANDFATHER", "AUNT", "UNCLE", "COUSIN", "FRIEND",
    "TEACHER", "STUDENT", "DOCTOR", "NURSE", "BOSS", "WORKER", "NAME",
    # Places
    "HOME", "HOUSE", "SCHOOL", "OFFICE", "STORE", "SHOP", "HOSPITAL",
    "CHURCH", "LIBRARY", "RESTAURANT", "BANK", "PARK", "CITY", "TOWN", "COUNTRY",
    "ROOM", "BATHROOM", "KITCHEN", "BEDROOM", "BUILDING", "STREET", "PLACE",
    # Numbers (fingerspelled or signed)
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    "HUNDRED", "THOUSAND", "MILLION", "FIRST", "SECOND", "THIRD", "LAST",
    # Colors
    "RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "ORANGE", "PURPLE",
    "PINK", "BROWN", "GRAY", "COLOR",
    # Adjectives
    "BIG", "SMALL", "LITTLE", "LARGE", "TALL", "SHORT", "LONG", "OLD", "YOUNG",
    "NEW", "HOT", "COLD", "WARM", "COOL", "FAST", "SLOW", "EASY", "HARD",
    "DIFFICULT", "SIMPLE", "BEAUTIFUL", "UGLY", "CLEAN", "DIRTY", "FULL",
    "EMPTY", "OPEN", "CLOSE", "RIGHT", "WRONG", "CORRECT", "TRUE", "FALSE",
    "SAME", "DIFFERENT", "OTHER", "ANOTHER", "IMPORTANT", "INTERESTING", "BORING",
    "HAPPY", "SAD", "ANGRY", "SCARED", "TIRED", "SICK", "HEALTHY", "HUNGRY",
    "THIRSTY", "BUSY", "FREE", "READY", "SURE", "AFRAID", "EXCITED", "NERVOUS",
    # Negation and modifiers
    "NOT", "NONE", "NOTHING", "NO-ONE", "NOBODY", "VERY", "REALLY",
    "ALMOST", "COMPLETELY", "ONLY", "JUST", "ALSO", "TOO", "ENOUGH", "MORE",
    "LESS", "MOST", "LEAST", "ALL", "EVERY", "EACH", "SOME", "ANY", "MANY",
    "FEW", "MUCH",
    # Things
    "THING", "FOOD", "WATER", "MONEY", "PHONE", "COMPUTER", "CAR", "BOOK",
    "PAPER", "PEN", "CHAIR", "TABLE", "DOOR", "WINDOW", "LIGHT", "KEY",
    "CLOTHES", "SHOES", "HAT", "BAG", "BOX", "PICTURE", "MOVIE", "MUSIC",
    # Actions/Concepts
    "CAN", "WILL", "MUST", "SHOULD", "WOULD", "COULD", "MAY", "MIGHT",
    "BECAUSE", "BUT", "AND", "OR", "IF", "THEN", "SO", "WITH", "WITHOUT",
    "FOR", "ABOUT", "FROM", "TO", "IN", "ON", "AT", "BY", "UP", "DOWN",
}


@dataclass
class TranslationCache:
    """SQLite-based cache for translation results."""

    cache_path: Path

    def __post_init__(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the cache database."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    hash TEXT PRIMARY KEY,
                    english TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_translations_english ON translations(english)"
            )

    def _hash_input(self, english: str, provider: str) -> str:
        """Create hash for cache key."""
        content = f"{english.lower().strip()}|{provider}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, english: str, provider: str) -> GlossSequence | None:
        """Retrieve cached translation."""
        cache_hash = self._hash_input(english, provider)
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT result FROM translations WHERE hash = ?",
                (cache_hash,)
            ).fetchone()
            if row:
                data = json.loads(row[0])
                return GlossSequence(**data)
        return None

    def set(self, english: str, provider: str, result: GlossSequence) -> None:
        """Store translation in cache."""
        cache_hash = self._hash_input(english, provider)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO translations (hash, english, provider, result)
                   VALUES (?, ?, ?, ?)""",
                (cache_hash, english, provider, result.model_dump_json())
            )

    def clear(self) -> None:
        """Clear all cached translations."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM translations")


class GlossTranslator:
    """Translate English text to ASL gloss using LLM with NMM prediction.

    Features:
    - Multiple LLM provider support (OpenAI, Gemini, Ollama)
    - Vocabulary restriction for valid ASL glosses
    - Explicit NMM prediction for grammatical accuracy
    - SQLite-based caching to avoid redundant API calls
    """

    # Enhanced few-shot examples with detailed NMM annotations
    FEW_SHOT_EXAMPLES = [
        {
            "english": "Hello, how are you?",
            "gloss": ["HELLO", "HOW", "YOU"],
            "nmm": {"question_type": "wh", "eyebrow_position": "furrowed", "is_question": True},
        },
        {
            "english": "My name is John.",
            "gloss": ["MY", "NAME", "J-O-H-N"],
            "nmm": {"facial_expression": "neutral"},
        },
        {
            "english": "What time is it?",
            "gloss": ["TIME", "WHAT"],
            "nmm": {"question_type": "wh", "eyebrow_position": "furrowed", "is_question": True},
        },
        {
            "english": "Do you understand?",
            "gloss": ["YOU", "UNDERSTAND"],
            "nmm": {"question_type": "yes_no", "eyebrow_position": "raised", "is_question": True},
        },
        {
            "english": "I don't know.",
            "gloss": ["I", "KNOW", "NOT"],
            "nmm": {"is_negation": True, "head_movement": "shake"},
        },
        {
            "english": "Nice to meet you.",
            "gloss": ["NICE", "MEET", "YOU"],
            "nmm": {"facial_expression": "happy"},
        },
        {
            "english": "Where is the bathroom?",
            "gloss": ["BATHROOM", "WHERE"],
            "nmm": {"question_type": "wh", "eyebrow_position": "furrowed", "is_question": True},
        },
        {
            "english": "Thank you very much.",
            "gloss": ["THANK-YOU"],
            "nmm": {"facial_expression": "happy", "head_movement": "nod"},
        },
        {
            "english": "I went to the store yesterday.",
            "gloss": ["YESTERDAY", "STORE", "I", "GO"],
            "nmm": {"facial_expression": "neutral"},
            "note": "Time reference comes first in ASL",
        },
        {
            "english": "If it rains, I will stay home.",
            "gloss": ["RAIN", "IF", "I", "HOME", "STAY"],
            "nmm": {
                "is_conditional": True,
                "eyebrow_position": "raised",
                "start_index": 0,
                "end_index": 2,
            },
            "note": "Conditional clause marked with raised eyebrows",
        },
    ]

    SYSTEM_PROMPT = """You are an expert ASL (American Sign Language) translator and linguist.
Your task is to translate English text to ASL gloss notation
with accurate Non-Manual Markers (NMMs).

## ASL Gloss Rules:
1. Use UPPERCASE for all signs
2. Use hyphens for compound signs (THANK-YOU, WAKE-UP)
3. Use hyphens for fingerspelling proper nouns/names (J-O-H-N)
4. ASL uses topic-comment structure: time → topic → comment
5. WH-questions place the question word at the END
6. Negation typically places NOT after the verb with headshake
7. Use only signs from the provided vocabulary when possible

## Non-Manual Markers (NMMs) - CRITICAL for ASL grammar:

### Question Markers:
- Yes/No questions: RAISED eyebrows, slight head tilt forward
- WH-questions (what, where, when, who, why, how): FURROWED eyebrows, head forward

### Negation:
- Headshake throughout negated phrase
- Negative facial expression (tight lips, frown)

### Conditional (if/when):
- Raised eyebrows during condition clause
- Head tilt, eye gaze shift

### Topic Marker:
- Raised eyebrows
- Slight pause after topic

### Mouth Morphemes (adverbial markers):
- "mm" = enjoyment, regularly
- "oo" = small, thin
- "cha" = large, very
- "pah" = finally, success

## Output Format:
Return JSON with these fields:
- gloss: array of sign glosses in ASL order
- nmm: object with facial_expression, head_movement, eyebrow_position,
  eye_gaze, mouth_morpheme, is_question, question_type, is_negation,
  is_conditional, is_topic
- estimated_duration_ms: estimated signing duration (typically 400-600ms per sign)
- difficulty: "beginner", "intermediate", or "advanced"

## Example vocabulary reference (use these when possible):
HELLO, HOW, YOU, I, ME, MY, NAME, WHAT, WHERE, WHEN, WHY, WHO, TIME, TODAY,
YESTERDAY, TOMORROW, THANK-YOU, PLEASE, SORRY, YES, NO, GOOD, BAD, UNDERSTAND,
KNOW, NOT, WANT, NEED, LIKE, GO, COME, HELP, LEARN, WORK, HOME, SCHOOL,
BATHROOM, EAT, DRINK, SLEEP, HAPPY, SAD, TIRED"""

    def __init__(
        self,
        provider: Literal["openai", "gemini", "ollama"] = "openai",
        model: str | None = None,
        config: PipelineConfig | None = None,
        enable_cache: bool = True,
        restrict_vocabulary: bool = True,
    ):
        """Initialize translator with specified LLM provider.

        Args:
            provider: LLM provider to use (openai, gemini, ollama)
            model: Specific model name (defaults per provider)
            config: Pipeline configuration
            enable_cache: Whether to cache translations
            restrict_vocabulary: Whether to restrict output to known vocabulary
        """
        self.config = config or load_config_from_env()
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.restrict_vocabulary = restrict_vocabulary
        self._client = None
        self._cache: TranslationCache | None = None

        # Initialize cache
        if enable_cache:
            cache_path = self.config.cache_dir / "translation_cache.db"
            self._cache = TranslationCache(cache_path)

    def _default_model(self, provider: Literal["openai", "gemini", "ollama"]) -> str:
        """Get default model for provider."""
        return {
            "openai": "gpt-4o",
            "gemini": "gemini-1.5-flash",
            "ollama": "llama3.2",
        }.get(provider, "gpt-4o")

    def _get_openai_client(self) -> Any:
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_gemini_client(self) -> Any:
        """Lazy load Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def _get_ollama_client(self) -> Any:
        """Lazy load Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama.Client()
        return self._client

    def _build_prompt(self, text: str) -> str:
        """Build the translation prompt with few-shot examples."""
        examples = []
        for ex in self.FEW_SHOT_EXAMPLES[:6]:
            example_payload = {"gloss": ex["gloss"], "nmm": ex.get("nmm", {})}
            examples.append(
                "English: "
                f"{ex['english']}\n"
                "ASL Gloss: "
                f"{json.dumps(example_payload, indent=2)}"
            )
        examples_str = "\n\n".join(examples)

        return f"""Translate the following English text to ASL gloss with NMM annotations.

## Examples:
{examples_str}

## Now translate this:
English: {text}

Return ONLY valid JSON (no markdown, no explanation)."""

    def translate(self, text: str) -> GlossSequence:
        """Translate English text to ASL gloss sequence.

        Args:
            text: English text to translate.

        Returns:
            GlossSequence with gloss array and NMM annotations.
        """
        text = text.strip()
        if not text:
            return GlossSequence(english="", gloss=[], estimated_duration_ms=0)

        # Check cache first
        if self._cache:
            cached = self._cache.get(text, self.provider)
            if cached:
                return cached

        # Build prompt and translate
        prompt = self._build_prompt(text)

        if self.provider == "openai":
            response = self._translate_openai(prompt)
        elif self.provider == "gemini":
            response = self._translate_gemini(prompt)
        else:
            response = self._translate_ollama(prompt)

        # Parse the response
        result = self._parse_response(text, response)

        # Validate and restrict vocabulary if enabled
        if self.restrict_vocabulary:
            result = self._validate_vocabulary(result)

        # Cache the result
        if self._cache:
            self._cache.set(text, self.provider, result)

        return result

    def _translate_openai(self, prompt: str) -> str:
        """Translate using OpenAI GPT-4o."""
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return response.choices[0].message.content or "{}"

    def _translate_gemini(self, prompt: str) -> str:
        """Translate using Google Gemini."""
        client = self._get_gemini_client()
        response = client.generate_content(
            f"{self.SYSTEM_PROMPT}\n\n{prompt}",
            generation_config={"response_mime_type": "application/json"},
        )
        return response.text or "{}"

    def _translate_ollama(self, prompt: str) -> str:
        """Translate using local Ollama."""
        client = self._get_ollama_client()
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
        )
        return response["message"]["content"] or "{}"

    def _parse_response(self, english: str, response: str) -> GlossSequence:
        """Parse LLM response into GlossSequence."""
        try:
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            nmm_data = data.get("nmm", {})

            # Build NMM object
            nmm = NonManualMarkers(
                facial_expression=nmm_data.get("facial_expression", "neutral"),
                head_movement=nmm_data.get("head_movement"),
                eyebrow_position=nmm_data.get("eyebrow_position"),
                eye_gaze=nmm_data.get("eye_gaze"),
                mouth_morpheme=nmm_data.get("mouth_morpheme"),
                is_question=nmm_data.get("is_question", False),
                question_type=nmm_data.get("question_type", "none"),
                is_negation=nmm_data.get("is_negation", False),
                is_conditional=nmm_data.get("is_conditional", False),
                is_topic=nmm_data.get("is_topic", False),
            )

            gloss = data.get("gloss", [])
            estimated_duration = data.get(
                "estimated_duration_ms",
                len(gloss) * 500  # Default: 500ms per sign
            )

            return GlossSequence(
                english=english,
                gloss=gloss,
                nmm=nmm,
                nmm_spans=[nmm],
                estimated_duration_ms=estimated_duration,
                difficulty=data.get("difficulty", "beginner"),
            )

        except (json.JSONDecodeError, KeyError):
            # Fallback: simple word-by-word translation
            words = english.upper()
            for char in "?.,!;:\"'":
                words = words.replace(char, "")
            word_list = words.split()

            # Detect basic NMM from punctuation
            nmm = NonManualMarkers()
            if "?" in english:
                nmm.is_question = True
                # Detect wh-question words
                wh_words = {"what", "where", "when", "why", "how", "who", "which"}
                if any(w.lower() in english.lower() for w in wh_words):
                    nmm.question_type = "wh"
                    nmm.eyebrow_position = "furrowed"
                else:
                    nmm.question_type = "yes_no"
                    nmm.eyebrow_position = "raised"

            return GlossSequence(
                english=english,
                gloss=word_list,
                nmm=nmm,
                nmm_spans=[nmm],
                estimated_duration_ms=len(word_list) * 500,
            )

    def _validate_vocabulary(self, result: GlossSequence) -> GlossSequence:
        """Validate gloss against known vocabulary.

        Unknown signs are kept but flagged for potential fingerspelling
        or dictionary lookup.
        """
        validated_gloss = []
        for sign in result.gloss:
            # Remove any special formatting
            clean_sign = sign.upper().strip()

            # Check if it's a fingerspelled word (contains hyphens between single chars)
            is_fingerspelled = (
                "-" in clean_sign
                and all(len(part) == 1 for part in clean_sign.split("-"))
            )

            if clean_sign in ASL_CORE_VOCABULARY or is_fingerspelled:
                validated_gloss.append(clean_sign)
            else:
                # Check if it's a compound sign or variant
                base_sign = clean_sign.split("-")[0] if "-" in clean_sign else clean_sign
                if base_sign in ASL_CORE_VOCABULARY:
                    validated_gloss.append(clean_sign)
                else:
                    # Keep unknown sign but could flag for review
                    validated_gloss.append(clean_sign)

        result.gloss = validated_gloss
        return result


def translate_batch(
    texts: list[str],
    provider: str = "openai",
    config: PipelineConfig | None = None,
) -> list[GlossSequence]:
    """Translate a batch of English texts to ASL gloss sequences.

    Args:
        texts: List of English texts to translate.
        provider: LLM provider to use.
        config: Pipeline configuration.

    Returns:
        List of GlossSequence objects.
    """
    if provider not in {"openai", "gemini", "ollama"}:
        raise ValueError(f"Unsupported provider: {provider}")
    provider_value = cast(Literal["openai", "gemini", "ollama"], provider)
    translator = GlossTranslator(provider=provider_value, config=config)
    return [translator.translate(text) for text in texts]
