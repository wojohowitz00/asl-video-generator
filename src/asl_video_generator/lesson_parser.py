"""Parse lesson scripts from the 500 sentences curriculum."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Sentence:
    """A single lesson sentence with metadata."""
    
    text: str
    scenario: str
    subsection: str
    sentence_number: int
    difficulty: Literal["beginner", "intermediate", "advanced"] = "beginner"


@dataclass
class LessonScenario:
    """A collection of sentences for a scenario."""
    
    name: str
    subsections: dict[str, list[Sentence]]
    
    @property
    def total_sentences(self) -> int:
        return sum(len(sentences) for sentences in self.subsections.values())


class LessonParser:
    """Parse the 500 sentences markdown file into structured data."""
    
    # Difficulty assignment based on sentence complexity
    DIFFICULTY_RULES = {
        "beginner": lambda s: len(s.split()) <= 5,
        "intermediate": lambda s: 5 < len(s.split()) <= 10,
        "advanced": lambda s: len(s.split()) > 10,
    }
    
    def __init__(self, file_path: str | Path | None = None):
        """Initialize parser with the lesson file path."""
        if file_path is None:
            # Default path
            file_path = Path(__file__).parent.parent.parent.parent / (
                "ASL-Immersion-Companion/attached_assets/"
                "500_Sentences_1769817458445.md"
            )
        self.file_path = Path(file_path)
    
    def parse(self) -> list[LessonScenario]:
        """Parse the markdown file into structured scenarios."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Lesson file not found: {self.file_path}")
        
        content = self.file_path.read_text()
        return self._parse_content(content)
    
    def _parse_content(self, content: str) -> list[LessonScenario]:
        """Parse markdown content into scenarios."""
        scenarios = []
        current_scenario = None
        current_subsection = None
        
        # Regex patterns
        scenario_pattern = re.compile(r"^## Scenario (\d+): (.+)$")
        subsection_pattern = re.compile(r"^### (.+)$")
        sentence_pattern = re.compile(r"^(\d+)\. (.+)$")
        
        for line in content.split("\n"):
            line = line.strip()
            
            # Check for new scenario
            scenario_match = scenario_pattern.match(line)
            if scenario_match:
                if current_scenario:
                    scenarios.append(current_scenario)
                scenario_name = scenario_match.group(2)
                current_scenario = LessonScenario(name=scenario_name, subsections={})
                current_subsection = None
                continue
            
            # Check for new subsection
            subsection_match = subsection_pattern.match(line)
            if subsection_match and current_scenario:
                current_subsection = subsection_match.group(1)
                current_scenario.subsections[current_subsection] = []
                continue
            
            # Check for sentence
            sentence_match = sentence_pattern.match(line)
            if sentence_match and current_scenario and current_subsection:
                num = int(sentence_match.group(1))
                text = sentence_match.group(2)
                difficulty = self._assign_difficulty(text)
                
                sentence = Sentence(
                    text=text,
                    scenario=current_scenario.name,
                    subsection=current_subsection,
                    sentence_number=num,
                    difficulty=difficulty,
                )
                current_scenario.subsections[current_subsection].append(sentence)
        
        # Don't forget the last scenario
        if current_scenario:
            scenarios.append(current_scenario)
        
        return scenarios
    
    def _assign_difficulty(self, text: str) -> Literal["beginner", "intermediate", "advanced"]:
        """Assign difficulty based on sentence complexity."""
        word_count = len(text.split())
        
        # Also consider question complexity and vocabulary
        has_medical_terms = any(term in text.lower() for term in [
            "prescription", "allergic", "antibiotics", "diagnosis",
            "symptom", "emergency", "appointment"
        ])
        
        if has_medical_terms:
            return "advanced"
        elif word_count <= 5:
            return "beginner"
        elif word_count <= 8:
            return "intermediate"
        else:
            return "advanced"
    
    def to_json(self) -> list[dict]:
        """Parse and convert to JSON-serializable format."""
        scenarios = self.parse()
        result = []
        
        for scenario in scenarios:
            for subsection, sentences in scenario.subsections.items():
                for sentence in sentences:
                    result.append({
                        "id": f"{scenario.name.lower().replace(' ', '_')}_{sentence.sentence_number:03d}",
                        "text": sentence.text,
                        "scenario": scenario.name,
                        "subsection": subsection,
                        "difficulty": sentence.difficulty,
                    })
        
        return result
    
    def get_sentences_by_difficulty(
        self, difficulty: Literal["beginner", "intermediate", "advanced"]
    ) -> list[Sentence]:
        """Get all sentences of a specific difficulty level."""
        scenarios = self.parse()
        result = []
        
        for scenario in scenarios:
            for sentences in scenario.subsections.values():
                result.extend(s for s in sentences if s.difficulty == difficulty)
        
        return result


def parse_lessons(file_path: str | None = None) -> list[dict]:
    """Convenience function to parse lessons into JSON format."""
    parser = LessonParser(file_path)
    return parser.to_json()
