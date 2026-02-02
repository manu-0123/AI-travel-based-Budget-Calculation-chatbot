import spacy
import re
import os

# -------------------------------------------------
# LOAD SPACY MODEL (SAFE)
# -------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class NLPUtils:
    def __init__(self):
        self.nlp = nlp
        # Supported travel modes
        self.travel_modes = ["bus", "train", "flight", "car"]

    def extract_entities(self, text):
        """
        Always returns ALL required keys:
        source, destination, days, people
        """
        doc = self.nlp(text)

        # ✅ Always initialize defaults
        entities = {
            "source": None,
            "destination": None,
            "days": 1,
            "people": 1,
            "travel_mode": "Flight"  # default ML-safe
        }

        text_lower = text.lower()

        # -------------------------------------------------
        # ROUTE EXTRACTION (from X to Y)
        # -------------------------------------------------
        from_match = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s", text_lower)
        to_match = re.search(
            r"to\s+([a-zA-Z\s]+?)(?:\s+for|\s+\d|\s+with|$)",
            text_lower
        )

        if from_match:
            entities["source"] = from_match.group(1).strip().title()

        if to_match:
            entities["destination"] = to_match.group(1).strip().title()

        # -------------------------------------------------
        # NER FALLBACK (GPE)
        # -------------------------------------------------
        locations = [
            ent.text.strip().title()
            for ent in doc.ents
            if ent.label_ == "GPE"
        ]

        if not entities["source"] and len(locations) >= 1:
            entities["source"] = locations[0]

        if not entities["destination"] and len(locations) >= 2:
            entities["destination"] = locations[1]

        # Handle single-city input: "Trip to Goa"
        if entities["source"] and not entities["destination"]:
            entities["destination"] = entities["source"]
            entities["source"] = None
        # -------------------------------------------------
        # DAYS & NIGHTS EXTRACTION
        # -------------------------------------------------
        day_match = re.search(r"(\d+)\s*days?", text_lower)
        night_match = re.search(r"(\d+)\s*nights?", text_lower)

        if day_match:
            entities["days"] = max(1, int(day_match.group(1)))
        elif night_match:
            entities["days"] = max(1, int(night_match.group(1)) + 1)

        # -------------------------------------------------
        # PEOPLE EXTRACTION
        # -------------------------------------------------
        people_match = re.search(
            r"(\d+)\s*(people|persons|pax|guys|members)",
            text_lower
        )

        if people_match:
            entities["people"] = max(1, int(people_match.group(1)))

        # -------------------------------------------------
        # TRAVEL MODE EXTRACTION
        # -------------------------------------------------
        for mode in self.travel_modes:
            if mode in text_lower:
                entities["travel_mode"] = mode.title()
                break

        return entities


    def clean_text(self, text):
        doc = self.nlp(text.lower())
        return " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        )
