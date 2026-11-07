"""Coach decision policy: risk > form cue > occasional good-rep."""
from __future__ import annotations

from dataclasses import dataclass


CUES = {
    2: "Push your knees out.",
    3: "Go lower.",
    4: "Keep your chest up.",
    5: "Keep your knee behind your toes.",
    6: "Keep your knee behind your toes.",
    7: "Flatten your lower back.",
    8: "Open your chest.",
}


@dataclass
class Decision:
    type: str  # stop|slow|form|good|none
    message: str
    priority: int


class Policy:
    def __init__(self, good_every=3):
        self.good_every = good_every
        self._rep_count = 0
        self._last_message = None

    def decide(self, risk=0.0, form_mistake=None, form_prob=0.0):
        """One message per rep. stop > slow > form > good."""
        self._rep_count += 1
        if risk >= 0.7:
            d = Decision("stop", "Stop the set.", 3)
        elif risk >= 0.4:
            d = Decision("slow", "Slow down, control the movement.", 2)
        elif form_mistake is not None and form_prob >= 0.6:
            d = Decision("form", CUES.get(int(form_mistake), "Check your form."), 1)
        elif self._rep_count % self.good_every == 0:
            d = Decision("good", "Good rep.", 0)
        else:
            d = Decision("none", "", -1)
        # debounce: never repeat the exact same message back-to-back
        if d.type != "none" and d.message == self._last_message:
            d = Decision("none", "", -1)
        if d.type != "none":
            self._last_message = d.message
        return d
