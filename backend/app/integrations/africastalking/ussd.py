"""Africa's Talking USSD service for feature phone screening flow."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class USSDService:
    """USSD screening flow for feature phones.

    Implements a multi-step USSD menu:
    1. Language selection (English/Luganda)
    2. Patient NIN or name entry
    3. Screening type selection
    4. Result display
    """

    def handle_callback(self, session_id: str, phone: str, text: str) -> str:
        """Process USSD menu interaction.

        Parameters
        ----------
        session_id : str
            Africa's Talking session ID.
        phone : str
            Caller's phone number.
        text : str
            Accumulated user input (levels separated by *).

        Returns
        -------
        str
            Response prefixed with CON (continue) or END (terminate).
        """
        parts = text.split("*") if text else []
        level = len(parts)

        if level == 0:
            return (
                "CON Welcome to RetinalAI Screening\n"
                "Nkulamusizza mu RetinalAI\n\n"
                "1. English\n"
                "2. Luganda"
            )

        language = "lg" if parts[0] == "2" else "en"

        if level == 1:
            if language == "lg":
                return "CON Yingiza erinnya ly'omulwadde oba NIN:"
            return "CON Enter patient name or NIN:"

        if level == 2:
            patient = parts[1]
            if language == "lg":
                return (
                    f"CON Omulwadde: {patient}\n\n" "1. Okukebera okuggya\n" "2. Okukebera ebivaamu"
                )
            return f"CON Patient: {patient}\n\n" "1. New screening\n" "2. Check result"

        if level == 3:
            if language == "lg":
                return (
                    "END Nsaba okozese app ya RetinalAI ku simu yo "
                    "okukwata ekifaananyi. USSD teyinza kukebera maaso. "
                    "Webale!"
                )
            return (
                "END Please use the RetinalAI app on a smartphone "
                "to capture fundus images. USSD cannot perform screening. "
                "Thank you!"
            )

        if language == "lg":
            return "END Webale okukozesa RetinalAI."
        return "END Thank you for using RetinalAI."
