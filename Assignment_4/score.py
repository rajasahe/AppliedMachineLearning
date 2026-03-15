"""
score.py - Scoring function for SMS Spam Classification.

Provides a function to score a trained sklearn model/pipeline on a single text input,
returning a binary prediction and a propensity (probability) score.
"""

import numpy as np


def score(text: str, model, threshold: float) -> tuple:
    """
    Score a trained model on a single text input.

    Parameters
    ----------
    text : str
        The input text to classify (e.g., an SMS message).
    model : sklearn estimator or Pipeline
        A trained sklearn model or Pipeline that can process raw text.
        Must support either `predict_proba` or `decision_function`.
    threshold : float
        Decision threshold for classification. If the propensity score
        is >= threshold, the prediction is True (spam); otherwise False (not spam).

    Returns
    -------
    prediction : bool
        True if the text is classified as spam (positive class), False otherwise.
    propensity : float
        The propensity (probability) score for the positive class, between 0 and 1.

    Raises
    ------
    ValueError
        If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected text to be a string, got {type(text)}")

    # Wrap the text in a list for sklearn compatibility (expects iterable)
    text_input = [text]

    # Calculate propensity score
    if hasattr(model, 'predict_proba'):
        # Models with predict_proba (e.g., LogisticRegression, RandomForest, NaiveBayes)
        proba = model.predict_proba(text_input)
        propensity = float(proba[0][1])  # Probability of positive class (spam)
    elif hasattr(model, 'decision_function'):
        # Models with decision_function (e.g., LinearSVC)
        # Apply sigmoid to convert decision function output to [0, 1] range
        decision = model.decision_function(text_input)
        propensity = float(1 / (1 + np.exp(-decision[0])))
    else:
        raise AttributeError(
            "Model must have either 'predict_proba' or 'decision_function' method."
        )

    # Apply threshold to determine prediction
    prediction = propensity >= threshold

    return prediction, propensity
