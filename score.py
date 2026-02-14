"""Metric for Deep Past Initiative 1 (121150): Geometric Mean of BLEU and CHRF++ scores."""

import math

import pandas as pd
import pandas.api.types
import sacrebleu


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    text_column_name: str,
) -> float:
    """Calculates the geometric average of BLEU and CHRF++ scores.

    This metric expects the solution and submission dataframes to contain text columns.

    The score is calculated as: sqrt(BLEU * CHRF++)
    Both BLEU and CHRF++ are on a 0-100 scale, so the result will be on a 0-100 scale.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing the ground truth text.

    submission : pd.DataFrame
        A DataFrame containing the predicted text.

    row_id_column_name : str
        The name of the column containing the row IDs. This column is removed
        before scoring.

    text_column_name : str
        The name of the column containing the text to be evaluated.

    Returns
    -------
    float
        The geometric mean of the BLEU and CHRF++ scores.

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> text_column_name = "text"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })

    Case: Perfect match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    100.0

    Case: Complete mismatch
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["Completely different.", "Nothing alike."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    0.0

    Case: Partial match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    75.7
    """

    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]

    # Validate submission column type
    if not (
        pandas.api.types.is_string_dtype(submission[text_column_name])
        or pandas.api.types.is_object_dtype(submission[text_column_name])
    ):
        raise ParticipantVisibleError(
            f"Submission column '{text_column_name}' must be of string type."
        )

    # Extract lists of strings
    references = solution[text_column_name].astype(str).tolist()
    hypotheses = submission[text_column_name].astype(str).tolist()

    # Calculate BLEU
    # corpus_bleu expects lists of references (list of lists)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    # Calculate CHRF++ (word_order=2)
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return math.sqrt(bleu.score * chrf.score)