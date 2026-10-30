"""MP2 sample submission: a learned verifier combined with answer agreement.

Method
------
For every question we score the four candidate traces with a linear model and
then aggregate those scores per *parsed final answer*:

1. Each candidate's final answer is parsed with the same rule the official
   scorer uses (the last ``####`` tail, else the last ``\\boxed{...}``).
2. A logistic-regression verifier scores each candidate from three feature
   blocks:
     * 1024-d whole-trace embedding, centred within the question and
       L2-normalised (question-specific variation cancels out);
     * 12 cheap surface features (length, boxed/#### formatting, equation
       density, fraction/approximation style, ...);
     * 6 cross-candidate features (how many candidates share this answer,
       embedding similarity to the group centroid, to agreeing candidates
       and to disagreeing ones, embedding norm).
3. Candidates that end in the same parsed answer form one group. A second,
   pairwise-trained comparator ranks whole groups using the candidate scores,
   the group's support and cohesion, and its text statistics; the best group
   wins and its highest-scoring candidate is returned. Grouping matters
   because the scorer grades final answers: two traces that agree deserve
   shared support, and the smaller comparison stage is what decides between
   two plausible groups.

Training is done by ``train.py`` on the supplied training split only (public
training references); validation is used to choose hyper-parameters and the
test split is never touched during development.
The fitted checkpoint (verifier plus comparator) is about 25 KiB, far below the
16 MiB asset limit, and full prediction for 467 questions takes well under a
second on CPU, far below the 30 s limit.

If no checkpoint is supplied (``--checkpoint`` omitted) the method degrades to
pure answer-agreement voting, which is a legitimate training-free baseline.
"""

import re
from common import parse_answer

import numpy as np



# ---------------------------------------------------------------------------
# Feature extraction.
# ---------------------------------------------------------------------------
DIGIT = re.compile(r'\d+')


def text_features(text):
    """12 surface / reasoning-style features for one candidate trace."""
    low = text.lower()
    return np.array([
        np.log1p(len(text)),
        float('####' in text),
        float('\\boxed' in text),
        np.log1p(text.count('=')),
        np.log1p(len(DIGIT.findall(text))),
        np.log1p(text.count('\n')),
        float(text.rstrip().endswith('.')),
        float('final answer' in low),
        float('\\frac' in text),
        float('approx' in low),
        np.log1p(text.count('$')),
        float(low.count('?') > 2),
    ], dtype=np.float64)


def embedding_features(embeddings):
    """Centre each question's embeddings and L2-normalise each candidate.

    All four candidates share the question, so the centred vector isolates
    what is specific to *this* trace, which transfers better to unseen
    questions than the raw embedding direction.
    """
    centred = embeddings - embeddings.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=2, keepdims=True)
    return centred / (norm + 1e-8)


def group_features(embeddings, answers):
    """6 features describing a candidate relative to its three siblings."""
    n = len(answers)
    unit = embeddings / (np.linalg.norm(embeddings, axis=2, keepdims=True) + 1e-8)
    out = np.zeros((n, 4, 6), dtype=np.float32)
    for q in range(n):
        row = answers[q]
        counts = {}
        for answer in row:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        centroid = unit[q].mean(axis=0)
        for i in range(4):
            same = [j for j in range(4) if j != i and row[j] == row[i] and row[i] is not None]
            diff = [j for j in range(4) if j != i and row[j] != row[i]]
            out[q, i] = [
                counts.get(row[i], 0) / 4.0,
                len(same) / 3.0,
                float(unit[q, i] @ centroid),
                float(unit[q, i] @ unit[q, same].mean(axis=0)) if same else 0.0,
                float(unit[q, i] @ unit[q, diff].mean(axis=0)) if diff else 0.0,
                float(np.linalg.norm(embeddings[q, i])),
            ]
    return out


def feature_matrix(bank, embeddings):
    """Build the [N, 4, 1042] design tensor from a bank and its embeddings."""
    n = len(bank)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.shape[:2] != (n, 4):
        raise ValueError('embeddings must have shape [N, 4, D]')
    text = np.zeros((n, 4, 12), dtype=np.float32)
    answers = [[None] * 4 for _ in range(n)]
    for q, row in enumerate(bank):
        for cand in row['candidates']:
            i = int(cand['index'])
            text[q, i] = text_features(cand['text'])
            answers[q][i] = parse_answer(cand['text'])
    parts = [embedding_features(embeddings), text, group_features(embeddings, answers)]
    return np.concatenate(parts, axis=2), answers


# ---------------------------------------------------------------------------
# Checkpoint handling.
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint):
    """Load the fitted parameters; return None when no usable asset exists."""
    if checkpoint is None:
        return None
    path = checkpoint
    if path.is_dir():
        path = path / 'model.npz'
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


# ---------------------------------------------------------------------------
# Selection rules.
# ---------------------------------------------------------------------------
def choose_by_agreement(bank, answers):
    """Training-free fallback: majority vote over parsed answers.

    Ties go to the earliest answer group, and within a group to the earliest
    candidate, which keeps the fallback deterministic.
    """
    choices = []
    for q in range(len(bank)):
        groups = {}
        for i in range(4):
            if answers[q][i] is not None:
                groups.setdefault(answers[q][i], []).append(i)
        if not groups:
            choices.append(0)
            continue
        best = max(groups, key=lambda a: (len(groups[a]), -groups[a][0]))
        choices.append(groups[best][0])
    return choices


def choose_by_model(bank, answers, scores, beta):
    """Pick the answer group with the best mean score, then its best candidate."""
    choices = []
    for q in range(len(bank)):
        groups = {}
        for i in range(4):
            if answers[q][i] is not None:
                groups.setdefault(answers[q][i], []).append(i)
        if not groups:
            choices.append(int(np.argmax(scores[q])))
            continue

        def group_score(answer):
            members = groups[answer]
            return float(scores[q, members].mean()) + beta * np.log(len(members))

        best = max(groups, key=group_score)
        members = groups[best]
        best_local = members[int(np.argmax(scores[q, members]))]
        choices.append(int(best_local))
    return choices


# ---------------------------------------------------------------------------
# Stage 2: group comparator.
#
# Stage 1 scores candidates independently. The decision, however, is between
# answer groups, so a second linear model ranks whole groups using the stage-1
# scores, the group's support, its embedding cohesion and its text statistics.
# It is trained on pairwise differences (correct group minus wrong group), so a
# single weight vector scores any group and the best group wins.
# ---------------------------------------------------------------------------
GROUP_FEATURE_WIDTH = 11


def group_rows(bank, answers, scores, embeddings, q):
    """Features for every answer group of question ``q``, or None.

    Returns (matrix [n_groups, 11], list of member index lists). All features
    vary across the groups of a question; constants would cancel out in the
    pairwise training objective.
    """
    groups = {}
    for i in range(4):
        if answers[q][i] is not None:
            groups.setdefault(answers[q][i], []).append(i)
    if not groups:
        return None

    emb = np.asarray(embeddings[q], dtype=np.float64)
    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    centroid = unit.mean(axis=0)
    rows, members = [], []
    for answer, idx in groups.items():
        texts = [bank[q]['candidates'][i]['text'] for i in idx]
        group_scores = scores[q, idx]
        if len(idx) > 1:
            cohesion = float(np.mean([unit[i] @ unit[j]
                                      for x, i in enumerate(idx) for j in idx[x + 1:]]))
        else:
            cohesion = 0.0
        rows.append([
            len(idx) / 4.0,
            float(group_scores.mean()),
            float(group_scores.max()),
            float(group_scores.min()),
            float(np.mean([unit[i] @ centroid for i in idx])),
            cohesion,
            float(np.mean([np.log1p(len(t)) for t in texts])),
            float(np.mean([np.log1p(t.count('=')) for t in texts])),
            float(np.mean([float('\\boxed' in t) for t in texts])),
            float(np.mean([float('\\frac' in t) for t in texts])),
            float(np.mean([np.linalg.norm(emb[i]) for i in idx])),
        ])
        members.append(idx)
    return np.asarray(rows, dtype=np.float64), members


def comparator_params(params):
    """Return (mean, scale, coef) when the checkpoint carries a comparator."""
    if params is None or 'cmp_coef' not in params:
        return None
    mean = np.asarray(params['cmp_mean']).ravel()
    scale = np.asarray(params['cmp_scale']).ravel()
    coef = np.asarray(params['cmp_coef']).ravel()
    if coef.size != GROUP_FEATURE_WIDTH or mean.size != GROUP_FEATURE_WIDTH:
        return None
    return mean, scale, coef


def choose_by_comparator(bank, answers, scores, embeddings, mean, scale, coef):
    """Rank answer groups with the comparator, then take the best candidate."""
    choices = []
    for q in range(len(bank)):
        built = group_rows(bank, answers, scores, embeddings, q)
        if built is None:
            choices.append(int(np.argmax(scores[q])))
            continue
        rows, members = built
        strength = ((rows - mean) / scale) @ coef
        idx = members[int(np.argmax(strength))]
        choices.append(int(idx[int(np.argmax(scores[q, idx]))]))
    return choices


# ---------------------------------------------------------------------------
# Public interface.
# ---------------------------------------------------------------------------
def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    features, answers = feature_matrix(bank, arrays['embeddings'])
    params = load_checkpoint(checkpoint)

    if params is None:
        print('[submission] no checkpoint found; using answer-agreement voting')
        return choose_by_agreement(bank, answers)

    n, four, width = features.shape
    flat = features.reshape(-1, width)
    # Stored parameters may be 1-D vectors or single-element arrays depending on
    # how the checkpoint was written; ravel() makes the load independent of that.
    mean = np.asarray(params['mean']).ravel()
    scale = np.asarray(params['scale']).ravel()
    coef = np.asarray(params['coef']).ravel()
    if mean.size != width or coef.size != width:
        print('[submission] checkpoint does not match features; using answer-agreement voting')
        return choose_by_agreement(bank, answers)

    z = (flat - mean) / scale
    intercept = float(np.asarray(params['intercept']).ravel()[0])
    beta = float(np.asarray(params.get('beta', np.array([0.0]))).ravel()[0])
    scores = (z @ coef + intercept).reshape(n, four)

    comparator = comparator_params(params)
    if comparator is not None:
        return choose_by_comparator(bank, answers, scores,
                                    arrays['embeddings'], *comparator)
    return choose_by_model(bank, answers, scores, beta)
