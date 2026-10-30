"""Fit the MP2 verifier checkpoint from the supplied training split.

The method itself lives in ``submission.py``; this script only learns the
linear verifier's weights and the standardisation statistics, then writes them
to ``runs/model_bundle/model.npz`` together with a small ``config.json``. It
never reads validation or test references.

Usage
-----
    python train.py                       # writes runs/model_bundle
    python train.py --output runs/model_bundle --C 0.01

Design notes
------------
* Labels are computed exactly as the official scorer does it: a candidate is
  positive when its parsed final answer equals the parsed reference answer.
* Training uses only questions whose four candidates do not all agree
  ("hard" groups). The hidden test bank is defined to contain at least two
  different valid parsed answers, so restricting training to non-unanimous
  groups keeps the training distribution closer to evaluation and stops the
  verifier from simply learning "unanimity is correct".
* ``C`` (inverse L2 strength) and ``beta`` (agreement bonus at prediction
  time) were selected on the validation split and are fixed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import submission
from common import load_inputs, read_jsonl, parse_answer

SEED = 0
FEATURE_WIDTH = 1024 + 13 + 6


def build_labels(bank, gold, arrays):
    """Return the [N, 4] design tensor, answers and binary correctness labels."""
    features, answers = submission.feature_matrix(bank, arrays['embeddings'])
    labels = np.zeros((len(bank), 4), dtype=np.int64)
    for q, (row, reference) in enumerate(zip(bank, gold)):
        if row['id'] != reference['id']:
            raise ValueError('Bank/reference misalignment at question %d' % q)
        target = parse_answer(reference['answer'])
        for i in range(4):
            labels[q, i] = int(answers[q][i] is not None and answers[q][i] == target)
    return features, answers, labels


def unanimous(row):
    """True when all four candidates of one question end in the same answer."""
    return all(a is not None and a == row[0] for a in row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/model_bundle'))
    parser.add_argument('--C', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    bank, arrays = load_inputs('train')
    gold = read_jsonl(Path('data/gsm8k_train.jsonl'))
    if len(gold) != len(bank):
        raise ValueError('Incomplete training references')

    features, answers, labels = build_labels(bank, gold, arrays)
    keep = np.array([not unanimous(row) for row in answers], dtype=bool)
    print('training questions: %d total, %d non-unanimous' % (len(bank), int(keep.sum())))

    rows = (np.where(keep)[0][:, None] * 4 + np.arange(4)).ravel()
    flat = features.reshape(-1, FEATURE_WIDTH)
    scaler = StandardScaler().fit(flat[rows])
    model = LogisticRegression(C=args.C, max_iter=5000, random_state=args.seed)
    model.fit(scaler.transform(flat[rows]), labels.reshape(-1)[rows])

    scores = (scaler.transform(flat) @ model.coef_.ravel() + model.intercept_[0]).reshape(-1, 4)
    choices = submission.choose_by_model(bank, answers, scores, args.beta)
    acc = 100.0 * np.mean([labels[q, choices[q]] for q in range(len(bank))])
    print('training accuracy (in-sample): %.2f%%' % acc)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / 'model.npz',
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        coef=model.coef_.ravel().astype(np.float64),
        intercept=np.array([model.intercept_[0]], dtype=np.float64),
        beta=np.array([args.beta], dtype=np.float64),
    )
    config = {
        'method': 'logistic verifier on embeddings + surface features + agreement features',
        'feature_width': FEATURE_WIDTH,
        'embedding_dim': 1024,
        'C': args.C,
        'beta': args.beta,
        'seed': args.seed,
        'train_questions': int(keep.sum()),
        'training_accuracy_in_sample': float(acc / 100.0),
        'selection': 'mean group score + beta*log(group_size), then best candidate',
        'validation_selected': True,
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    size = (args.output / 'model.npz').stat().st_size
    print('wrote %s (%.1f KiB) and config.json' % (args.output / 'model.npz', size / 1024))


if __name__ == '__main__':
    main()
