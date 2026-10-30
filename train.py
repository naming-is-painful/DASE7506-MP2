"""Fit the MP2 verifier and group comparator from the training split.

Stage 1 (``submission.predict``) scores each candidate with a linear verifier.
Stage 2 ranks whole answer groups with a pairwise comparator. Both are fitted
here and written to ``runs/model_bundle/model.npz`` with a small config file.
Neither stage reads validation or test references.

The comparator is trained on *out-of-fold* stage-1 scores: five grouped folds
produce scores for questions the stage-1 model did not see, so the comparator
does not learn overconfident margins. Its own quality is estimated with a
cross-validation over those same folds and printed at the end.

Usage
-----
    python train.py                       # non-unanimous questions only
    python train.py --trainall            # every training question
    python train.py --output runs/model_bundle --C 0.01 --C-cmp 1.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import submission
from common import load_inputs, read_jsonl, parse_answer

SEED = 0
FEATURE_WIDTH = 1024 + 12 + 6
N_FOLDS = 5


def build_labels(bank, gold, arrays):
    """Return the design tensor, parsed answers and binary correctness labels."""
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


def fit_stage1(flat, labels, rows, C, seed):
    scaler = StandardScaler().fit(flat[rows])
    model = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    model.fit(scaler.transform(flat[rows]), labels.reshape(-1)[rows])
    return scaler, model


def stage1_scores(scaler, model, flat):
    """Scores in a fixed [N, 4] shape."""
    return (scaler.transform(flat) @ model.coef_.ravel() + model.intercept_[0]).reshape(-1, 4)


def out_of_fold_scores(flat, labels, answers, use_all, C, seed):
    """Stage-1 scores where each question was scored by a model that never saw it."""
    n = len(answers)
    selected = (np.ones(n, dtype=bool) if use_all
                else np.array([not unanimous(row) for row in answers], dtype=bool))
    oof = np.zeros((n, 4))
    fold_of = np.zeros(n, dtype=np.int64)
    folds = KFold(N_FOLDS, shuffle=True, random_state=seed)
    for k, (train_q, test_q) in enumerate(folds.split(np.arange(n))):
        fold_of[test_q] = k
        fit_q = [q for q in train_q if selected[q]]
        rows = (np.array(fit_q)[:, None] * 4 + np.arange(4)).ravel()
        test_rows = (test_q[:, None] * 4 + np.arange(4)).ravel()
        scaler, model = fit_stage1(flat, labels, rows, C, seed)
        oof[test_q] = stage1_scores(scaler, model, flat[test_rows]).reshape(len(test_q), 4)
    return oof, fold_of


def comparator_pairs(bank, answers, labels, scores, embeddings, fold_of):
    """Pairwise differences (correct group - wrong group) in both directions."""
    diffs, targets, folds = [], [], []
    for q in range(len(bank)):
        built = submission.group_rows(bank, answers, scores, embeddings, q)
        if built is None:
            continue
        rows, members = built
        correct = [bool(labels[q, idx].any()) for idx in members]
        if sum(correct) != 1:
            continue
        pos = int(np.argmax(correct))
        for j in range(len(members)):
            if j == pos:
                continue
            diffs.append(rows[pos] - rows[j]); targets.append(1); folds.append(fold_of[q])
            diffs.append(rows[j] - rows[pos]); targets.append(0); folds.append(fold_of[q])
    return np.asarray(diffs), np.asarray(targets), np.asarray(folds)


def fit_comparator(diffs, targets, C):
    scaler = StandardScaler().fit(diffs)
    model = LogisticRegression(C=C, max_iter=5000, fit_intercept=False)
    model.fit(scaler.transform(diffs), targets)
    return scaler, model


def comparator_cv(bank, answers, labels, scores, embeddings, fold_of, pair_folds,
                  diffs, targets, C):
    """Cross-validated selection accuracy of the comparator on training questions."""
    accuracies = []
    for k in range(N_FOLDS):
        train = pair_folds != k
        if targets[train].min() == targets[train].max():
            continue
        scaler, model = fit_comparator(diffs[train], targets[train], C)
        weight = model.coef_.ravel()
        test_q = np.where(fold_of == k)[0]
        hits = 0
        for q in test_q:
            built = submission.group_rows(bank, answers, scores, embeddings, q)
            if built is None:
                continue
            rows, members = built
            strength = ((rows - scaler.mean_) / scaler.scale_) @ weight
            idx = members[int(np.argmax(strength))]
            hits += int(answers[q][idx[0]] == correct_answer(answers, labels, q))
        accuracies.append(100.0 * hits / len(test_q))
    return float(np.mean(accuracies)), float(np.std(accuracies))


def correct_answer(answers, labels, q):
    for i in range(4):
        if labels[q, i]:
            return answers[q][i]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/model_bundle'))
    parser.add_argument('--C', type=float, default=0.01, help='stage-1 inverse L2 strength')
    parser.add_argument('--C-cmp', dest='C_cmp', type=float, default=1.0,
                        help='comparator inverse L2 strength')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='agreement bonus used only when no comparator is present')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--trainall', action='store_true',
                        help='Train on all questions, not just non-unanimous ones')
    args = parser.parse_args()

    bank, arrays = load_inputs('train')
    gold = read_jsonl(Path('data/gsm8k_train.jsonl'))
    if len(gold) != len(bank):
        raise ValueError('Incomplete training references')

    features, answers, labels = build_labels(bank, gold, arrays)
    keep = (np.ones(len(bank), dtype=bool) if args.trainall
            else np.array([not unanimous(row) for row in answers], dtype=bool))
    print('training questions: %d total, %d selected' % (len(bank), int(keep.sum())))

    rows = (np.where(keep)[0][:, None] * 4 + np.arange(4)).ravel()
    flat = features.reshape(-1, FEATURE_WIDTH)
    scaler, model = fit_stage1(flat, labels, rows, args.C, args.seed)
    scores = stage1_scores(scaler, model, flat)
    choices = submission.choose_by_model(bank, answers, scores, args.beta)
    acc = 100.0 * np.mean([labels[q, choices[q]] for q in range(len(bank))])
    print('stage-1 training accuracy (in-sample): %.2f%%' % acc)

    print('building out-of-fold stage-1 scores (%d folds)...' % N_FOLDS)
    oof, fold_of = out_of_fold_scores(flat, labels, answers, args.trainall, args.C, args.seed)
    base_choices = submission.choose_by_model(bank, answers, oof, args.beta)
    base_cv = 100.0 * np.mean([labels[q, base_choices[q]] for q in range(len(bank))])
    print('stage-1 out-of-sample accuracy: %.2f%%' % base_cv)

    diffs, targets, pair_folds = comparator_pairs(
        bank, answers, labels, oof, arrays['embeddings'], fold_of)
    print('comparator pairs: %d' % len(diffs))
    mean_cv, std_cv = comparator_cv(bank, answers, labels, oof, arrays['embeddings'],
                                    fold_of, pair_folds, diffs, targets, args.C_cmp)
    print('comparator CV accuracy: %.2f +- %.2f%%' % (mean_cv, std_cv))

    cmp_scaler, cmp_model = fit_comparator(diffs, targets, args.C_cmp)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / 'model.npz',
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        coef=model.coef_.ravel().astype(np.float64),
        intercept=np.array([model.intercept_[0]], dtype=np.float64),
        beta=np.array([args.beta], dtype=np.float64),
        cmp_mean=cmp_scaler.mean_.astype(np.float64),
        cmp_scale=cmp_scaler.scale_.astype(np.float64),
        cmp_coef=cmp_model.coef_.ravel().astype(np.float64),
    )
    config = {
        'method': 'linear verifier (stage 1) + pairwise group comparator (stage 2)',
        'feature_width': FEATURE_WIDTH,
        'group_feature_width': submission.GROUP_FEATURE_WIDTH,
        'embedding_dim': 1024,
        'C': args.C,
        'C_comparator': args.C_cmp,
        'beta_when_no_comparator': args.beta,
        'seed': args.seed,
        'folds': N_FOLDS,
        'train_questions': int(keep.sum()),
        'train_all': bool(args.trainall),
        'stage1_training_accuracy_in_sample': float(acc / 100.0),
        'stage1_out_of_sample_accuracy': float(base_cv / 100.0),
        'comparator_cv_accuracy': float(mean_cv / 100.0),
        'comparator_cv_std': float(std_cv / 100.0),
        'selection': 'rank answer groups by comparator, then best candidate in the group',
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    size = (args.output / 'model.npz').stat().st_size
    print('wrote %s (%.1f KiB) and config.json' % (args.output / 'model.npz', size / 1024))


if __name__ == '__main__':
    main()
