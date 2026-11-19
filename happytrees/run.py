"""Grow a decision tree one level at a time and watch it start painting noise.

The 2023 script scored 100 % on one Iris split without testing how its
complexity affected generalisation. Here the canvas is two interleaved
half-moons; each label is flipped with probability 0.10 before splitting.

    python -m happytrees.run          # prints the table, writes assets/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

ASSETS = Path(__file__).resolve().parent.parent / "assets"
SEED = 11
# Two paints: sap green and Prussian blue.
CANVAS = ListedColormap(["#c9d9b3", "#b6c6dc"])
INK = ListedColormap(["#3f6b2a", "#1f3a5f"])


def data():
    X, y = make_moons(n_samples=600, noise=0.20, random_state=SEED)
    rng = np.random.default_rng(SEED)
    flip = rng.random(len(y)) < 0.10
    y = np.where(flip, 1 - y, y)
    return train_test_split(X, y, test_size=0.4, random_state=SEED)


def paint(ax, clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 500), np.linspace(-1.6, 2.1, 400))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=CANVAS, alpha=1.0)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=INK, s=7, linewidths=0)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    Xtr, Xte, ytr, yte = data()
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 10})

    depths = range(1, 21)
    train_acc, test_acc, leaves = [], [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        train_acc.append(clf.score(Xtr, ytr))
        test_acc.append(clf.score(Xte, yte))
        leaves.append(clf.get_n_leaves())

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(depths, train_acc, color="#1f3a5f", lw=2.2, label="training accuracy (what it memorised)")
    ax.plot(depths, test_acc, color="#3f6b2a", lw=2.2, label="test accuracy (what it learned)")
    ax.axhline(0.90, color="k", lw=0.8, alpha=0.35)
    ax.text(19.5, 0.907, "90 % ideal-signal expectation", ha="right", fontsize=8, alpha=0.7)
    ax.set_xlabel("max tree depth")
    ax.set_ylabel("accuracy")
    ax.set_xticks(list(depths)[::2])
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "learning_vs_memorising.png")
    plt.close(fig)

    # The palette knife: cost-complexity pruning, alpha picked by 5-fold CV on training data.
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(Xtr, ytr)
    alphas = path.ccp_alphas[:-1]
    cv = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=SEED), Xtr, ytr, cv=5).mean()
          for a in alphas]
    a_best = float(alphas[int(np.argmax(cv))])
    pruned = DecisionTreeClassifier(ccp_alpha=a_best, random_state=SEED).fit(Xtr, ytr)
    full = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)

    # Four canvases: too timid, pruned, busy, painting every accident.
    panels = [
        ("depth 1", DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xtr, ytr)),
        ("pruned by CV", pruned),
        ("depth 8", DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)),
        ("unlimited", full),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=150)
    for ax, (name, clf) in zip(axes, panels):
        paint(ax, clf, Xtr, ytr, f"{name} · {clf.get_n_leaves()} leaves · test {clf.score(Xte, yte):.2f}")
    fig.tight_layout()
    fig.savefig(ASSETS / "four_canvases.png")
    plt.close(fig)

    print("depth  leaves  train   test")
    for d, lv, a, b in zip(depths, leaves, train_acc, test_acc):
        if d in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
            print(f"{d:5d}  {lv:6d}  {a:5.3f}  {b:5.3f}")
    print(f"unpruned: {full.get_n_leaves()} leaves, test {full.score(Xte, yte):.3f}")
    print(f"pruned (ccp_alpha={a_best:.4f} by CV): {pruned.get_n_leaves()} leaves, test {pruned.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
