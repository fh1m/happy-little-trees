import shutil
import subprocess

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from happytrees.export import to_c
from happytrees.run import SEED, data


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
@pytest.mark.parametrize("depth", [2, 4, None])
def test_generated_c_agrees_with_sklearn_on_every_point(tmp_path, depth):
    Xtr, Xte, ytr, _ = data()
    clf = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(Xtr, ytr)
    X = np.vstack([Xtr, Xte]).astype(np.float32)

    src = tmp_path / "t.c"
    src.write_text(
        "#include <stdio.h>\n" + to_c(clf)
        + "int main(void){float x[2];while(scanf(\"%f %f\",&x[0],&x[1])==2)printf(\"%d\\n\",predict(x));return 0;}\n"
    )
    exe = tmp_path / "t"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(src)], check=True)
    out = subprocess.run([str(exe)], input="\n".join(f"{a:.9g} {b:.9g}" for a, b in X),
                         capture_output=True, text=True, check=True).stdout.split()

    assert list(map(int, out)) == list(clf.predict(X))
