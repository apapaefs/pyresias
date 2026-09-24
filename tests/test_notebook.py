"""Execute the distributed tutorial, including its plot, in a fresh kernel."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager


REPO = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_notebook_runs_and_agrees_with_the_script(self):
        notebook = nbformat.read(REPO / 'pyresias_nb.ipynb', as_version=4)
        notebook.cells.append(nbformat.v4.new_code_cell('''
import pyresias_test as tutorial
tutorial.Q, tutorial.Qc = Q, Qc
tutorial.fixedScale, tutorial.scaleoption = fixedScale, scaleoption
tutorial.tMethod = tMethod
seed(12345)
expected = []
for _ in range(Nevolve):
    expected.extend(tutorial.Evolve(Q, Qc, tutorial.get_alphaS_over(Q, Qc)))
np.testing.assert_allclose(AllEmissions, expected, rtol=0, atol=0)
assert len(AllEmissions) > 0
# Reinitializing the coupling must still work after defining alphaS(t, z, ...).
aS = RunningCoupling(0.118, 91.1876)
assert np.isfinite(alphaS(Q*Q, .5, Qc, alphaS_over))
'''))
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND='Agg',
                       IPYTHONDIR=directory, JUPYTER_RUNTIME_DIR=directory)
            with patch.dict(os.environ, env):
                manager = AsyncKernelManager(kernel_name='python3',
                                             connection_file=str(Path(directory) / 'kernel.json'))
                manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                            '-f', '{connection_file}']
                client = NotebookClient(notebook, km=manager, timeout=120, allow_errors=False)
                client.execute(cwd=directory, env=env, cleanup_kc=True)
            output = Path(directory) / 'plots' / 'momentumfrac.pdf'
            self.assertTrue(output.read_bytes().startswith(b'%PDF'))


if __name__ == '__main__':
    unittest.main()
