import subprocess as sp

from paths import paths
from pathlib import Path 

if __name__ == '__main__':
    pdb_dir = Path(paths.kiba_pdb)
    for pdb_file in pdb_dir.glob('*.pdb'):
        sp.call(f'dpock {str(pdb_file)}', shell=True)