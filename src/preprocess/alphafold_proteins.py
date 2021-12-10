from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import shutil
from tqdm import tqdm
from collections import OrderedDict as odict
from copy import deepcopy 
import numpy as np 
import itertools as it
from scipy.spatial.transform import Rotation

CWD = Path(__file__).parent 
test_dir = CWD / 'test'

@dataclass 
class Dbref:
    idCode: str
    chainID: str
    seqBegin: int
    seqEnd: int
    database: str
    dbAccession: str
    dbIdCode: str
    dbseqBegin: int
    dbseqEnd: int
    def __str__(self) -> str:
        return f'DBREF  {self.idCode:>4} {self.chainID:>1} {self.seqBegin:>4}  {self.seqEnd:>4}  {self.database:<6} {self.dbAccession:<8} {self.dbIdCode:<12} {self.dbseqBegin:>5}  {self.dbseqEnd:>5} \n'
    @classmethod
    def parse(cls, record, split=False):
        if split:
            _, idCode, chainID, seqBegin, seqEnd, database, dbAccession, dbIdCode, dbseqBegin, dbseqEnd = record.split()
            seqBegin = int(seqBegin)
            seqEnd = int(seqEnd)
            dbseqBegin = int(dbseqBegin)
            dbseqEnd = int(dbseqEnd)
            if len(dbAccession) > 8:
                dbAccession = dbAccession[:8]
            if len(dbIdCode) > 12:
                dbIdCode = dbIdCode[:12]
        else:
            idCode = record[7:11].strip()
            chainID = record[12]
            seqBegin = int(record[14:18])
            seqEnd = int(record[20:24])
            database = record[26:32].strip()
            dbAccession = record[33:41].strip()
            dbIdCode = record[42:54].strip()
            dbseqBegin = int(record[55:60])
            dbseqEnd = int(record[62:67]) 
        return cls(idCode, chainID, seqBegin, seqEnd, database, dbAccession, dbIdCode, dbseqBegin, dbseqEnd)

@dataclass 
class Atom:
    serial: int
    name: str
    resName: str
    chainID: str
    resSeq: int
    x: float
    y: float
    z: float
    coord: np.ndarray
    occupancy: float
    tempFactor: float
    element: str
    def __str__(self):
        name3 = self.name + (3 - len(self.name)) * ' '
        return f'ATOM  {self.serial:>5} {name3:>4} {self.resName:>3} {self.chainID}{self.resSeq:>4}    {self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}{self.occupancy:>6.2f}{self.tempFactor:>6.2f}          {self.element:>2}  \n'
    @classmethod
    def parse(cls, record):
        serial = int(record[6:11])
        name = record[12:16].strip()
        resName = record[17:20].strip()
        chainID = record[21]
        resSeq = record[22:26].strip()
        x = float(record[30:38])
        y = float(record[38:46])
        z = float(record[46:54])
        coord = np.array([x, y, z])
        occupancy = float(record[54:60])
        tempFactor = float(record[60:66])
        element = record[76:78].strip()
        return cls(serial, name, resName, chainID, resSeq, x, y, z, coord, occupancy, tempFactor, element)

@dataclass 
class Residue:
    resName: str
    chainID: str
    resSeq: int
    atoms: Dict[str, Atom]
    def __init__(self, atoms:Dict[str, Atom]):
        ca_atom = atoms['CA']
        ca_atom:Atom
        self.resName = ca_atom.resName
        self.chainID = ca_atom.chainID
        self.resSeq = ca_atom.resSeq
        self.atoms = atoms
    def __getitem__(self, name):
        return self.atoms[name]

@dataclass 
class Chain:
    chainID: str 
    residues: List[Residue]
    def __init__(self, residues:List[Residue]):
        first_residue = residues[0]
        self.chainID = first_residue.chainID
        self.residues = residues
    def __len__(self):
        return len(self.residues)
    def __iter__(self):
        return iter(self.residues)
    def __getitem__(self, i):
        return self.residues[i] 
    def get_seqres_record(self:int, serNum:int, resNames:List[str]) -> str:
        s = f'SEQRES {serNum:>3} {self.chainID} {len(self.residues):>4} '
        for resName in resNames:
            s += f' {resName:>3}'
        if len(resNames) < 13:
            s += ' ' * (4 * (13 - len(resNames)))
        return s + '\n'
    def get_seqres_records(self) -> str:
        s = ''
        n = len(self.residues)
        for i in range(n // 13):
            resNames = [residue.resName for residue in self.residues[13*i:13*(i+1)]]
            s += self.get_seqres_record(1 + i, resNames)
        if n % 13 != 0:
            resNames = [residue.resName for residue in self.residues[13*(n//13):]]
            s += self.get_seqres_record(1 + n // 13, resNames)
        return s 
    def get_atom_records(self) -> str:
        s = ''
        for residue in self.residues:
            for atom in residue.atoms.values():
                s += str(atom)
        return s
    def get_ter_record(self) -> str:
        last_atom = list(self.residues[-1].atoms.values())[-1]
        serial = last_atom.serial + 1
        return f'TER   {serial:>5}      {last_atom.resName} {last_atom.chainID}{last_atom.resSeq:>4} \n'

@dataclass
class ProteinInfo:
    header_lines: str
    middle_lines: str
    final_lines: str
    chains: Dict[str, Chain]
    dbrefs: Dict[str, Dbref] 
    def __str__(self):
        s = ''
        s += self.header_lines 
        for dbref in self.dbrefs.values():
            s += str(dbref)
        for chain in self.chains.values():
            s += chain.get_seqres_records()
        s += self.middle_lines 
        for chain in self.chains.values():
            s += chain.get_atom_records()
            s += chain.get_ter_record()
        s += self.final_lines
        return s 
    @classmethod
    def parse(cls, line_gen, dbref_split=True):
        header_lines = ''
        middle_lines = ''
        final_lines = ''
        chains = {}
        dbrefs = {}
        header_lines_finished = False
        middle_lines_finished = False
        for line in line_gen():
            line:str
            if line.startswith('DBREF '):
                dbref = Dbref.parse(line, split=dbref_split)
                dbrefs[dbref.chainID] = dbref
                header_lines_finished = True 
            elif line.startswith('SEQRES '):
                pass 
            elif line.startswith('ATOM '):
                if not middle_lines_finished:
                    middle_lines_finished = True
                atom = Atom.parse(line)
                if atom.chainID in chains:
                    chains[atom.chainID].append(atom)
                else:
                    chains[atom.chainID] = [atom]
            elif line.startswith('TER '):
                pass
            elif not header_lines_finished:
                header_lines += line
            elif not middle_lines_finished:
                middle_lines += line
            else:
                final_lines += line
        for chainID in chains:
            residues = []
            partial_residue = []  
            for atom in chains[chainID]:
                atom:Atom
                if len(partial_residue) == 0:
                    partial_residue.append((atom.name, atom))
                else:
                    last_atom = partial_residue[-1][1]
                    last_atom:Atom
                    if last_atom.resSeq != atom.resSeq:
                        residues.append(Residue(odict(partial_residue)))
                        partial_residue = []
                    partial_residue.append((atom.name, atom))

            residues.append(Residue(odict(partial_residue)))

            chains[chainID] = Chain(residues)
        return cls(header_lines, middle_lines, final_lines, chains, dbrefs)