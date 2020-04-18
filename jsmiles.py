import re
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem.Atom import GetTotalDegree, GetFormalCharge, GetTotalValence, IsInRing, GetIsAromatic

# from rdkit.Chem import AllChem

# xp = np


Chiral = {"CHI_UNSPECIFIED": 0, "CHI_TETRAHEDRAL_CW": 1, "CHI_TETRAHEDRAL_CCW": 2, "CHI_OTHER": 3}
Hybridization = {"UNSPECIFIED": 0, "S": 1, "SP": 2, "SP2": 3, "SP3": 4, "SP3D": 5, "SP3D2": 6, "OTHER": 7}

atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo

H_Vector = [0] * atomInfo
H_Vector[0] = 1

lowerReg = re.compile(r'^[a-z]+$')


def islower(s):
    return lowerReg.match(s) is not None


upperReg = re.compile(r'^[A-Z]+$')


def isupper(s):
    return upperReg.match(s) is not None



m = Chem.MolFromSmiles('C=C(C)[C@H]1[C@@H]2C(=O)O[C@H]1[C@H]1OC(=O)[C@@]34O[C@@H]3C[C@]2(O)[C@@]14C.CC(C)(O)[C@H]1'
                       '[C@@H]2C(=O)O[C@H]1[C@H]1OC(=O)[C@@]34O[C@@H]3C[C@]2(O)[C@@]14C')

n = 0
while n < m.GetNumAtoms() :
    if m.GetAtomWithIdx(n).GetSymbol() == 'H':
        feature = [1, 0, 0, 0, 0]
    elif m.GetAtomWithIdx(n).GetSymbol() == 'C':
        feature = [0, 1, 0, 0, 0]
    elif m.GetAtomWithIdx(n).GetSymbol() == 'O':
        feature = [0, 0, 1, 0, 0]
    elif m.GetAtomWithIdx(n).GetSymbol() == 'N':
        feature = [0, 0, 0, 1, 0]
    else:
        feature = [0, 0, 0, 0, 1]

    # feature.append(m.GetTotalNumHs() / 8) doesn't work
    # feature.append(m.GetTotalDegree() / 4)
    # feature.append(m.GetFormalCharge() / 8)
    # feature.append(m.GetTotalValence() / 8)
    # feature.append(m.IsInRing() * 1)
    # feature.append(m.GetIsAromatic() * 1)

    f = [0] * (len(Chiral) - 1)
    if Chiral.get(str(m.GetChiralTag()), 0) != 0:
        f[Chiral.get(str(m.GetChiralTag()), 0)] = 1
    feature.extend(f)

    f = [0] * (len(Hybridization) - 1)
    if Hybridization.get(str(m.GetHybridization()), 0) != 0:
        f[Hybridization.get(str(m.GetHybridization()), 0)] = 1
    feature.extend(f)

    n += 1
