from rdkit import Chem
import re
import string
import numpy as np

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
    
def calc_atom_feature(molecule, i):  # i is just the index for the atoms in the molecules
    '''
    Convert Single Letter Character, i.e Atomic Symbol of a Smiles String into OneHotEncode 1D array length 42 long
    
    Inputs
    Single Letter Character of SMILES Strings
    
    Return
    OneHotEncode 1D array 42 vectors long with information on the type of atoms, Chirality, Hybridization and etc.
    
    Examples
    calc_atom_feature("Cl.Nc1ccc(-c2ccc3ccccc3n2)cc1","c")
    The first Carbon returns 
    [0, 0, 0, 0, 1, 0.125, 0.25, 0.0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    
    '''
    # first 21 smiles letters features

    m = Chem.MolFromSmiles(molecule)

    c = m.GetAtomWithIdx(i).GetSymbol()

    # One hot encode atoms

    if c == 'H':
        feature = [1, 0, 0, 0, 0]
    elif c == 'C':
        feature = [0, 1, 0, 0, 0]
    elif c == 'O':
        feature = [0, 0, 1, 0, 0]
    elif c == 'N':
        feature = [0, 0, 0, 1, 0]
    else:
        feature = [0, 0, 0, 0, 1]

    # One hot encode auxiliary features

    at = m.GetAtomWithIdx(i)

    feature.append(at.GetTotalNumHs() / 8)
    feature.append(at.GetTotalDegree() / 4)
    feature.append(at.GetFormalCharge() / 8)
    feature.append(at.GetTotalValence() / 8)
    feature.append(at.IsInRing() * 1)
    feature.append(at.GetIsAromatic() * 1)

    # Adding Chirality and Hybridization

    f = [0] * (len(Chiral) - 1)
    if Chiral.get(str(at.GetChiralTag()), 0) != 0:
        f[Chiral.get(str(at.GetChiralTag()), 0)] = 1
    feature.extend(f)

    f = [0] * (len(Hybridization) - 1)
    if Hybridization.get(str(at.GetHybridization()), 0) != 0:
        f[Hybridization.get(str(at.GetHybridization()), 0)] = 1
    feature.extend(f)

    # Add the last 21 vectors on to the array

    if islower(c):
        pass
    elif isupper(c):
        if c == "H":
            feature.extend(H_Vector)
        else:
            feature.extend([0] * 21)

    return feature
    
def calc_structure_feature(c):

    '''
    Convert Single Non-Letter Character, i.e SMILES Symbol denoting bonds and aromatic into OneHotEncode 1D array length 42 long
    
    Inputs
    Single non-letter character of SMILES Strings
    
    Return
    OneHotEncode 1D array 42 vectors long with information of what type of character it is and its function and etc.
    
    Examples
    calc_structure_feature("O=C1NC(=O)/C(=C/c2ccc3c(c2)OC(F)(F)O3)S1")
    The first "=" sign returns 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    * The 1 is toward the end at the 28th position.
    
    '''
    feature = [0] * structInfo

    flag = 0  # Placeholder assignment. Everything has flag = 0
    if c == '(':
        feature[0] = 1
        flag = 0
    elif c == ')':
        feature[1] = 1
        flag = 0
    elif c == '[':
        feature[2] = 1
        flag = 0
    elif c == ']':
        feature[3] = 1
        flag = 0
    elif c == '.':
        feature[4] = 1
        flag = 0
    elif c == ':':
        feature[5] = 1
        flag = 0
    elif c == '=':
        feature[6] = 1
        flag = 0
    elif c == '#':
        feature[7] = 1
        flag = 0
    elif c == '\\':
        feature[8] = 1
        flag = 0
    elif c == '/':
        feature[9] = 1
        flag = 0
    elif c == '@':
        feature[10] = 1
        flag = 0
    elif c == '+':
        feature[11] = 1
        flag = 1
    elif c == '-':
        feature[12] = 1
        flag = 1
    elif c.isdigit():  # There is no flag in our data set
        if flag == 0:
            if c in label:
                feature[20] = 1
            else:
                label.append(c)
                feature[19] = 1
        else:
            feature[int(c) - 1 + 12] = 1
            flag = 0

    feature = [0] * atomInfo + feature

    return feature

def one_hot_encoder(molecule):

    '''
    Combine the letter arrays and the non-letter arrays together into one 42*400 np array that 
    follows the order which the symbols appear
    
    Inputs
    SMILES Strings of the molecule
    
    Return
    42*400 2D nparray one hot encode arrays of letter and non-letter character
    
    '''
    
    OHE_array = np.array([])
    split_molecule = list(molecule)

    i = 0
    for n in range(len(molecule)):
        ch = split_molecule[n]
        if ch in list(string.ascii_lowercase):
            if ch == "i" or "a" or "e" or "l" or "s" or "r":   # Make sure symbols like Na or Si doesn't get counted twice
                continue
            else:
                b = np.array([calc_atom_feature(molecule, i)]) # b is a placeholder
                OHE_array = np.append (OHE_array, b)
                
            i += 1

        else:
            b = np.array([calc_structure_feature(molecule)])
            OHE_array = np.append (OHE_array, b)
            
    return OHE_array
