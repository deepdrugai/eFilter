from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmarts

# from efilter.output.Mol2Writer import _sybyl_atom_type
from efilter.representation.Molecule import Molecule
from efilter.utilities import constants
from efilter.utilities.logging import log


def fileToString(file):
    """
    Read the entire contents of a file into a string

    @input: file -- valid path to a file
    """
    with open(file) as f:
        return f.read()


def to_mol(molPath):
    """Create Molecule object from file path (string)"""
    mol = getRDKitMolecule(molPath)
    return Molecule(mol, molPath.name)


def getRDKitMolecule(path):
    """
    Given a path object, return the corresponding RDKit molecule object
    WARNING: If file contains multiple molecules getRDKitMolecule only returns first mol.
    """
    content = fileToString(path)
    path = path if isinstance(path, Path) else Path(path)
    mol = convertToRDkit(content, path)
    return mol[0][1] if isinstance(mol, list) else mol


def convertToRDkit(contents, path):
    """
    Attempt to read and convert the input file into an RdKit.Mol object

    @input: contents -- string contents of a file
    @input: mol_file -- input molecule file name

    @output: rdkit_mol -- rdkit molecule object
    """

    extension = path.suffix
    log.debug(f"Running {path} as extension: {extension}.")

    mol = None

    if extension not in constants.ACCEPTED_FORMATS:
        log.error(f"Input file type with extension {extension} ({path.name}) not supported.")
        raise NotImplementedError

    if extension == constants.MOL2_FORMAT_EXT:
        mol = readMol2File(contents)
        if mol is None:
            log.error(f"Rdkit failed to process file {path.name}.")
            return None
        return mol

    if extension == constants.SMILES_FORMAT_EXT:
        from efilter.utilities import smilesreader

        return smilesreader.readSmilesFile(contents)

    if extension == constants.FASTA_FORMAT_EXT:
        mol = Chem.MolFromFASTA(contents)

    elif extension == constants.YAML_FORMAT_EXT:
        mol = Chem.MolFromHELM(contents)

    elif extension == constants.MOL_FORMAT_EXT:
        mol = Chem.MolFromMolBlock(contents)

    elif extension == constants.PDB_FORMAT_EXT:
        mol = Chem.MolFromPDBBlock(contents)

    elif extension == constants.SMARTS_FORMAT_EXT:
        mol = Chem.MolFromSmarts(contents)

    elif extension == constants.TPL_FORMAT_EXT:
        mol = Chem.MolFromTPLBlock(contents)

    if not mol:
        log.error(f"Molecule file ({path.name}) was not read in due to RDKit Error.")
        return None

    if path:
        log.warning(f"Input file type {extension} ({path.name}) will not preserve molecule SYBL atom types.")
        return mol

    return None


def readMol2File(contents):
    # Turn off rdkit error messages
    # from rdkit import RDLogger
    # RDLogger.DisableLog('rdApp.*')

    return (
        Chem.MolFromMol2Block(contents)  # fmt: skip
        or Chem.MolFromMol2Block(contents, removeHs=False, cleanupSubstructures=False)
        # or Chem.MolFromMol2Block(contents, sanitize=False)
        # or Chem.MolFromMol2Block(contents, sanitize=False, removeHs=False)
        # or Chem.MolFromMol2Block(contents, sanitize=False, removeHs=False, cleanupSubstructures=False)
    )


def getMolecules(files):
    """
    From the set of input files, acquire the corresponding Rdkit molecules.

    @input: The list of input files
    @output: Molecule objects (each containing an Rdkit.Mol object)

    USER ISSUE: WHAT if a file with multiple molecules is input?
    """
    mols = []

    for current_file in files:
        # get the contents of the file and the file type (extension) for processing
        file_contents = fileToString(current_file)

        # Attempt to interpret the molecule
        try:
            mol = convertToRDkit(file_contents, current_file)
        except Exception as e:
            log.error(f"Error reading file {current_file}: {e}")
            continue
        else:
            # To Mol2Block and Back to Mol Object (to get TriposAtomType)
            # mol = MolToMol2Block(mol)
            # mol = Chem.MolFromMol2Block(mol)
            if mol is None:
                log.warning(f"Molecule {current_file} empty.")
                continue
            if isinstance(mol, list):
                log.debug(f"{mol = }")
                mols += [
                    (
                        Molecule(setAtomTypes(mol), f"{current_file.stem}-{name}")
                        if name and current_file.stem != name
                        else Molecule(setAtomTypes(mol), f"{current_file.stem}")
                    )
                    for name, mol in mol
                ]
            else:
                mols += [Molecule(setAtomTypes(mol), current_file.name)]

    if not mols:
        log.error(f"No molecules generated from files list: {[x.name for x in files]}.")

    return mols


def setAtomTypes(mol):
    # Use _sybyl_atom_type to set atom types
    for idx, atom in enumerate(mol.GetAtoms()):
        mol.GetAtomWithIdx(idx).SetProp("_TriposAtomType", _sybyl_atom_type(atom))

    return mol


def _sybyl_atom_type(atom):
    """Asign sybyl atom type
    Reference #1: http://www.tripos.com/mol2/atom_types.html
    Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    """
    sybyl = None
    atom_symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    hyb = atom.GetHybridization() - 1  # -1 since 1 = sp, 2 = sp1 etc
    hyb = min(hyb, 3)
    degree = atom.GetDegree()
    aromtic = atom.GetIsAromatic()

    # define groups for atom types
    guanidine = "[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])"  # strict
    # guanidine = '[NX3]([!O])([!O])!:C!:[NX3]([!O])([!O])' # corina compatible
    # guanidine = '[NX3]!@C(!@[NX3])!@[NX3,NX2]'
    # guanidine = '[NX3]C([NX3])=[NX2]'
    # guanidine = '[NX3H1,NX2,NX3H2]C(=[NH1])[NH2]' # previous
    #

    if atomic_num == 6:
        if aromtic:
            sybyl = "C.ar"
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):
            sybyl = "C.cat"
        else:
            sybyl = "%s.%i" % (atom_symbol, hyb)
    elif atomic_num == 7:
        if aromtic:
            sybyl = "N.ar"
        elif _atom_matches_smarts(atom, "C(=[O,S])-N"):
            sybyl = "N.am"
        elif degree == 3 and _atom_matches_smarts(atom, "[$(N!-*),$([NX3H1]-*!-*)]") or _atom_matches_smarts(atom, guanidine):
            sybyl = "N.pl3"
        elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
            sybyl = "N.4"
        else:
            sybyl = "%s.%i" % (atom_symbol, hyb)
    elif atomic_num == 8:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 1 and _atom_matches_smarts(atom, "[CX3](=O)[OX1H0-]"):
            sybyl = "O.co2"
        elif degree == 2 and not aromtic:  # Aromatic Os are sp2
            sybyl = "O.3"
        else:
            sybyl = "O.2"
    elif atomic_num == 16:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 3 and _atom_matches_smarts(atom, "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]"):
            sybyl = "S.O"
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        elif _atom_matches_smarts(atom, "S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]"):
            sybyl = "S.o2"
        else:
            sybyl = "%s.%i" % (atom_symbol, hyb)
    elif atomic_num == 15 and hyb == 3:
        sybyl = "%s.%i" % (atom_symbol, hyb)

    if not sybyl:
        sybyl = atom_symbol
    return sybyl


def _atom_matches_smarts(atom, smarts):
    idx = atom.GetIdx()
    return any(idx in m for m in atom.GetOwningMol().GetSubstructMatches(MolFromSmarts(smarts)))
