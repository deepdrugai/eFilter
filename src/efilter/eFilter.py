import os
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.models import Model
from rdkit import Chem
from rdkit.Chem import AllChem

from eFilter.src.efilter.utilities.logging import log
from eFilter.src.efilter.utilities.options import Options

from pathlib import Path

from eFilter.src.efilter.utilities import constants, moleculereader


def main():
    """
    eFilter

    (1) Parse input arguments
    (2) Read input files into Molecule objects
    (3) Fragment molecules
    (4) Output fragments as specified by the user

    """
    options = Options()
    # if not options.isRunnable():
    #     log.error(f'Command-line arguments failed to parse; execution of eMolFrag will stop.')
    #     return

    # Verify Tools and Parse Command Line

    # Get files
    mol_files = getFiles(options)

    if not mol_files:
        log.error(f"No files were found in {options.INPUT_PATH}).")
        return

    mol_files_string = ", ".join([str(m.name) for m in mol_files])
    log.info(f"{len(mol_files)} file{'s'[:len(mol_files) ^ 1]} to be processed: {mol_files_string}.")

    # Get molecules
    molecules = moleculereader.getMolecules(mol_files)
    molecules_string = ", ".join([str(m).split(" ")[0] for m in molecules])
    if not molecules:
        log.error(f"No molecules were found in {mol_files_string}.")
        return
    log.info(f"{len(molecules)} molecule{'s'[:len(molecules) ^ 1]} to be analyzed: {molecules_string}.")

    # RUN MODELS HERE
    # Example SMILES strings
    smiles = ["C1=CC=CC=C1", "CC(C)C", "O=C(O)C(Br)Cl"]

    models_dir = "src/efilter/models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]

    results = {}

    # Process each SMILES string
    for smi in smiles:
        X_smiles = smiles_to_fp(smi)
        X_smiles = X_smiles.reshape(1, -1)

        results[smi] = {}

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace(".keras", "")

            if os.path.exists(model_path):
                best_nn_model = tf.keras.models.load_model(model_path)

                # Predict intermediate features
                intermediate_layer_model = Model(inputs=best_nn_model.input, outputs=best_nn_model.layers[-2].output)
                intermediate_features = intermediate_layer_model.predict(X_smiles)

                nn_predictions = best_nn_model.predict(X_smiles)
                results[smi][model_name] = {"NN Prediction": nn_predictions.flatten()[0]}

            else:
                results[smi][model_name] = "Model file not found"

    # Display or process results as needed
    for smi, preds in results.items():
        print(f"Predictions for SMILES {smi}:")
        for model, prediction in preds.items():
            print(f"  {model}: {prediction}")

    # if len(brick_db) == len(linker_db) == len(fa_db) == 0:
    #     log.error(f"No files were generated from {mol_files}.")
    #     return


if __name__ == "__main__":
    main()


# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits))
    else:
        return np.zeros((n_bits,))  # Return a zero array if the SMILES is invalid


def getFiles(options):
    """
    From the input directory specified, acquire all applicable molecule files
    """
    folderPath = Path(options.INPUT_PATH)
    files = []
    bad_files = []

    # Non-existing directory means no files to process
    if not folderPath.exists():
        log.error(f"Input path {options.INPUT_PATH} does not exist.")
        return []

    # Path is not a directory
    if not folderPath.is_dir():
        log.info(f"Input path {options.INPUT_PATH} is not a directory. Did you mean: '{folderPath.parent}/'?")
        current_file = folderPath
        folderPath = folderPath.parent
        log.error(f"??? {current_file}")
        # if the file extension is not a supportedd format, add the file to the bad file list, otherwise add it to the file list
        extension = current_file.suffix
        if extension in constants.ACCEPTED_FORMATS:
            files.append(folderPath / current_file.name)
        else:
            bad_files.append(current_file)

        # Report unacceptable files
        if bad_files:
            log.warning(f'emolFrag2 only accepts the following formats {", ".join(constants.ACCEPTED_FORMATS)}')
            log.warning(f'The following files will be ignored: {", ".join([bf.name for bf in bad_files])}')

        return files
        # return []

    # grab each file with acceptable molecule extension
    for current_file in folderPath.iterdir():
        # if the file extension is not a supportedd format, add the file to the bad file list, otherwise add it to the file list
        extension = current_file.suffix
        if extension in constants.ACCEPTED_FORMATS:
            files.append(folderPath / current_file.name)
        else:
            bad_files.append(current_file)

    # Report unacceptable files
    if bad_files:
        log.warning(f'emolFrag2 only accepts the following formats {", ".join(constants.ACCEPTED_FORMATS)}.')
        log.warning(f'The following files will be ignored: {", ".join([bf.name for bf in bad_files])}.')

    return files
