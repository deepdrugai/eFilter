import os
from pathlib import Path

import numpy as np
from rdkit import Chem

# from rdkit.Chem import AllChem
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# import tensorflow as tf  # Add this line

from efilter.utilities import constants, moleculereader
from efilter.utilities.logging import log
from efilter.utilities.options import Options


# Function to convert SMILES to fingerprint
# def smiles_to_fp(smiles, n_bits=2048):
def smiles_to_fp(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return np.array(GetMorganFingerprintAsBitVect(mol, 2, n_bits))
    return np.zeros((n_bits,))  # Return a zero array if the SMILES is invalid


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

    # Get files
    mol_files = getFiles(options)

    if not mol_files:
        log.error(f"No files were found in {options.INPUT_PATH}).")
        return

    mol_files_string = ", ".join([str(m.name) for m in mol_files])
    log.info(f"{len(mol_files)} file{'s'[:len(mol_files) ^ 1]} to be processed: {mol_files_string}.")

    log.info(f"Writing csv output to: {options.OUTPUT_PATH}.")

    # Get molecules
    molecules = moleculereader.getMolecules(mol_files)
    molecules_string = ", ".join([str(m).split(" ")[0] for m in molecules])
    molecules = [m.getRDKitObject() for m in molecules]

    if not molecules:
        log.error(f"No molecules were found in {mol_files_string}.")
        return
    log.info(f"{len(molecules)} molecule{'s'[:len(molecules) ^ 1]} to be analyzed: {molecules_string}.")

    # RUN MODELS HERE
    # Example SMILES strings
    # smiles = ["C1=CC=CC=C1", "CC(C)C", "O=C(O)C(Br)Cl"]

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    log.info(f"Models directory: {models_dir}")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]

    results = {}

    log.debug(f"molecules found: {molecules}.")

    import tensorflow as tf
    from keras.api.models import Model, load_model
    import csv

    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    # Define the prediction function
    @tf.function(reduce_retracing=True)
    def predict_with_model(model, X_smiles):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        # intermediate_layer_model.predict(X_smiles)
        # TODO: Run XGBoost model here on intermediate_layer_model
        return model.predict(X_smiles)

    # Process each SMILES string
    for mol in molecules:
        smi = Chem.MolToSmiles(mol)
        X_smiles = smiles_to_fp(smi)

        log.info(f"X_smiles shape: {X_smiles.shape}")
        # X_smiles = X_smiles.reshape(1, -1)

        if X_smiles.ndim == 1:
            X_smiles = X_smiles.reshape(1, -1)
            log.warning(f"X_smiles re-shaped: {X_smiles.shape}")

        results[smi] = {}

        # if not isinstance(X_smiles, tf.Tensor):
        #     X_smiles = tf.convert_to_tensor(X_smiles)
        # log.info(f"X_smiles type: {type(X_smiles)}, shape: {X_smiles.shape}")

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.split("_")[-1].replace(".keras", "")

            if os.path.exists(model_path):
                log.info(f"Processing model: {model_name}")

                best_nn_model = load_model(model_path)

                # # Predict intermediate features
                # intermediate_layer_model = Model(inputs=best_nn_model.input, outputs=best_nn_model.layers[-2].output)
                # intermediate_layer_model.predict(X_smiles)

                # nn_predictions = best_nn_model.predict(X_smiles)
                # results[smi][model_name] = {"NN Prediction": nn_predictions.flatten()[0]}

                # Predict using the defined function
                # log.debug(predict_with_model)  # Should print <function predict_with_model at 0x...>
                nn_predictions = predict_with_model(best_nn_model, X_smiles)  # type: ignore
                results[smi][model_name] = nn_predictions.flatten()[0]

            else:
                results[smi][model_name] = "Model file not found"

    # Display or process results as needed
    for smi, preds in results.items():
        print(f"Predictions for SMILES {smi}:")
        for model, prediction in preds.items():
            print(f"  {model.split('_')[-1]}: {prediction}")

    # Extract the model names from the first molecule's predictions
    model_names = list(results[next(iter(results))].keys())

    # Define the output file path
    output_file = options.OUTPUT_PATH
    # "results.csv"

    # Create the directory structure if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Write the results to the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ["SMILES"] + model_names
        writer.writerow(header)

        # Write the data rows
        for smi, preds in results.items():
            row = [smi] + [preds[model] for model in model_names]
            writer.writerow(row)

    print(f"Results saved to {output_file}.")


if __name__ == "__main__":
    main()


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
