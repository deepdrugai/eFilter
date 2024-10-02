import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import TemporaryDirectory
from openbabel import pybel
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from efilter.utilities.logging import log
import signal


class DockingVina:
    def __init__(self, target):
        self.set_box_parameters(target)
        self.target = target
        self.vina_program = "utils_sac/docking/qvina02"
        self.receptor_file = f"utils_sac/docking/{target}.pdbqt"
        self.num_cpu = os.cpu_count()
        self.exhaustiveness = 1
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100

        # Create a temporary directory using tempfile
        self.temp_dir_context = TemporaryDirectory()
        self.temp_dir = self.temp_dir_context.name
        log.info(f"Temporary directory created: {self.temp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Clean up the temporary directory
        self.temp_dir_context.cleanup()
        log.info(f"Temporary directory {self.temp_dir} has been removed.")

    def set_box_parameters(self, target):
        target_boxes = {
            "fa7": ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
            "parp1": ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
            "5ht1b": ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
            "jak2": ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
            "braf": ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106)),
            "JNK1": ((9, 30, 23), (48, 46, 49)),
            "JNK2": ((-25, -50, 40), (52, 70, 60)),
            "JNK3": ((15, 10, 20), (35, 40, 40)),
        }
        self.box_center, self.box_size = target_boxes.get(target, ((84.194, 6.949, -7.081), (25, 25, 25)))

    def gen_3d(self, smi, ligand_mol_file):
        run_line = ["obabel", f"-:{smi}", "--gen3D", "-O", ligand_mol_file]
        subprocess.run(
            run_line, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=self.timeout_gen3d, universal_newlines=True, check=True
        )

    def docking(self, ligand_pdbqt_file, docking_pdbqt_file):
        run_line = (
            f"{self.vina_program} --receptor {self.receptor_file} --ligand {ligand_pdbqt_file} --out {docking_pdbqt_file} "
            f"--center_x {self.box_center[0]} --center_y {self.box_center[1]} --center_z {self.box_center[2]} "
            f"--size_x {self.box_size[0]} --size_y {self.box_size[1]} --size_z {self.box_size[2]} "
            f"--cpu {self.num_cpu} --num_modes {self.num_modes} --exhaustiveness {self.exhaustiveness}"
        )
        # run_line += f" --cpu 1 --num_modes {self.num_modes} --exhaustiveness {self.exhaustiveness}"

        subprocess.run(
            run_line.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=self.timeout_dock, universal_newlines=True, check=True
        )

    def parse_affinity(self, docking_pdbqt_file):
        affinity_list = []
        with open(docking_pdbqt_file, "r") as file:
            for line in file:
                if line.startswith("REMARK VINA RESULT:"):
                    tokens = line.strip().split()
                    if len(tokens) >= 4:
                        affinity_list.append(float(tokens[3]))
        return affinity_list

    def docking_task(self, smi_idx):
        smi, idx = smi_idx
        ligand_mol_file = os.path.join(self.temp_dir, f"ligand_{idx}.mol")
        ligand_pdbqt_file = os.path.join(self.temp_dir, f"ligand_{idx}.pdbqt")
        docking_pdbqt_file = os.path.join(self.temp_dir, f"dock_{idx}.pdbqt")
        try:
            self.gen_3d(smi, ligand_mol_file)
            ms = list(pybel.readfile("mol", ligand_mol_file))
            if not ms:
                raise ValueError(f"Failed to read molecule from {ligand_mol_file}")
            m = ms[0]
            m.write("pdbqt", ligand_pdbqt_file, overwrite=True)

            self.docking(ligand_pdbqt_file, docking_pdbqt_file)
            affinity_list = self.parse_affinity(docking_pdbqt_file)
            if not affinity_list:
                return idx, 99.9
            return idx, affinity_list[0]
        except subprocess.CalledProcessError as e:
            if e.returncode < 0:
                # Process was terminated by a signal
                signum = -e.returncode
                if signum == signal.SIGINT:
                    # Subprocess was terminated by SIGINT
                    raise KeyboardInterrupt
            # Other subprocess errors
            log.error(f"Subprocess error with idx {idx}, SMILES {smi}: {e}")
            return idx, 99.9
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            log.error(f"Error with docking for idx {idx}, SMILES {smi}: {e}")
            return idx, 99.9

    def predict(self, smiles_list):
        log.info("Starting the docking predictions.")
        total_tasks = len(smiles_list)
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_cpu) as executor:
            future_to_idx = {executor.submit(self.docking_task, (smi, idx)): idx for idx, smi in enumerate(smiles_list)}
            try:
                with tqdm(total=total_tasks, file=sys.stdout) as pbar:
                    for future in as_completed(future_to_idx):
                        idx, affinity = future.result()
                        results[idx] = affinity
                        pbar.update(1)
            except KeyboardInterrupt:
                log.warning("Docking process interrupted by user. Shutting down executor...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        # Collect the results in the order of SMILES
        affinity_list = [results.get(idx, 99.9) for idx in range(total_tasks)]
        return affinity_list


def reward_vina(affinity_list):
    """
    Calculate rewards based on affinity list.

    Args:
        affinity_list (list): List of affinity scores.

    Returns:
        numpy.ndarray: Array of rewards.
    """
    reward = -np.array(affinity_list)
    reward = np.clip(reward, 0, None)
    return reward


def generate_unique_filename(output_dir, base_filename, extension=".txt"):
    """
    Generate a unique filename to avoid overwriting existing files.

    Args:
        output_dir (Path): Directory to save the file.
        base_filename (str): Base name of the file.
        extension (str): File extension.

    Returns:
        Path: Unique file path.
    """
    output_path = output_dir / f"{base_filename}{extension}"
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{base_filename}-{counter}{extension}"
        counter += 1
    return output_path


def main():
    log.info("Starting the docking process.")
    directory = Path("utils_sac/docking/")
    # filename = "halicin_1k_mix_all"
    filename = "halicin_1k_mix"
    # filename = "halicin_1k_mix_randshuffle"
    input_file = directory / f"{filename}.smi"

    output_directory = directory / "output3"
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        with input_file.open("r") as file:
            smiles_list = [line.strip() for line in file if line.strip()]
        log.info(f"Loaded {len(smiles_list):,} SMILES from '{input_file}'.")
    except Exception as e:
        log.error(f"Failed to read SMILES from '{input_file}': {e}")
        sys.exit(1)

    targets = [
        "JNK1",
        "JNK2",
        "JNK3",
        "fa7",
    ]
    try:
        for target in targets:
            with DockingVina(target) as docking:
                start_time = time.time()
                log.info(f"Starting docking for target: '{docking.target}'.")
                affinity_list = docking.predict(smiles_list)
                reward = reward_vina(affinity_list)

                base_output_filename = f"{docking.target}_{filename}_reward"
                output_file = generate_unique_filename(output_directory, base_output_filename)

                np.savetxt(output_file, reward, delimiter="\t", fmt="%.1f")
                log.info(f"Rewards saved to: '{output_file}'.")

                docking_time = time.time() - start_time
                formatted_time = time.strftime("%Hh %Mm %Ss", time.gmtime(docking_time))
                log.info(f"{docking.target} docking time: {formatted_time}")
    except KeyboardInterrupt:
        log.warning("Process interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        log.error(f"An error occurred during docking: {e}")
        sys.exit(1)

    log.info("Docking process completed successfully.")


if __name__ == "__main__":
    main()
