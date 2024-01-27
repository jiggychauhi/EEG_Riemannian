import mne
import moabb
from moabb.datasets.base import BaseDataset
import os
import glob
import zipfile
import io

URL = "https://zenodo.org/record/5055046/files/"

class Neuroergonomics2021Dataset(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),  # 15 participants
            sessions_per_subject=3,  # 3 sessions per subject
            events={'resting': 1, 'easy': 2, 'medium': 3, 'hard': 4},  # Event labels
            code='Neuroergonomics2021',
            interval=[0, 2],  # Epochs are 2-second long
            paradigm='motor imagery' 
        )

    def data_path(
            self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subjects:
            raise ValueError("Invalid subject number")

        # Check if the zip file exists in memory
        zip_url = f"{URL}P{subject:02}.zip"
        zip_file = moabb.download.fs_issue_request("GET", zip_url, {})

        # Extract the zip file in memory
        with zipfile.ZipFile(io.BytesIO(zip_file.content), 'r') as zip_ref:
            # List the contents of the zip file
            zip_contents = zip_ref.namelist()

            # Extract the contents into memory
            extracted_data = {file: zip_ref.read(file) for file in zip_contents}

        # Set the path for in-memory extraction
        path_folder = f"P{subject:02}"

        # Check if the folder already exists
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)

        # Save the extracted files into the folder
        for file, content in extracted_data.items():
            with open(os.path.join(path_folder, file), 'wb') as f:
                f.write(content)

        return [path_folder]

    def _get_single_subject_data(self, subject):
        """Load data for a single subject."""
        data = {}
        subject_path = f"data/subject_{subject:02d}/"  # Modify the path as needed

        for session in range(1, self.sessions_per_subject + 1):
            session_path = os.path.join(subject_path, f"session_{session}/eeg/")
            raw_files = glob.glob(os.path.join(session_path, '*.set'))

            raw_list = [mne.io.read_raw_eeglab(f, preload=True) for f in raw_files]
            # Concatenate all raw objects
            raw = mne.concatenate_raws(raw_list)

            # Load channel locations
            montage_file = os.path.join(session_path, 'get_chanlocs.txt')
            montage = mne.channels.read_montage(kind='standard_1005')
            raw.set_montage(montage)

            data[session] = raw

        return data

    def _get_all_subjects_data(self):
        """Load data for all subjects."""
        return {subject: self._get_single_subject_data(subject) for subject in self.subjects}

dataset = Neuroergonomics2021Dataset()

# Example usage with MOABB
from moabb.paradigms import MotorImagery
paradigm = MotorImagery()
X, y, metadata = paradigm.get_data(dataset)