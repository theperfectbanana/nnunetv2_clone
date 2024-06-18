class nnUNetDatasetWithDose(nnUNetDataset):
    def load_case(self, identifier: str):
        data, seg, properties = super().load_case(identifier)
        dose_data = np.load(join(self.folder, f"{identifier}_dose.npy"))
        dose_data = dose_data[np.newaxis, ...]  # Add channel dimension
        seg = np.concatenate((seg, dose_data), axis=0)
        return data, seg, properties
