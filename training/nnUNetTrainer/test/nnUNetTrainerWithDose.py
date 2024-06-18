class nnUNetTrainerWithDose(nnUNetTrainer):
    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = nnUNetDatasetWithDose(self.preprocessed_dataset_folder, tr_keys,
                                           folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                           num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDatasetWithDose(self.preprocessed_dataset_folder, val_keys,
                                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                            num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
            self.network = CustomUNet(self.num_input_channels, self.label_manager.num_segmentation_heads + 1, self.enable_deep_supervision).to(self.device)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            self.loss = CustomLoss(self._build_loss(), nn.MSELoss(), weight_dose=1.0)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. That should not happen.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        dose_target = data[:, -1, ...]  # Extract the dose target from the input
        data = data[:, :-1, ...]  # Remove the dose channel from the input

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        dose_target = dose_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            segmentation_output, dose_output = self.network(data)
            l = self.loss(segmentation_output, dose_output, target, dose_target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        dose_target = data[:, -1, ...]  # Extract the dose target from the input
        data = data[:, :-1, ...]  # Remove the dose channel from the input

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        dose_target = dose_target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            segmentation_output, dose_output = self.network(data)
            l = self.loss(segmentation_output, dose_output, target, dose_target)

        return {'loss': l.detach().cpu().numpy()}

