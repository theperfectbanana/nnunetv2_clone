class CustomLoss(nn.Module):
    def __init__(self, seg_loss, dose_loss, weight_dose=1.0):
        super(CustomLoss, self).__init__()
        self.seg_loss = seg_loss
        self.dose_loss = dose_loss
        self.weight_dose = weight_dose

    def forward(self, segmentation_output, dose_output, target, dose_target):
        seg_loss = self.seg_loss(segmentation_output, target)
        dose_loss = self.dose_loss(dose_output, dose_target)
        return seg_loss + self.weight_dose * dose_loss


    def custom_loss_function(preds, target_masks, target_doses, alpha=0.5):
        # Assuming preds is a tuple of (pred_masks, pred_doses)
        pred_masks, pred_doses = preds

        # Calculate Dice loss for segmentation
        dice = dice_loss(pred_masks, target_masks)

        # Calculate MSE loss for dose prediction
        mse = F.mse_loss(pred_doses, target_doses)

        # Combine the losses
        total_loss = alpha * dice + (1 - alpha) * mse
        return total_loss