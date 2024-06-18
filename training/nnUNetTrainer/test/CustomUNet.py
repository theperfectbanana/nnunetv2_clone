class CustomUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, deep_supervision):
        super(CustomUNet, self).__init__()
        self.input_layer = nn.Conv3d(num_input_channels, initial_filters, kernel_size=3, padding=1)
        # Define other layers...
        self.output_layer_segmentation = nn.Conv3d(last_layer_filters, num_output_channels - 1, kernel_size=1)
        self.output_layer_dose = nn.Conv3d(last_layer_filters, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)
        # Define forward pass...
        segmentation_output = self.output_layer_segmentation(x)
        dose_output = self.output_layer_dose(x)
        return segmentation_output, dose_output
