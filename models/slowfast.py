from base_model import BaseModel
from pytorchvideo.models import create_slowfast


class SlowFast(BaseModel):
    def __init__(self, num_classes, hypparams):
        super(SlowFast, self).__init__(hypparams)
        self.model = create_slowfast(input_channels=(4,4), model_num_class=num_classes, slowfast_channel_reduction_ratio=8, model_depth=18,
                       stem_dim_outs=(64,8),#(64,6),
                        stem_conv_kernel_sizes=((1, 3, 3), (3, 3, 3)),
                        stem_conv_strides = ((1, 2, 2), (1, 2, 2)),
                        stem_pool_strides = ((1, 2, 2), (1, 2, 2)),
                        slowfast_conv_channel_fusion_ratio=2,
                        slowfast_fusion_conv_stride = (6,1,1), # nb of frames_slow divided by slowfast_fusion_conv_stride[0] has to be nb of frames fast
                        head_pool_kernel_sizes = ((6, 2, 2), (36, 2, 2)), # head_pool_kernel_sizes[0,0]*slowfast_fusion_conv_stride[0] = head_pool_kernel_sizes[1,0]
                        dropout_rate=hypparams['resnet_dropout']
                                    )

    def forward(self, x):
        # x needs to be a list containing [x_slow, x_fast]
        return self.model(x)
