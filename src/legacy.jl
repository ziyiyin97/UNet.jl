export LegacyUnet

struct LegacyUnet
    conv_down_blocks
    conv_blocks
    up_blocks
end
  
@functor LegacyUnet
  
function LegacyUnet(channels::Int = 1, labels::Int = channels)
    conv_down_blocks = Chain(ConvDown(64,64),
                ConvDown(128,128),
                ConvDown(256,256),
                ConvDown(512,512))
  
    conv_blocks = Chain(UNetConvBlock(channels, 3),
           UNetConvBlock(3, 64),
           UNetConvBlock(64, 128),
           UNetConvBlock(128, 256),
           UNetConvBlock(256, 512),
           UNetConvBlock(512, 1024),
           UNetConvBlock(1024, 1024))
  
    up_blocks = Chain(UNetUpBlock(1024, 512),
          UNetUpBlock(1024, 256),
          UNetUpBlock(512, 128),
          UNetUpBlock(256, 64,p = 0.0f0),
          Chain(x->leakyrelu.(x,0.2f0),
          Conv((1, 1), 128=>labels;init=_random_normal)))									  
          LegacyUnet(conv_down_blocks, conv_blocks, up_blocks)
end
  
function (u::LegacyUnet)(x::AbstractArray)
    op = u.conv_blocks[1:2](x)
  
    x1 = u.conv_blocks[3](u.conv_down_blocks[1](op))
    x2 = u.conv_blocks[4](u.conv_down_blocks[2](x1))
    x3 = u.conv_blocks[5](u.conv_down_blocks[3](x2))
    x4 = u.conv_blocks[6](u.conv_down_blocks[4](x3))
  
    up_x4 = u.conv_blocks[7](x4)
  
    up_x1 = u.up_blocks[1](up_x4, x3)
    up_x2 = u.up_blocks[2](up_x1, x2)
    up_x3 = u.up_blocks[3](up_x2, x1)
    up_x5 = u.up_blocks[4](up_x3, op)
    tanh.(u.up_blocks[end](up_x5))
end
  