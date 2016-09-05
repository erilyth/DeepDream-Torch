#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "math"
require "cutorch"
require "cunn"
require "cudnn"
require "qtwidget"
require "os"
require "loadcaffe"

print('Usage')
print('qlua deepdream.lua source_img layer_max iterations update_rate')

imgfile = arg[1]
layer_cut = arg[2]
iterations = arg[3]
update_rate = arg[4]

-- Set this from the input image first before calling pre_process
local Normalization = {mean = 0/255, std = 0/255}

w1 = qtwidget.newwindow(500, 500)
w2 = qtwidget.newwindow(500, 500)

function reducenet(net, layer)
	local network = nn.Sequential()
	for i=1,layer do
		network:add(net:get(i))
	end
	return network
end

-- Make sure Normalization is set
function pre_process(img)
	img_new = img:double()
	img_new:div(255.0)
	img_new:add(-Normalization.mean)
	img_new:div(Normalization.std)
	return img_new
end

function post_process(img)
	img_new = img * Normalization.std
	img_new:add(Normalization.mean)
	img_new:mul(255.0)
	return img_new
end

use_cuda = 1

full_model = loadcaffe.load('Models/VGG_S/deploy.prototxt', 'Models/VGG_S/VGG_CNN_S.caffemodel')
print(full_model)

netw = reducenet(full_model,layer_cut)
print(netw)

criterion = nn.MSECriterion()
if use_cuda == 1 then
	criterion = criterion:cuda()
	netw = netw:cuda()
end

input = image.load(imgfile,3,'byte')
Normalization.mean = torch.mean(input:float())/255.0
Normalization.std = torch.std(input:float())/255.0
print(Normalization.mean .. " " .. Normalization.std)

input = pre_process(input)

-- Generally networks use 3x244x244 images as inputs
input = image.scale(input,224,224)
image.display{image=(post_process(input)), win=w1}

if use_cuda == 1 then
	input = input:cuda()
end

for tt=1,iterations do
    -- Forward prop in the neural network
    local outputs_cur = netw:forward(input)
    -- Set the output gradients at the outermost layer to be equal to the outputs (So they keep getting amplified)
    local output_grads = outputs_cur
    local inp_grad = netw:updateGradInput(input,output_grads)
    -- Gradient ascent
    input:add(inp_grad:mul(update_rate/torch.abs(inp_grad):mean()))
    image.display{image=(post_process(input)), win=w2}
    print(tt)
end