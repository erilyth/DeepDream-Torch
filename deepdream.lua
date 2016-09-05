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
	img_new = img:float()
	img_new = img_new:div(255.0)
	img_new = img_new:add(-Normalization.mean)
	img_new = img_new:div(Normalization.std)
	return img_new
end

function post_process(img)
	img_new = img * Normalization.std
	img_new = img_new:add(Normalization.mean)
	img_new = img_new:mul(255.0)
	return img_new
end

use_cuda = 1

full_model = loadcaffe.load('Models/VGG_F/deploy.prototxt', 'Models/VGG_F/VGG_CNN_F.caffemodel')
print(full_model)

netw = reducenet(full_model,layer_cut)
print(netw)

criterion = nn.MSECriterion()
if use_cuda == 1 then
	criterion = criterion:cuda()
	netw = netw:cuda()
end

input = image.load(imgfile,3,'byte')
input_copy = image.load(imgfile,3,'byte')
Normalization.mean = torch.mean(input:float())/255.0
Normalization.std = torch.std(input:float())/255.0
print(Normalization.mean .. " " .. Normalization.std)

input = pre_process(input)
input_copy = pre_process(input_copy)

-- Generally networks use 3x244x244 images as inputs
input = image.scale(input,224,224)
input_copy = image.scale(input_copy,224,224)

image.display{image=(post_process(input)), win=w1}

local total_octaves = 8
local drop_scale = 1.25/1.4
cur_drop_scale = drop_scale

local octaves = {}
octaves[total_octaves] = input:float()
local base_size = input:size()
for j=total_octaves-1,1,-1 do
    local cur_octave = image.scale(octaves[j+1], math.floor(cur_drop_scale*base_size[2]), math.floor(cur_drop_scale*base_size[3]),'bicubic')
    cur_drop_scale = cur_drop_scale * drop_scale
    octaves[j] = cur_octave
end

local prev_change
local final_img

for oct=1,total_octaves do
	local cur_oct = octaves[oct]
	local cur_size = cur_oct:size()
	if oct > 1 then
		prev_change = image.scale(prev_change, cur_size[2], cur_size[3], 'bicubic')
		cur_oct:add(prev_change)
	end
	cur_oct = image.scale(cur_oct, 224, 224, 'bicubic')
	cur_oct = cur_oct:cuda()
	for tt=1,iterations do
	    -- Forward prop in the neural network
	    local outputs_cur = netw:forward(cur_oct)
	    -- Set the output gradients at the outermost layer to be equal to the outputs (So they keep getting amplified)
	    local output_grads = outputs_cur
	    local inp_grad = netw:updateGradInput(cur_oct,output_grads)
	    -- Gradient ascent
	    cur_oct = cur_oct:add(inp_grad:mul(update_rate/torch.abs(inp_grad):mean()))
	    image.display{image=(post_process(cur_oct)), win=w2}
	    print(oct,tt)
	end
	cur_oct = cur_oct:float()
	cur_oct = image.scale(cur_oct, cur_size[2], cur_size[3], 'bicubic')
	prev_change = cur_oct - octaves[oct]
	final_img = cur_oct
end

image.display{image=(post_process(final_img)), win=w2}
print('Done processing')