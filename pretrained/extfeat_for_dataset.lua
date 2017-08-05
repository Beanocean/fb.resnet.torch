--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--
--  The scripts can process a dataset like this:
--    ./dataset
--        |--video_00/
--        |     |--video_00-000001.jpeg
--        |     |--video_00-000002.jpeg
--        |     |-- ...
--        |--video_01/
--        |-- ...
--
-- USAGE
--
-- BATCH MODE
--          th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_FOLDERS] [OUTPUT_DIRECOTRY]
--
-- @author: beanocean
--  @email: beanocean@outlook.com
--   @date: 05 Aug 2017


require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'hdf5'
require 'lfs'
local cjson = require 'cjson'
local t = require '../datasets/transforms'


if #arg < 2 or tonumber(arg[2]) == nil then
  io.stderr:write('Usage (Batch mode)      : th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES] [OUTPUT_DIRECOTRY] \n')
  os.exit(1)
end


-- Load the model
if not paths.filep(arg[1]) then
  io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
  os.exit(1)
end
local model = torch.load(arg[1]):cuda()

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
  t.Scale(256),
  t.ColorNormalize(meanstd),
  t.CenterCrop(224),
}

-- process a batch of images
local function forward_a_batch(batch_size, list_of_filenames)

  local number_of_files = #list_of_filenames
  print(number_of_files, batch_size)
  if batch_size > number_of_files then batch_size = number_of_files end
  local features

  for i=1,number_of_files,batch_size do
    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform 

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do 
      img_name = list_of_filenames[i+j-1] 

      if img_name  ~= nil then
        image_count = image_count + 1
        local img = image.load(img_name, 3, 'float')
        img = transform(img)
        img_batch[{j, {}, {}, {} }] = img
      end
    end

    -- if this is last batch it may not be the same size, so check that
    if image_count ~= batch_size then
      img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

    -- Get the output of the layer before the (removed) fully connected layer
    local output = model:forward(img_batch:cuda()):squeeze(1)


    -- this is necesary because the model outputs different dimension based on size of input
    if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 

    if not features then
      features = torch.FloatTensor(number_of_files, output:size(2)):zero()
    end
    features[{ {i, i-1+image_count}, {}  } ]:copy(output)
  end

  return features

end


local bs = tonumber(arg[2])  -- batch_size
local dir_path = arg[3]
local outprefix = arg[4] .. '/' .. paths.basename(arg[3], paths.extname(arg[3]))
local h5file = hdf5.open(outprefix .. '.h5', 'w')

local dataset_filenames = {}

for vname in lfs.dir(dir_path) do  -- iterate over video frame folders
  if vname ~= '.' and vname ~= '..' then

    -- get the list of frames
    local subdir = dir_path .. '/' .. vname
    local list_of_frames = {}
    for frame in lfs.dir(subdir) do
      if frame ~= '.' and frame ~= '..' then
        table.insert(list_of_frames, subdir..'/'..frame)
      end
    end

    if #list_of_frames >= 1 then  -- at least have one frame
      table.sort(list_of_frames, function (a, b) return a < b end)
      dataset_filenames[vname] = list_of_frames

      -- process one folder
      local features = forward_a_batch(bs, list_of_frames)
      h5file:write('/'..vname, features)
      print('saved '..vname..' to ' .. outprefix .. '.h5')
    else
      io.stderr:write(vname .. ' is empty \n')
    end

  end
end

h5file:close()

local jsfile = io.open(outprefix .. '.json', 'w')
jsfile:write(cjson.encode(dataset_filenames))
jsfile:close()

