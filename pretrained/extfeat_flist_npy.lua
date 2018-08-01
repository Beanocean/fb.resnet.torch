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
npy4th = require 'npy4th'
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
  -- print(number_of_files, batch_size)
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
local list_file = arg[3]
local outprefix = arg[4]  -- .. '/' .. paths.basename(arg[3], paths.extname(arg[3]))
-- local outprefix = arg[4] .. '/' .. paths.basename(arg[3], paths.extname(arg[3]))
-- local h5file = hdf5.open(outprefix .. '.h5', 'w')

local dataset_filenames = {}

-- code reference: https://stackoverflow.com/a/11204889/4202137
-- http://lua-users.org/wiki/FileInputOutput
-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function load_lines_from(file)
  if not file_exists(file) then
    print("Not found " .. file)
    return {}
  end

  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

local videonames = load_lines_from(list_file)

for idx, frame_dir in pairs(videonames) do  -- iterate over video frame folders

  if idx % 100 == 0 then
    print("process: " .. idx .. '/' .. #videonames)
  end

  vname = paths.basename(frame_dir)
  -- get the list of frames
  local list_of_frames = {}
  for frame in lfs.dir(frame_dir) do
    if frame ~= '.' and frame ~= '..' then
      table.insert(list_of_frames, frame_dir..'/'..frame)
    end
  end

  if #list_of_frames >= 1 then  -- at least have one frame
    table.sort(list_of_frames, function (a, b) return a < b end)

    dataset_filenames[vname] = list_of_frames

    -- process one folder
    local features = forward_a_batch(bs, list_of_frames)
    -- h5file:write('/'..vname, features)
    local outfile = outprefix .. '/' .. vname .. '.npy'
    npy4th.savenpy(outfile, features)
    print('saved '..vname..' to ' .. outfile)
  else
    io.stderr:write(vname .. ' is empty \n')
  end

end

-- h5file:close()

-- local jsfile = io.open(outprefix .. '.json', 'w')
-- jsfile:write(cjson.encode(dataset_filenames))
-- jsfile:close()

