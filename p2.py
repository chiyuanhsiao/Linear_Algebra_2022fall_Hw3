import cv2
import numpy as np
from p1 import gen_basis
import os
import sys

# input: Basis (B), 2D image array (I)
# output: coefficient in 2D (a)
def CosineTrans2d(B, I):
    # TODO
    # implement 2D DCT
    return

# input: Basis (B), coefficient in 2D (a)
# output: reconstructed image (I')
def InvCosineTrans2d(B, a):
    # TODO
    # implement 2D iDCT
    return

'''
function : compress_grid
---
2D DCT -> coefficient -> inverse 2D DCT -> reconstructed and compressed image 
'''
def compress_grid(I):
  count = 0
  N = len(I)
    
  # generate basis "B" and compute variable "DCT" coefficient of grid I  
  B = gen_basis(N)
  DCT = CosineTrans2d(B, I)
  
  # make a copy for DCT
  DCT_chopped = DCT.copy()
  # quantize DCT given pre-defined Quantization matrix and do rounding 
  DCT_quantized = np.around(DCT / Q_table)
  
  for x in range(N):
      for u in range(N):
          for r in range(3):
              # convert the remaining DCT value
              DCT_chopped[x, u, r] = DCT_quantized[x, u, r] * Q_table[x, u, r]
              # count the number of values in matrix that has been set to 0
              if DCT_chopped[x, u, r] == 0.0:
                  count += 1
            
  # do inverse 2D DCT on "DCT_chopped" to reconstruct the grid, save as "reconstruct_I"
  reconstruct_I = InvCosineTrans2d(B, DCT_chopped)

  return reconstruct_I + 128, count

if __name__ == '__main__':
  im_path = sys.argv[1]
  output_path = sys.argv[2]
  grid_sz = 8
  tot_count = 0 # count the number of value for compression
  Q_table_one = np.expand_dims(np.array(
    [[16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]]).astype(float), axis=2)
  Q_table = np.concatenate((Q_table_one, Q_table_one, Q_table_one), axis=2)
  
  # control quality factor
  QF = 70.0
  scale = 200.0 - 2 * QF
  scale /= 100.0
  Q_table *= scale
  # read image
  I = cv2.imread(im_path).astype('float')
  I -= 128.0
  hight = np.shape(I)[0]
  width = np.shape(I)[1]

  reconstruct_I = np.zeros((hight // grid_sz * grid_sz, width // grid_sz * grid_sz, 3))
  # crop original image to 8*8 grid and do JPEG compression for each grid
  for r in range(hight // grid_sz):
    for c in range(width // grid_sz):
      grid = I[r*grid_sz:r*grid_sz+grid_sz, c*grid_sz:c*grid_sz+grid_sz]
      reconstruct_chopped, count = compress_grid(grid)
      tot_count += count
      reconstruct_I[r*grid_sz:r*grid_sz+grid_sz, c*grid_sz:c*grid_sz+grid_sz, :] = reconstruct_chopped
  print(f'JPEG compression: only use {1 - tot_count / (hight * width * 3)} size of original image')
  # save reconstructed image
  cv2.imwrite(os.path.join(output_path, 'JPEG_reconstructed.png'), reconstruct_I)
