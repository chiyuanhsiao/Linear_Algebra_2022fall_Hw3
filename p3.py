import cv2
import numpy as np
from p1 import gen_basis
from p2 import CosineTrans2d, InvCosineTrans2d
import os
import sys

'''
function : reconstruct
---
2D DCT -> coefficient -> inverse 2D DCT -> reconstructed image 
'''
def reconstruct(I):
  count = 0

  N = len(I)  
  B = gen_basis(N)
  A = CosineTrans2d(B, I)
  # TODO:

  # do inverse 2D DCT on "DCT_chopped" to reconstruct the grid, save as "reconstruct_I"
  reconstruct_I = InvCosineTrans2d(B, A)

  return reconstruct_I + 128

if __name__ == '__main__':
  im_path = sys.argv[1]
  output_path = sys.argv[2]

  # read image
  I = cv2.imread(im_path).astype(float)
  I -= 128.0

  reconstruct_I = reconstruct(I)

  # save reconstructed image
  cv2.imwrite(os.path.join(output_path, 'reconstructed.png'), reconstruct_I)
