"""
Visual Secret Sharing
to print on transparent paper
"""

import sys
import argparse
import numpy as np
import math
import os
import io
from imageio import imwrite
from array import array
from PIL import Image
import random
import time
import cv2
from pathlib import Path

random.seed(time.time())
# np.set_printoptions(threshold=sys.maxsize)
A = np.array([[1,0], [0,1]], dtype=np.uint8)
B = np.array([[0,1],[1,0]], dtype=np.uint8)
# ONES = [[1,1],[1,1]]
# ZEROS = [[0,0],[0,0]]
BOX =  [A.tolist(),B.tolist()]
L = 2 #enlarging factor

def image_to_BnW(fname):
	img = Image.open(fname)
	tmp = img.convert("L")
	return tmp
	
def image_to_bits(image):
	Matrix = np.array(image)
# 	print(Matrix)
	Bin_Matrix = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
	x,y = 0,0
	while y < image.size[0]:
		if Matrix[x][y] != 0:
			Bin_Matrix[x][y]=1
		x = (x+1) % image.size[1]
		if x == 0:
			y = (y+1) 
	return Bin_Matrix

def make_box(b):
	r = random.randint(0,1)
	share1 = BOX[r]
	if b == 0:
		share2 = share1
	else:
		share2=BOX[(r + 1 )%2]
	return (share1, share2)

	
def make_dual_shares(matrix,matrix2):
	share1 = np.zeros((matrix.shape[0]*L, matrix.shape[1]*L), dtype=np.uint8)
	share2 = np.zeros((matrix.shape[0]*L, matrix.shape[1]*L), dtype=np.uint8)
	share3 = np.zeros((matrix.shape[0]*L, matrix.shape[1]*L), dtype=np.uint8)
	x,y = 0,0
	i,j = 0,0
	while j < matrix.shape[1]:
		r=random.randint(0,1)
		A = BOX[r]
		if matrix[i][j] == 0:
			B = A
		else:
			B = BOX[(r + 1 )%2]
		if matrix2[i][j] == 0:
			C = A
		else: 
			C = BOX[(r + 1 )%2]
		share1[x][y]=A[0][0]
		share2[x][y]=B[0][0]
		share3[x][y]=C[0][0]
		share1[x+1][y]=A[1][0]
		share2[x+1][y]=B[1][0]
		share3[x+1][y]=C[1][0]
		share1[x+1][y+1]=A[1][1]
		share2[x+1][y+1]=B[1][1]
		share3[x+1][y+1]=C[1][1]
		share1[x][y+1]=A[0][1]
		share2[x][y+1]=B[1][0]
		share3[x][y+1]=C[1][0]
		i = (i+1) % matrix.shape[0]
		if i == 0:
			j = (j+1) 
		x = (x+2) % (matrix.shape[0]*2)
		if x == 0:
			y = (y+2) % (matrix.shape[1]*2)
	return (share1,share2,share3)
	
def make_three_shares(matrix):
	(share1, share2)=make_shares(matrix)
	
def make_shares(matrix):
	share1 = np.zeros((matrix.shape[0]*2, matrix.shape[1]*2), dtype=np.uint8)
	share2 = np.zeros((matrix.shape[0]*2, matrix.shape[1]*2), dtype=np.uint8)
	x,y = 0,0
	i,j = 0,0
	while j < matrix.shape[1]:
		(A,B) = make_box(matrix[i][j])
		share1[x][y]=A[0][0]
		share2[x][y]=B[0][0]
		share1[x+1][y]=A[1][0]
		share2[x+1][y]=B[1][0]
		share1[x+1][y+1]=A[1][1]
		share2[x+1][y+1]=B[1][1]
		share1[x][y+1]=A[0][1]
		share2[x][y+1]=B[1][0]
		i = (i+1) % matrix.shape[0]
		if i == 0:
			j = (j+1) 
		x = (x+2) % (matrix.shape[0]*2)
		if x == 0:
			y = (y+2) % (matrix.shape[1]*2)
	return (share1, share2)
	
def overlay(share1,share2):
	share = np.zeros((share1.shape[0], share1.shape[1]), dtype=np.uint8)
	x,y = 0,0
	while y < share1.shape[1]:
		if (share1[x][y]== 1 or share2[x][y] == 1):
			share[x][y] = 1
		x = (x+1) % share1.shape[0]
		if x == 0:
			y = (y+1) 
	return share
	


def make_image_from_matrix(matrix):
	tmp = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.uint8)
	x,y = 0,0
	while y < matrix.shape[1]:
		if matrix[x][y]!= 0:
			tmp[x][y] = 255
		x = (x+1) % matrix.shape[0]
		if x == 0:
			y = (y+1)
	return Image.fromarray(tmp)
	
def myparser():
    parser = argparse.ArgumentParser(
    description=('''\\
    This programs does
    --------------------------------
    '''))
    # Required File name argument
    parser.add_argument('file_name', type=str,
                    help='This is the file name')
    parser.add_argument('--extra', type=str, help='optional second file')
    parser.add_argument('--test', type=str, help='optional test mode')
    return parser
    
    
if __name__ == '__main__':
	parser = myparser()
	args = parser.parse_args()
	fname = args.file_name
	fname2 = args.extra
	if args.test:
		List = [[1,2,3],[4,5,6]]
		print(List.reverse())
	BnWimage = image_to_BnW(fname)
# 	print(BnWimage)
	Matrix = image_to_bits(BnWimage)
# 	print(Matrix)
	dimensions = Matrix.shape
	make_image_from_matrix(Matrix).save("Input1.png")

	if fname2:
		BnWimage2 = image_to_BnW(fname2)
		Matrix2 = image_to_bits(BnWimage2)
		make_image_from_matrix(Matrix2).save("Input2.png")
		(Share1,Share2,Share3) = make_dual_shares(Matrix,Matrix2)
		make_image_from_matrix(Share1).save("Share1.png")
		make_image_from_matrix(Share2).save("Share2.png")
		make_image_from_matrix(Share2).save("Share3.png")
		make_image_from_matrix(Share2).save("Share4.png")
		Reconstruct = overlay(Share1,Share2)
		make_image_from_matrix(Reconstruct).save("Overlay1+2.png")
		Reconstruct = overlay(Share1,Share3)
		make_image_from_matrix(Reconstruct).save("Overlay1+3.png")
		Reconstruct = overlay(Share2,Share3)
		make_image_from_matrix(Reconstruct).save("Overlay2+3.png")
		Reconstruct = overlay(Reconstruct,Share1)
		make_image_from_matrix(Reconstruct).save("Overlay1+2+3.png")
	else:
		(Share1,Share2) = make_shares(Matrix)
		make_image_from_matrix(Share1).save(f'{path}')
		make_image_from_matrix(Share2).save("Share_2.png")
		Reconstruct = overlay(Share1,Share2)
		make_image_from_matrix(Reconstruct).save("Overlay_1+2.png")
