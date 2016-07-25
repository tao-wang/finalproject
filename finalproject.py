import numpy as np
import cv2
import blend
import sys
import math
import getopt

DEBUG = False
last_smiles = []
last_eyes = []
CAPTURE = False

def get_unit_masks(mask):
	u_mask = mask[:,:,0].astype(np.float) / 255.0
	inv_mask = u_mask.copy() + 1.0
	inv_mask[inv_mask > 1.0] = 0.0
	return u_mask, inv_mask

def run_blend(img1, img2, mask):
	# if max(img1.shape[:2]) == 0:
	if 0 in img1.shape[:2]:
		return img1
	res_mask = cv2.resize(mask, img1.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
	u_mask, inv_mask = get_unit_masks(res_mask)
	res_img2 = cv2.resize(img2, img1.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
	new_layers = []
	for c in range(3):
		L = u_mask * img1[:,:,c] + inv_mask * res_img2[:,:,c]
		new_layers.append(L)
	new_img = cv2.merge(new_layers)
	# new_img = blend.blend(res_img2, img1, res_mask.astype(np.float)/255.0)
	return new_img

def best_eyes(eyes, face):
	(fx,fy,fw,fh) = face
	min_diff_y = sys.maxint
	best_eyes = []
	for i in xrange(len(eyes)):
		(x1,y1,w1,h1) = eyes[i]
		for j in xrange(i,len(eyes)):
			(x2,y2,w2,h2) = eyes[j]
			if (x1<fw/2 and x2>fw/2) or (x1>fw/2 and x2<fw/2):
				diff_y = abs(y1-y2)
				if diff_y < min_diff_y:
					min_diff_y = diff_y
					best_eyes = [eyes[i], eyes[j]]
	if len(best_eyes) > 0 and (best_eyes[0][0] > best_eyes[0][1]):
		best_eyes = best_eyes[::-1]
	return best_eyes

def best_smile(smiles, face):
	(fx,fy,fw,fh) = face
	min_distance = sys.maxint
	rect = (0,0,0,0)
	for (x,y,w,h) in smiles:
		dist = math.sqrt((x-2*fw/3)**2 + (y-2*fh/3)**2)
		if y >= fh/2 and dist < min_distance:
			min_distance = dist
			rect = (x,y,w,h)
	return rect

if __name__ == '__main__':
	try:
		opts, args = getopt.getopt(sys.argv[1:], "", ["debug"])
	except getopt.GetoptError:
		print 'finalproject.py [--debug]'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'finalproject.py [--debug]'
			sys.exit()
		elif opt == '--debug':
			DEBUG = True

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

	hat = cv2.imread('props/hat.png')
	hat_mask = cv2.imread('props/hat_mask.jpg')

	sunglasses = cv2.imread('props/sunglasses.png')
	sunglasses_mask = cv2.imread('props/sunglasses_mask.jpg')

	mustache = cv2.imread('props/mustache.png')
	mustache_mask = cv2.imread('props/mustache_mask.jpg')

	cap = cv2.VideoCapture(0)

	while True:
		ret, img = cap.read()
		# img = cv2.imread('portrait.jpg')
		img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		debug_img = img.copy()
		if CAPTURE:
			print "Original captured and saved"
			cv2.imwrite("out_original.jpg", debug_img)

		# find faces
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)

		for (x,y,w,h) in faces:
			if DEBUG:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

			if CAPTURE:
				print "Face captured"
				cv2.rectangle(debug_img,(x,y),(x+w,y+h),(255,0,0),2)

			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			# eyes
			eyes = eye_cascade.detectMultiScale(roi_gray)
			eye_pair = best_eyes(eyes, (x,y,w,h))
			if len(eye_pair) >= 2:
				last_eyes = np.array(eye_pair)
			mean_y, mean_height = 0, 0
			for (ex,ey,ew,eh) in last_eyes:
				if DEBUG:
					cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				if CAPTURE:
					print "Eye captured"
					cv2.rectangle(debug_img,(ex+x,ey+y),(ex+x+ew,ey+y+eh),(0,255,0),2)
				mean_y += ey + y
				mean_height += eh
			mean_y = mean_y/2
			mean_height = mean_height/2

			glasses_section = img[mean_y:mean_y+mean_height, x:x+w]
			img[mean_y:mean_y+mean_height, x:x+w] = run_blend(glasses_section, sunglasses, sunglasses_mask)

			# hat
			hat_height = 2*h/3
			y_hat = y - hat_height/2
			hat_section = img[y_hat:y_hat+hat_height, x:x+w]
			img[y_hat:y_hat+hat_height, x:x+w] = run_blend(hat_section, hat, hat_mask)

			# mustaches
			smiles = smile_cascade.detectMultiScale(roi_gray)
			last_smiles.insert(0, best_smile(smiles, (x,y,w,h)))
			if len(last_smiles) > 1:
				last_smiles.pop()
			(sx,sy,sw,sh) = np.mean(last_smiles, axis=0).astype(np.int)
			if DEBUG:
				cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),1)

			if CAPTURE:
				print "Smile captured"
				cv2.rectangle(debug_img,(sx+x,sy+y),(sx+x+sw,sy+y+sh),(0,0,255),1)

			roi_color[sy:sy+sh, sx:sx+sw] = run_blend(roi_color[sy:sy+sh, sx:sx+sw], mustache, mustache_mask)

			if CAPTURE:
				print "Filter and debug images saved"
				cv2.imwrite("out_filtered.jpg", img)
				cv2.imwrite("out_debug.jpg", debug_img)
		CAPTURE = False

		cv2.imshow('webcam', np.concatenate((img, debug_img)))
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('c'):
			print "Screen capture!"
			CAPTURE = True

	cap.release()
	cv2.destroyAllWindows()