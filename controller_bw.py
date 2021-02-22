import cv2
import minimega
import requests
import sys, os

c_minimega = False

max_avail_bw = 50000 #Kpbs


def controller_bw(frame_size, cam_id, data_json, w1, w2, overlap_thresh=0.7):

	cam_num = int(cam_id)

	cam_pr_list_t = [0] * cam_num

	total_obj = 0
	total_length_cam = [0] * cam_num
	feature_cam = []
	for i in range(cam_num):
		# w1: static information: number of object counts
		total_obj = total_obj + data_json[i]['total_obj']

		# w2: motion information: displacenment length of all objects
		total_length_per_cam = 0
		# if(len(data_json[i][unique_obj_bbox]))
		feature_per_cam = []
		for j in data_json[i]['unique_obj_bbox']:
			total_length_per_cam = total_length_per_cam + data_json[i]['unique_obj_bbox'][j]['length']
			feature_per_cam.append(data_json[i]['unique_obj_bbox'][j]['feature'])
		total_length_cam[i] = total_length_per_cam
		feature_cam.append(feature_per_cam)

	# bias: find the occurances of commonly objects
	biases = [0] * cam_num
	for i in range(len(feature_cam)):
		for j in range(i+1, len(feature_cam)):
			for k in feature_cam[i]:
				for l in feature_cam[j]:
					if dist.eucliean(k, l) > overlap_thresh:
						# there is a common objects in the 2 cameras
						if(total_length_cam[i] > total_length_cam[j]):
							biases[i] = biases[i] + 1
							biases[j] = biases[j] - 1
						else:
							biases[i] = biases[i] - 1
							biases[j] = biases[j] + 1

	# step 2: optimize
	for i in range(cam_num):
		x1 = data_json[i]['total_obj'] / total_obj # static information
		x2 = total_length_cam[i] / sum(total_length_cam)
		if(sum(biases) != 0):
			bias = biases[i] / sum (biases)
		else:
			bias = 0
		cam_pr_list_t[i] = w1 * x1 + w2 * x2 - bias

	return cam_pr_list_t


if __name__ == '__main__':
	# get cam id:
	try:
		cam_id = str(sys.argv[1])
		print("Get camera number: ", cam_id)
	except:
		print("No number of camera present. Aborting")
		exit()
	nataddrs = []
	if c_minimega:
		mm = minimega.connect(debug=False)
		nataddrs = [nataddr[12] for nataddr in mm.vm_info()[0]['Tabular']]
		dataset_dir = '../others/dds/dataset/WildTrack/src/C'
	else:
		nataddrs = ['e127.0.0.1l']
		dataset_dir = '/home/waynewhbaaa/dds/dataset/WildTrack/src/C'

	for i in range(int(cam_id)):
    	# send init server request
		data = requests.get('http://%s:5002/init' % nataddrs[i][1:-1], params={'id': cam_id, "dataset_dir": dataset_dir})
		if(data.status_code == 200):
			print("Connection successful")
		else:
			print("Could not connect to camera. Aborting...")
			exit()

	# now run the experiment
	print("Start running the experiment!")

	# iterate 27 times because the data folder contains 401 images
	for i in range(27):
		data_from_cams = []
		start_id = i * 15
		for nataddr in nataddrs:
			data_json = requests.get('http://%s:5002/bbox' % nataddr[1:-1], params={'start_id': start_id}).json()
			data_from_cams.append(data_json)

		# run bw controller_bw
		cam_pr_list_t = controller_bw(15, cam_id, data_from_cams, 0.5, 0.5)

		index = 0
		for nataddr in nataddrs:
			bitrate = int(cam_pr_list_t[0] * max_avail_bw)
			vid_data = requests.get('http://%s:5002/video' % nataddr[1:-1], params={'start_id': start_id, 'bitrate': bitrate}).content

			# write to somewhere for save
			os.makedirs('/media/waynewhbaaa/Elements/myexp-temp/' + str(index), exist_ok=True)
			with open(os.path.join('/media/waynewhbaaa/Elements/myexp-temp/' + str(index), str(i) + ".mp4"), "wb") as f:
				f.write(vid_data)

			index = index + 1
