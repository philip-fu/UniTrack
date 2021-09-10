import json
import fire
import os
import cv2

def write_frame_imgs(video_file, image_folder, img_extension='.jpg'):
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      print('Read a new frame # {}: '.format(count), success)
      cv2.imwrite(os.path.join(image_folder, str(count).zfill(6) + img_extension), image)
      count += 1
      success,image = vidcap.read()
      
    print('In total {} frames saved to {}'.format(count-1, image_folder))
      


def get_detections_from_log(json_file, output_base_dir, video_file=None,
                            frame_rate=12, num_frames=None, image_w=1920, image_h=1080,
                            image_folder='img1', img_extension='.jpg'):
    """Parse log file from local regression test. Write to MOT format.
    """
    seq_name = os.path.basename(json_file).split('.')[0]
    output_base_dir = os.path.join(output_base_dir, seq_name)
    image_folder_full = os.path.join(output_base_dir, image_folder)
    os.makedirs(os.path.join(output_base_dir, 'det'), exist_ok = True)
    os.makedirs(os.path.join(output_base_dir, 'gt'), exist_ok = True)
    os.makedirs(image_folder_full, exist_ok = True)
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # write a sequence info file
    seq_info_filename='seqinfo.ini'
    with open(os.path.join(output_base_dir, seq_info_filename), 'w') as f:
        f.write('[Sequence]\n')
        f.write('name={}\n'.format(seq_name))
        f.write('imDir={}\n'.format(image_folder))
        f.write('frameRate={}\n'.format(frame_rate))
        f.write('seqLength={}\n'.format(num_frames if num_frames is not None else len(json_data['annotated_frames'])))
        f.write('imWidth={}\n'.format(image_w))
        f.write('imHeight={}\n'.format(image_h))
        f.write('imExt={}\n'.format(img_extension))
    

    # write detections
    with open(os.path.join(output_base_dir, 'det/det.txt'), 'w') as f:
        # this is for items
        for det in json_data['items']:
            frame_id = det['seq_num'] + 1 # should start with 1
            tracking_id = det['tracking_id']
            bb_left = det['xMin']
            bb_top = det['yMin']
            bb_width = det['xMax'] - det['xMin']
            bb_height = det['yMax'] - det['yMin']
            conf = det['confidence']

            f.write('{},{},{},{},{},{},{},-1,-1,-1\n'.format(
                frame_id,
                tracking_id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                conf
            ))
    
    with open(os.path.join(output_base_dir, 'det/det_hands.txt'), 'w') as f:
        # this is for hands
        for det in json_data['hands']:
            frame_id = det['seq_num'] + 1 # should start with 1
            tracking_id = det['tracking_id'] if det['tracking_id'] != 'None' else -1
            bb_left = det['xMin']
            bb_top = det['yMin']
            bb_width = det['xMax'] - det['xMin']
            bb_height = det['yMax'] - det['yMin']
            conf = 0 # set to 0 in gt to avoid evaluation

            f.write('{},{},{},{},{},{},{},-1,-1,-1\n'.format(
                frame_id,
                tracking_id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                conf
            ))
    
    # save frames
    if video_file is not None:
        write_frame_imgs(video_file, image_folder_full, img_extension)

    return True



if __name__ == '__main__':
    fire.Fire()