import copy
import csv
import glob
import json
import os
import fire
import cv2
import tqdm
# remove args due to security check. can't make this file to md since other files are calling the method from it.
# previous commit with args is
# https://jet-tfs.visualstudio.com/Kepler/_git/ap/commit/50fb2d05c493cd00ada25178c044de5d6a876f1d?refName=refs%2Fheads%2Fmaster
# id 50fb2d05c493cd00ada25178c044de5d6a876f1d
TEST_NAME = '4184_regular_v2'
BASE_DIR = '/home/jadran/projects/kepler/asset-protection/services/store/mis-scan-mvp/data/automated_tests_v10plus/'



def write_frame_imgs(video_file, image_folder, img_extension='.jpg'):
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      #print('Read a new frame # {}: '.format(count), success)
      cv2.imwrite(os.path.join(image_folder, str(count).zfill(6) + img_extension), image)
      count += 1
      success,image = vidcap.read()
      
    print('In total {} frames saved to {}'.format(count-1, image_folder))
    return count-1
      


def get_detections_from_log(json_file, output_base_dir, video_file=None,
                            frame_rate=12, image_w=1920, image_h=1080,
                            image_folder='img1', img_extension='.jpg', prefix=None):
    """Parse log file from local regression test. Write to MOT format.
    """
    seq_name = os.path.basename(json_file).split('.')[0]
    if prefix is not None:
        seq_name = prefix + seq_name
    
    output_base_dir = os.path.join(output_base_dir, seq_name)
    image_folder_full = os.path.join(output_base_dir, image_folder)
    os.makedirs(os.path.join(output_base_dir, 'det'), exist_ok = True)
    os.makedirs(os.path.join(output_base_dir, 'gt'), exist_ok = True)
    os.makedirs(image_folder_full, exist_ok = True)
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    

    # write detections
    with open(os.path.join(output_base_dir, 'det/det.txt'), 'w') as f:
        # this is for items
        for det in json_data['items']:
            frame_id = det['seq_num'] + 1 # should start with 1
            tracking_id = det['tracking_id'] if det['tracking_id'] is not None else -1
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
            tracking_id = det['tracking_id'] if det['tracking_id'] is not None else -1
            bb_left = det['xMin']
            bb_top = det['yMin']
            bb_width = det['xMax'] - det['xMin']
            bb_height = det['yMax'] - det['yMin']
            conf = det['confidence'] # set to 0 in gt to avoid evaluation

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
        num_frms = write_frame_imgs(video_file, image_folder_full, img_extension)


    # write a sequence info file
    seq_info_filename='seqinfo.ini'
    with open(os.path.join(output_base_dir, seq_info_filename), 'w') as f:
        f.write('[Sequence]\n')
        f.write('name={}\n'.format(seq_name))
        f.write('imDir={}\n'.format(image_folder))
        f.write('frameRate={}\n'.format(frame_rate))
        f.write('seqLength={}\n'.format(num_frms))
        f.write('imWidth={}\n'.format(image_w))
        f.write('imHeight={}\n'.format(image_h))
        f.write('imExt={}\n'.format(img_extension))
    

    return True


def main(local_test_dir, run_name, output_dir):
    log_dir = os.path.join(local_test_dir, 'runs', run_name,'logs')
    video_dir = os.path.join(local_test_dir, 'source_videos')
    run_logs = glob.glob(os.path.join(log_dir, '**/*.json'), recursive=True)

    for run_log in tqdm.tqdm(run_logs):
        print(run_log)
        lane_id = os.path.basename(os.path.dirname(run_log))
        store_id = os.path.basename(os.path.dirname(os.path.dirname(run_log)))
        prefix = "{}_{}_{}_".format(run_name, store_id, lane_id)
        video_filename = os.path.basename(run_log).replace('.json', '.mp4')
        source_video = os.path.join(video_dir, store_id, lane_id, video_filename)

        get_detections_from_log(json_file=run_log,
                                output_base_dir=output_dir, 
                                video_file=source_video,
                                prefix=prefix)



if __name__ == "__main__":
    fire.Fire(main)
