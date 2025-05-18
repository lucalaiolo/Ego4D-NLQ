import numpy as np
import os
import av
import ffmpeg

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    
    :param container: PyAV container.
    :param indices: list of frame indices to decode.
    
    :return: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def prepare_clip(path):
    '''
    Load and preprocess the video clip with UID clip_uid (already trimmed to the segment of interest)
    so it's ready for inference by the Video-LLaVA model.


    :param path: path to the .mp4 file to process

    :return: processed clip (ready to be fed to the Video LLaVA model)
    '''
    container = av.open(path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    return read_video_pyav(container, indices)

def cut_clip(path, clip_uid, start_time, end_time):
    '''
    Cut the clip according to the input start and end times.

    :param path: path to the .mp4 files
    :param clip_uid: uid of the input clip
    :param start_time: start time of the cut video clip
    :param end_time: end time of the cut video clip

    :return: path to the cut video clip
    '''
    out_path = os.path.join(path, f"{clip_uid}_{start_time}_{end_time}.mp4")
    (
        ffmpeg
        .input(os.path.join(path, f"{clip_uid}.mp4"), ss=start_time, to=end_time)
        .output(out_path, c='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path

def build_prompt(query):
    '''
    Returns the prompt for the Video-LLaVA model.
    The prompt is simply "USER: <video>\nquery ASSISTANT:"

    :param query: the query used to build the prompt
    
    :return: the prompt
    '''
    prompt = f"USER: <video>\n{query} ASSISTANT:"
    return prompt